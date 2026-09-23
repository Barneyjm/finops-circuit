"""The questions and the gates, built from a tag taxonomy. `taxonomy.toml` is the default policy.

Spend on LLM calls is allocated the way cloud spend is: by tags. Each tag in the taxonomy is a
question a model answers with probabilities and a gate that turns the answer into a value, or
"untagged" when the circuit is not sure, as an unlabelled resource is in a cloud bill. A tag
with a parent (subtask under task) is asked in a second request, only over the options for the
value its parent got. On top of the tags, the same request asks what a cost review needs: is it
work, how hard is it, would a small model do, is it a common request. Nothing here generates
text.
"""

from __future__ import annotations

import functools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_circuits import Circuit, Q, argmax

UNTAGGED = "untagged"
DEFAULT_TAXONOMY = Path(__file__).with_name("taxonomy.toml")

COMPLEXITY = [
    "Trivial: a lookup, a greeting, a one-line rewrite",
    "Routine: a standard email, a short explanation, a simple function",
    "Involved: several steps, a long document, code that has to fit a codebase",
    "Hard: careful multi-step reasoning, subtle bugs, expert judgment where errors are costly",
]


@dataclass
class Tag:
    name: str
    question: str
    options: dict[str, str] = field(default_factory=dict)  # a top-level tag's values
    parent: str | None = None
    by_parent: dict[str, dict[str, str]] = field(default_factory=dict)  # a child tag's values, per parent value
    min_confidence: float = 0.2


@dataclass
class Taxonomy:
    tags: dict[str, Tag]  # in the order they are asked and reported
    actions: dict[str, dict[str, list[str]]]  # action -> {tag: values that trigger it}

    @property
    def top(self) -> list[Tag]:
        return [t for t in self.tags.values() if not t.parent]

    def children(self, parent: str) -> list[Tag]:
        return [t for t in self.tags.values() if t.parent == parent]


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Read a taxonomy TOML (see the default for the format) and check it hangs together."""
    raw: dict[str, Any] = tomllib.loads(Path(path or DEFAULT_TAXONOMY).read_text())
    floor = float(raw.get("settings", {}).get("min_confidence", 0.2))
    tags: dict[str, Tag] = {}
    for name, spec in raw.get("tags", {}).items():
        if name == "app":
            raise ValueError("tag 'app' is reserved: it is set in code from the prompt template")
        parent = spec.get("parent")
        opts = spec.get("options", {})
        tag = Tag(name, spec["question"], parent=parent, min_confidence=float(spec.get("min_confidence", floor)))
        if parent:
            tag.by_parent = {k: dict(v) for k, v in opts.items()}
        else:
            tag.options = dict(opts)
            if len(tag.options) < 2:
                raise ValueError(f"tag {name!r} needs at least two options")
        tags[name] = tag
    for t in tags.values():
        if t.parent and t.parent not in tags:
            raise ValueError(f"tag {t.name!r} has parent {t.parent!r}, which is not a tag")
        if t.parent and tags[t.parent].parent:
            raise ValueError(f"tag {t.name!r}: only one level of parent is supported")
        if t.parent and (unknown := set(t.by_parent) - set(tags[t.parent].options)):
            raise ValueError(f"tag {t.name!r} has options for {sorted(unknown)}, which are not values of {t.parent!r}")
    actions = {a: {k: list(v) for k, v in when.items()} for a, when in raw.get("actions", {}).items()}
    for a, when in actions.items():
        for k, values in when.items():
            if k not in tags or tags[k].parent:
                raise ValueError(f"action {a!r} reads tag {k!r}, which is not a top-level tag")
            if unknown := set(values) - set(tags[k].options):
                raise ValueError(f"action {a!r}: {sorted(unknown)} are not values of {k!r}")
    return Taxonomy(tags, actions)


@functools.cache
def default_taxonomy() -> Taxonomy:
    """`taxonomy.toml`, read once."""
    return load_taxonomy()


def build_circuit(taxonomy: Taxonomy | None = None, v2: bool = False) -> Circuit:
    """Stage one: every top-level tag and the cost questions. `v2` adds `purpose` (locate: the
    line that shows what the conversation was for) for circuit v2 models."""
    tx = taxonomy or default_taxonomy()
    c = Circuit()
    for t in tx.top:
        c.choice(t.name, t.question, t.options)
    c.noul("work", "Is this being done for a job or a business, rather than for the person themselves?", true="For work or a business", false="Personal, school or entertainment")
    c.score("complexity", "How hard is what the assistant was asked to do?", COMPLEXITY)
    c.noul(
        "small_model_ok",
        "Could a small, cheap language model have produced answers just as good here?",
        true="Yes: common knowledge, simple writing or simple code",
        false="No: it needs strong reasoning, niche expertise or long careful output",
    )
    c.noul(
        "repeatable",
        "Is this a request many people would make in nearly the same words, so a template or a cached answer would serve it?",
        true="A common, standard request",
        false="Specific to this situation",
    )
    if v2:
        c.locate("purpose", "Which line shows best what the person was trying to get done?", none="no line makes the purpose clear")

    # ---- tags: the pick, or untagged when the circuit is not sure --------------------
    for t in tx.top:
        c.gate(t.name, argmax(t.name, min_confidence=t.min_confidence), on_uncertain="default", default=UNTAGGED)

    # ---- actions ----------------------------------------------------------------------
    c.gate("business", Q("work") >= 0.5, band=0.1, on_uncertain="escalate")
    c.gate("downgrade", (Q("small_model_ok") & ~Q("complexity")[3]) >= 0.65, band=0.1, on_uncertain="default", default=False)
    c.gate("cache", Q("repeatable") >= 0.7, band=0.1, on_uncertain="default", default=False)
    # The configured actions: "any of these tag values" as one OR over their probabilities.
    for action, when, tau, band, default in (("hold", tx.actions.get("hold"), 0.5, 0.15, True), ("dev_test", tx.actions.get("dev_test"), 0.6, 0.1, False)):
        refs = [Q(tag)[v] for tag, values in (when or {}).items() for v in values]
        if refs:
            expr = refs[0]
            for r in refs[1:]:
                expr = expr | r
            c.gate(action, expr >= tau, band=band, on_uncertain="default", default=default)
    return c


def build_child_circuit(tag: Tag, parent_value: str) -> Circuit | None:
    """Stage two for one child tag, given its parent's value; None when that value has no
    options for it."""
    options = tag.by_parent.get(parent_value)
    if not options:
        return None
    c = Circuit()
    if len(options) == 1:
        options = {**options, "other": "None of these"}
    c.choice(tag.name, tag.question.format(parent=parent_value.replace("_", " ")), options)
    c.gate(tag.name, argmax(tag.name, min_confidence=tag.min_confidence), on_uncertain="default", default=UNTAGGED)
    return c
