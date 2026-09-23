"""What happens after the circuit decides: the tags, the cost line, the actions.

Everything below is code over the circuit's gates and the price table. The model tags; it never
prices anything and never writes the report.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from decision_circuits.types import answer_distributions

from .circuit import UNTAGGED, Taxonomy, build_child_circuit, build_circuit, load_taxonomy
from .conversations import transcript
from .pricing import Cost, PriceTable, cost, default_prices
from .tags import ADHOC

NOT_APPLICABLE = "n/a"  # a child tag whose parent's value has no options for it


def tag_keys(taxonomy: Taxonomy | None = None) -> list[str]:
    """The tags every conversation gets: app, then the taxonomy's, in its order."""
    return ["app", *(taxonomy or load_taxonomy()).tags]


CONTEXT_HEAVY_TURNS = 6  # a conversation this long, where resending earlier turns is this share of the bill,
CONTEXT_HEAVY_SHARE = 1 / 3  # is flagged: a summary or a fresh thread would cost less


@dataclass
class Finding:
    id: str
    model: str
    timestamp: str | None  # ISO date and time of the conversation, when the data has one
    turns: int
    cost: Cost
    tags: dict[str, str]  # app, every taxonomy tag ("untagged" where the circuit was not sure), and any declared extras
    tag_source: dict[str, str]  # per key: "declared" (the conversation's own metadata), "inferred" (the circuit), "code" (app)
    business: bool | None  # None: not sure
    actions: list[str]
    savings_usd: float  # what the actions with a dollar figure would have saved on this conversation, together
    action_savings: dict[str, float]  # the same, per action
    audit: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Finding:
        """A finding as `--save` wrote it (extra keys such as first_message are ignored)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names} | {"cost": Cost(**d["cost"]), "action_savings": d.get("action_savings", {})})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False)


def state_for(conv: dict[str, Any], backend: Any) -> dict[str, Any]:
    """What the model reads: the transcript, one line per turn, and nothing about the model or
    the price, which must not sway how the conversation is tagged."""
    state: dict[str, Any] = {"transcript": transcript(conv)}
    if type(backend).__name__ == "FakeBackend":
        state["expected_answers"] = conv.get("expected_answers", {})
    return state


def analyze(conv: dict[str, Any], backend: Any, *, model: str | None = None, circuit=None, taxonomy: Taxonomy | None = None, app: str = ADHOC, prices: PriceTable | None = None) -> Finding:
    """One conversation in: its tags, what it cost, and what to change. `app` comes from
    `tags.app_ids` over the whole set, since a template is only visible across conversations.

    Tags the conversation already carries (`conv["tags"]`, from an API key, a project, a
    header) win over inferred ones, as declared tags do on a cloud resource: the circuit fills
    the gaps, and declared tags the taxonomy does not know (team, cost_center) pass through.
    Environment in particular is rarely in the text and usually in the metadata."""
    tx = taxonomy or load_taxonomy()
    c = circuit or build_circuit(tx)
    table = prices or default_prices()
    spend = cost(conv["model"], conv["turns"], table)
    state = state_for(conv, backend)
    t0 = time.perf_counter()
    out = c.run(backend, state, model=model)
    g = out["gates"]
    declared = {str(k): str(v) for k, v in (conv.get("tags") or {}).items()}
    tags = {"app": app} | {t.name: str(g[t.name]["value"]) for t in tx.top} | declared
    subs: list[dict[str, Any]] = []
    for t in tx.tags.values():  # stage two: each child tag, asked only over its parent value's options
        if not t.parent or t.name in declared:
            continue
        parent_value = tags.get(t.parent, UNTAGGED)
        child = build_child_circuit(t, parent_value) if parent_value != UNTAGGED else None
        if child is None:
            tags[t.name] = UNTAGGED if parent_value == UNTAGGED else NOT_APPLICABLE
            continue
        subs.append(child.run(backend, state, model=model))
        tags[t.name] = str(subs[-1]["gates"][t.name]["value"])
    order = [*tag_keys(tx), *(k for k in declared if k not in tx.tags and k != "app")]
    tags = {k: tags.get(k, UNTAGGED) for k in order}
    source = {k: ("declared" if k in declared else "code" if k == "app" else "inferred") for k in order}
    ms = (time.perf_counter() - t0) * 1000

    business = bool(g["business"]["value"]) if g["business"]["outcome"] == "decided" else None
    actions, saved = decide(conv, tags, declared, {k: v["value"] for k, v in g.items()}, business, spend, tx, table)
    replies = sum(1 for t in conv["turns"] if t["role"] == "assistant")

    last = getattr(backend, "last_response", None) or {}
    answers = {k: _compact(v) for r in (out, *subs) for k, v in r["answers"].items()}
    gates = {k: {"value": v["value"], "p": v.get("p"), "outcome": v["outcome"], "trace": v.get("trace")} for r in (out, *subs) for k, v in r["gates"].items()}
    audit = {
        "backend": type(backend).__name__,
        "model": last.get("model") or model or getattr(backend, "model", None),
        "request_id": last.get("request_id"),
        "latency_ms": round(ms, 1),
        "answers": answers,
        "gates": gates,
    }
    return Finding(conv["id"], conv["model"], conv.get("timestamp"), replies, spend, tags, source, business, actions, sum(saved.values()), saved, audit)


def decide(
    conv: dict[str, Any], tags: dict[str, str], declared: dict[str, str], g: dict[str, Any], business: bool | None, spend: Cost, tx: Taxonomy, table: PriceTable
) -> tuple[list[str], dict[str, float]]:
    """The actions and what each saves, from the tags, the gate values and the cost line. Code
    only: a saved finding's gates are enough to decide again under a new price table."""

    def triggered(action: str) -> str | None:
        """The first `tag=value` that sets off a configured action, from the tags as they stand."""
        for tag, values in tx.actions.get(action, {}).items():
            if tags.get(tag) in values:
                return f"{tag}={tags[tag]}"
        return None

    # Declared values decide an action on their own; otherwise the gate over the model's probabilities does.
    hold_on = [k for k in tx.actions.get("hold", {}) if k in declared]
    hold = bool(triggered("hold")) if hold_on else bool(g.get("hold", False))
    dev_on = [k for k in tx.actions.get("dev_test", {}) if k in declared]
    dev_test = bool(triggered("dev_test")) if dev_on else bool(g.get("dev_test", False))
    here, small = table.of(conv["model"]), table.of(table.small_model)
    premium = here.input > small.input
    to_small = spend.usd - spend.usd_on_small_model
    actions: list[str] = []
    saved: dict[str, float] = {}
    moved = False  # a cheaper model was recommended: caching is then priced on that model
    if business is False:
        actions.append("policy: spend with no business use")
        saved["policy"] = spend.usd
    elif dev_test and premium and not hold:
        actions.append(f"dev/test on a premium model: use {table.small_model}")
        saved["dev/test on a premium model"] = to_small
        moved = True
    elif g["downgrade"] and premium and not hold:
        actions.append(f"downgrade to {table.small_model}")
        saved["downgrade"] = to_small
        moved = True
    if business is not False and not hold:
        on = small if moved else here
        caching = spend.resent_tokens * (on.input - on.cached_input) / 1e6 if on.cached_input is not None else 0.0
        if caching > 0 and spend.usd and caching / spend.usd >= 0.1:
            actions.append(f"prompt caching: {caching / spend.usd:.0%} of the cost is history the cache would serve")
            saved["prompt caching"] = caching
    if g["cache"] and not hold:
        actions.append("cache or template: a common request")  # no dollar figure: it depends on how often it repeats
    if hold:
        actions.append(f"hold: {triggered('hold') or 'sensitive data'}, a person decides before it moves models or is cached")
    replies = sum(1 for t in conv["turns"] if t["role"] == "assistant")
    no_cache = "prompt caching" not in saved and here.cached_input is None
    if business is not False and no_cache and replies >= CONTEXT_HEAVY_TURNS and spend.usd and spend.resent_usd / spend.usd >= CONTEXT_HEAVY_SHARE:
        actions.append(f"trim context: {spend.resent_usd / spend.usd:.0%} of the cost is resending earlier turns, and this model has no prompt cache")
    primary = tx.top[0].name  # the taxonomy's first tag is the one a conversation must have
    if tags.get(primary) == UNTAGGED or business is None:
        actions.append("review: the circuit could not tag what this was for")
    return actions, saved


def reprice(saved: dict[str, Any], conv: dict[str, Any], *, taxonomy: Taxonomy | None = None, prices: PriceTable | None = None) -> Finding:
    """A saved finding (as `--save` writes it) under a new price table: the cost line and the
    actions again, from the conversation and the gates the model already decided. No backend
    call: change prices.toml and rerun."""
    tx, table = taxonomy or load_taxonomy(), prices or default_prices()
    spend = cost(conv["model"], conv["turns"], table)
    tags = saved["tags"]
    declared = {k: v for k, v in tags.items() if saved.get("tag_source", {}).get(k) == "declared"}
    g = {k: v["value"] for k, v in saved["audit"]["gates"].items()}
    actions, saved_usd = decide(conv, tags, declared, g, saved["business"], spend, tx, table)
    return Finding.from_dict(saved | {"cost": asdict(spend), "actions": actions, "savings_usd": sum(saved_usd.values()), "action_savings": saved_usd})


def _compact(a: dict[str, Any]) -> Any:
    views = answer_distributions(a)
    if list(views) == [""]:
        d = views[""]
        return round(d["yes"], 3) if set(d) == {"yes", "no"} else {k: round(v, 3) for k, v in d.items()}
    return {k.strip("[]"): round(v.get("yes", max(v.values())), 3) for k, v in views.items()}


def _by_month(findings: list[Finding]) -> dict[str, float]:
    months: dict[str, float] = defaultdict(float)
    for f in findings:
        if f.timestamp:
            months[f.timestamp[:7]] += f.cost.usd
    return months


def report(findings: list[Finding], by: tuple[str, ...] | None = None, labels: dict[str, str] | None = None) -> dict[str, Any]:
    """The review: spend grouped by any tag keys (as a cloud bill is grouped by tags), tag
    coverage, and what the actions would save."""
    total = sum(f.cost.usd for f in findings)
    by = by or tuple(k for k in (findings[0].tags if findings else {}) if k != "app")[:1]

    def share(x: float) -> float:
        return round(x / total, 3) if total else 0.0

    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"conversations": 0, "usd": 0.0})
    for f in findings:
        key = " / ".join(f.tags.get(k, UNTAGGED) for k in by)
        groups[key]["conversations"] += 1
        groups[key]["usd"] += f.cost.usd
    keys = list(dict.fromkeys(k for f in findings for k in f.tags))
    coverage = {k: share(sum(f.cost.usd for f in findings if f.tags.get(k, UNTAGGED) != UNTAGGED)) for k in keys}
    savings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for f in findings:
        for a in f.actions:
            counts[a.split(":")[0].split(" to ")[0]] += 1
        for k, v in f.action_savings.items():
            savings[k] += v
    out_tokens = sum(f.cost.output_tokens for f in findings)
    result = {
        "conversations": len(findings),
        "spend_usd": round(total, 4),
        "by": list(by),
        "groups": {k: {"conversations": int(v["conversations"]), "usd": round(v["usd"], 4), "share": share(v["usd"])} for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["usd"])},
        "tag_coverage": coverage,  # share of spend with the tag set ("adhoc" counts as set for app)
        "declared_share": {k: share(sum(f.cost.usd for f in findings if f.tag_source.get(k) == "declared")) for k in keys},
        "fully_tagged_share": share(sum(f.cost.usd for f in findings if UNTAGGED not in f.tags.values())),
        "business_share": share(sum(f.cost.usd for f in findings if f.business)),
        "actions": dict(counts),
        "savings_usd": {k: round(v, 4) for k, v in savings.items()},
        "savings_share": share(sum(savings.values())),
        "output_tokens_measured_share": round(sum(f.cost.output_measured for f in findings) / out_tokens, 3) if out_tokens else 0.0,
        "tokens": {
            "input": sum(f.cost.input_tokens for f in findings),
            "cached": sum(f.cost.cached_tokens for f in findings),
            "output": out_tokens,
            "input_measured_share": share(sum(f.cost.usd for f in findings if f.cost.input_measured)),
        },
        "contracted_usd": round(sum(f.cost.contracted_usd for f in findings), 4),
        "by_month": {m: round(v, 4) for m, v in sorted(_by_month(findings).items())},
    }
    if labels and "app" in by:
        result["apps"] = labels
    return result
