"""What happens after the circuit decides: the cost line, the value bucket, the actions.

Everything below is code over the circuit's gates and the price table. The model classifies;
it never prices anything and never writes the report.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from decision_circuits.types import answer_distributions

from .circuit import build_circuit
from .conversations import transcript
from .pricing import SMALL_MODEL, Cost, cost, price_of

CONTEXT_HEAVY_TURNS = 6  # a conversation this long, where resending earlier turns is this share of the bill,
CONTEXT_HEAVY_SHARE = 1 / 3  # is flagged: a summary or a fresh thread would cost less


@dataclass
class Finding:
    id: str
    model: str
    turns: int
    cost: Cost
    value: str  # the bucket, or "unclassified" when the circuit was not sure
    business: bool | None  # None: not sure
    actions: list[str]
    savings_usd: float  # what the actions with a dollar figure would have saved on this conversation
    audit: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False)


def state_for(conv: dict[str, Any], backend: Any) -> dict[str, Any]:
    """What the model reads: the transcript, one line per turn, and nothing about the model or
    the price, which must not sway what the conversation is judged to be worth."""
    state: dict[str, Any] = {"transcript": transcript(conv)}
    if type(backend).__name__ == "FakeBackend":
        state["expected_answers"] = conv.get("expected_answers", {})
    return state


def analyze(conv: dict[str, Any], backend: Any, *, model: str | None = None, circuit=None, prices: dict | None = None) -> Finding:
    """One conversation in: what it cost, what it was for, and what to change."""
    c = circuit or build_circuit()
    spend = cost(conv["model"], conv["turns"], prices)
    t0 = time.perf_counter()
    out = c.run(backend, state_for(conv, backend), model=model)
    ms = (time.perf_counter() - t0) * 1000
    g = out["gates"]

    value = g["value"]["value"] if g["value"]["outcome"] == "decided" else "unclassified"
    business = bool(g["business"]["value"]) if g["business"]["outcome"] == "decided" else None
    hold = bool(g["hold"]["value"])
    small = price_of(conv["model"], prices) <= price_of(SMALL_MODEL, prices)
    actions, savings = [], 0.0
    if g["policy"]["value"]:
        actions.append("policy: spend with no business use")
        savings += spend.usd
    elif g["downgrade"]["value"] and not small and not hold:
        actions.append(f"downgrade to {SMALL_MODEL}")
        savings += spend.usd - spend.usd_on_small_model
    if g["cache"]["value"] and not hold:
        actions.append("cache or template: a common request")  # no dollar figure: it depends on how often it repeats
    if hold:
        actions.append("hold: sensitive data, a person decides before it moves models or is cached")
    replies = sum(1 for t in conv["turns"] if t["role"] == "assistant")
    policy = bool(g["policy"]["value"])
    if not policy and replies >= CONTEXT_HEAVY_TURNS and spend.usd and spend.resent_usd / spend.usd >= CONTEXT_HEAVY_SHARE:
        actions.append(f"trim context: {spend.resent_usd / spend.usd:.0%} of the cost is resending earlier turns")
    if value == "unclassified" or business is None:
        actions.append("review: the circuit was not sure what this was for")

    last = getattr(backend, "last_response", None) or {}
    audit = {
        "backend": type(backend).__name__,
        "model": last.get("model") or model or getattr(backend, "model", None),
        "request_id": last.get("request_id"),
        "latency_ms": round(ms, 1),
        "answers": {k: _compact(v) for k, v in out["answers"].items()},
        "gates": {k: {"value": v["value"], "p": v.get("p"), "outcome": v["outcome"], "trace": v.get("trace")} for k, v in g.items()},
    }
    return Finding(conv["id"], conv["model"], replies, spend, value, business, actions, savings, audit)


def _compact(a: dict[str, Any]) -> Any:
    views = answer_distributions(a)
    if list(views) == [""]:
        d = views[""]
        return round(d["yes"], 3) if set(d) == {"yes", "no"} else {k: round(v, 3) for k, v in d.items()}
    return {k.strip("[]"): round(v.get("yes", max(v.values())), 3) for k, v in views.items()}


def report(findings: list[Finding]) -> dict[str, Any]:
    """The review: spend by value bucket, business or not, and what the actions would save."""
    total = sum(f.cost.usd for f in findings)
    by_value: dict[str, dict[str, float]] = defaultdict(lambda: {"conversations": 0, "usd": 0.0})
    for f in findings:
        by_value[f.value]["conversations"] += 1
        by_value[f.value]["usd"] += f.cost.usd
    savings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for f in findings:
        for a in f.actions:
            key = a.split(":")[0].split(" to ")[0]
            counts[key] += 1
        if any(a.startswith("policy") for a in f.actions):
            savings["policy"] += f.savings_usd
        elif any(a.startswith("downgrade") for a in f.actions):
            savings["downgrade"] += f.savings_usd
    out_tokens = sum(f.cost.output_tokens for f in findings)
    return {
        "conversations": len(findings),
        "spend_usd": round(total, 4),
        "business_share": round(sum(f.cost.usd for f in findings if f.business) / total, 3) if total else 0.0,
        "by_value": {
            k: {"conversations": int(v["conversations"]), "usd": round(v["usd"], 4), "share": round(v["usd"] / total, 3) if total else 0.0}
            for k, v in sorted(by_value.items(), key=lambda kv: -kv[1]["usd"])
        },
        "actions": dict(counts),
        "savings_usd": {k: round(v, 4) for k, v in savings.items()},
        "savings_share": round(sum(savings.values()) / total, 3) if total else 0.0,
        "output_tokens_measured_share": round(sum(f.cost.output_measured for f in findings) / out_tokens, 3) if out_tokens else 0.0,
    }
