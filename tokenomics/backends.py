"""Which model answers the questions. One name, one line to change.

Every backend speaks to the circuit the same way, so swapping the model changes nothing
in circuit.py or agent.py. Keys come from the environment; see .env.example.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from decision_circuits.backends import SystemOne
from decision_circuits.types import answer_from_probabilities

BACKENDS = {
    "jev": "TypeSafe's hosted Jev (TYPESAFE_API_KEY)",
    "circuits": "decisioncircuits.com, open weights hosted (DECISIONCIRCUITS_API_KEY, or a free key is issued)",
    "local": "the same open weights on your own machine (S1_URL, S1_API_KEY)",
    "semif": "SemIf on the LangSmith gateway (LANGSMITH_API_KEY)",
    "openai": "a chat model read through logprobs (OPENAI_API_KEY); slower, not calibrated",
    "anthropic": "Claude asked for probabilities through tool use (ANTHROPIC_API_KEY); slower, not calibrated",
    "fake": "hand-written answers from samples/*.json, for tests and offline demos",
}


def _free_circuits_key() -> str:
    """decisioncircuits.com issues a key with no account; keep it in the environment after."""
    req = urllib.request.Request("https://api.decisioncircuits.com/v1/keys", data=b"{}", headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["key"]


class FakeBackend:
    """Answers come from the conversation file itself (`expected_answers`), so the circuit can be
    exercised with no model and no network. Missing questions get a flat distribution."""

    model = "fake"

    def __init__(self, answers: dict[str, dict[str, float]] | None = None):
        self.answers = answers or {}

    def answer(self, state: Any, questions: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
        table = self.answers or (state.get("expected_answers", {}) if isinstance(state, dict) else {})
        out = {}
        for qid, q in questions.items():
            probs = table.get(qid)
            if q["type"] == "multi":  # independent per option; unwritten options do not apply
                probs = {k: float((probs or {}).get(k, 0.05)) for k in q["criteria"]}
                out[qid] = {"type": "multi", "selected": sorted((k for k, p in probs.items() if p >= 0.5), key=lambda k: -probs[k]), "probabilities": probs}
                continue
            if q["type"] == "locate":  # {"transcript[2]": p, ..., "none": p}; the text comes from the state
                probs = probs or {"none": 1.0}
                sentences = state.get("transcript", []) if isinstance(state, dict) else []
                located = [{"path": k, "text": sentences[int(k.split("[")[1].rstrip("]"))], "probability": float(p)} for k, p in sorted(probs.items(), key=lambda kv: -kv[1]) if k != "none"]
                out[qid] = {"type": "locate", "located": located[:3], "none": float(probs.get("none", 0.0)), "confidence": 0.5}
                continue
            if probs is None:
                keys = ["yes", "no"] if q["type"] == "noul" else (list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))])
                probs = dict.fromkeys(keys, 1.0 / len(keys))
            out[qid] = answer_from_probabilities(q, probs)
        return out


V2_BACKENDS = ("circuits", "local", "fake")  # the ones that answer multi and locate
V1_MODELS = ("circuit-8b",)  # served by those backends on v1 weights: no multi or locate yet


def speaks_v2(backend: str, model: str | None) -> bool:
    """Whether to ask the v2 questions by default: a v2 backend, and not a v1 model on it."""
    return backend.lower() in V2_BACKENDS and not (model or "").startswith(V1_MODELS)


def pick_backend(name: str, model: str | None = None) -> Any:
    name = name.lower()
    if name == "jev":
        return SystemOne("https://api.typesafe.ai/v1/systemone", api_key=_need("TYPESAFE_API_KEY"), model=model or "jev-latest")
    if name == "circuits":
        key = os.environ.get("DECISIONCIRCUITS_API_KEY") or _free_circuits_key()
        return SystemOne("https://api.decisioncircuits.com/v1/systemone", api_key=key, model=model or "circuit-1.7b")
    if name == "local":
        return SystemOne(
            os.environ.get("S1_URL", "http://localhost:8901/v1/systemone"),
            api_key=os.environ.get("S1_API_KEY", "local"),
            model=model or "circuit-1.7b",
        )
    if name == "semif":
        return SystemOne("https://gateway.smith.langchain.com/v1/systemone", api_key=_need("LANGSMITH_API_KEY"), model=model or "semif-qwen3.5-4b")
    if name == "openai":
        from decision_circuits.backends import OpenAILogprobs

        return OpenAILogprobs(model=model or "gpt-4o-mini")
    if name == "anthropic":
        from decision_circuits.backends import Anthropic

        return Anthropic(model=model or "claude-sonnet-5")
    if name == "fake":
        return FakeBackend()
    raise SystemExit(f"unknown backend {name!r}; one of {', '.join(BACKENDS)}")


def _need(var: str) -> str:
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"{var} is not set; see .env.example")
    return v
