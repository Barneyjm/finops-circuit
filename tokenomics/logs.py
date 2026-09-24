"""Gateway logs in, conversations out: LiteLLM, Helicone, or OpenAI request/response pairs.

A gateway logs one row per API call, and a chat API is sent the whole history on every call, so
the rows of one thread are prefixes of each other. They are stitched back into one conversation:
a call continues a thread when its messages start with that thread's messages plus its last
reply. Each reply keeps the provider's own token counts as `usage`, so input and cached tokens
are measured, not estimated (see pricing.py).

    litellm    StandardLoggingPayload rows (a callback or the spend-log export): `id`, `model`,
               `messages`, `response`, `prompt_tokens`, `completion_tokens`, `startTime`,
               `metadata` (team and key aliases, requester_metadata), `request_tags`
    helicone   request rows from its query API: `request_id`, `request_created_at`,
               `request_model`, `request_body`, `response_body`, `prompt_tokens`,
               `completion_tokens`, `prompt_cache_read_tokens`, `request_properties`
    openai     {"request": <chat completions body>, "response": <ChatCompletion>}, one per line,
               with an optional "created_at" and "metadata"

What the gateway knows about who sent a call (team, key alias, request tags, custom
properties) becomes the conversation's declared tags, which win over inferred ones. User ids
are not carried over: a tag names a team or a product, not a person.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMATS = ("litellm", "helicone", "openai")


def _text(content: Any) -> str:
    """Message content as text: a string, or the text parts of a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"))
    return ""


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _time(v: Any) -> str | None:
    if v in (None, ""):
        return None
    if isinstance(v, int | float):
        return datetime.fromtimestamp(v, UTC).replace(tzinfo=None).isoformat(timespec="seconds")
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _cached(usage: dict[str, Any]) -> int:
    """Cached input tokens as OpenAI (prompt_tokens_details) or Anthropic (cache_read_input_tokens) report them."""
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return int(details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0)


def _reply(response: Any) -> str:
    r = _json(response) or {}
    if isinstance(r, dict) and r.get("choices"):
        return _text(r["choices"][0].get("message", {}).get("content"))
    if isinstance(r, dict) and isinstance(r.get("content"), list):  # an Anthropic message
        return _text(r["content"])
    return r if isinstance(r, str) else ""


def _tags(pairs: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in pairs.items() if v not in (None, "") and not isinstance(v, dict | list)}


def call(row: dict[str, Any], fmt: str) -> dict[str, Any]:
    """One logged API call in a common form: id, model, time, messages, reply, usage, tags."""
    if fmt == "litellm":
        meta = _json(row.get("metadata")) or {}
        response = _json(row.get("response")) or {}
        usage = (response.get("usage") if isinstance(response, dict) else None) or {}
        tags = _tags({"team": meta.get("user_api_key_team_alias"), "api_key": meta.get("user_api_key_alias")})
        tags |= _tags(_json(meta.get("requester_metadata")) or {})
        tags |= {f"tag:{t}": "true" for t in (_json(row.get("request_tags")) or [])}
        return {
            "id": str(row["id"]),
            "model": row["model"],
            "time": _time(row.get("startTime")),
            "messages": _json(row.get("messages")) or [],
            "reply": _reply(response),
            "usage": {"input_tokens": int(row.get("prompt_tokens") or 0), "output_tokens": int(row.get("completion_tokens") or 0), "cached_tokens": _cached(usage)},
            "tags": tags,
        }
    if fmt == "helicone":
        body, response = _json(row.get("request_body")) or {}, _json(row.get("response_body")) or {}
        return {
            "id": str(row["request_id"]),
            "model": row.get("request_model") or body.get("model", ""),
            "time": _time(row.get("request_created_at")),
            "messages": body.get("messages") or [],
            "reply": _reply(response),
            "usage": {"input_tokens": int(row.get("prompt_tokens") or 0), "output_tokens": int(row.get("completion_tokens") or 0), "cached_tokens": int(row.get("prompt_cache_read_tokens") or 0)},
            "tags": _tags(_json(row.get("request_properties")) or {}),
        }
    if fmt == "openai":
        req, response = _json(row["request"]), _json(row["response"])
        usage = response.get("usage") or {}
        return {
            "id": str(response.get("id") or row.get("id")),
            "model": response.get("model") or req.get("model", ""),
            "time": _time(row.get("created_at") or response.get("created")),
            "messages": req.get("messages") or [],
            "reply": _reply(response),
            "usage": {"input_tokens": int(usage.get("prompt_tokens", 0)), "output_tokens": int(usage.get("completion_tokens", 0)), "cached_tokens": _cached(usage)},
            "tags": _tags(_json(req.get("metadata")) or {}) | _tags(row.get("metadata") or {}),
        }
    raise ValueError(f"unknown log format {fmt!r}; one of {', '.join(FORMATS)}")


def detect(row: dict[str, Any]) -> str:
    if "request_body" in row or "request_id" in row:
        return "helicone"
    if "request" in row and "response" in row:
        return "openai"
    if "messages" in row and "startTime" in row:
        return "litellm"
    raise ValueError(f"cannot tell the log format from keys {sorted(row)[:8]}; pass --format")


def _key(messages: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(f"{m.get('role')}\x00{_text(m.get('content'))}\x01".encode())
    return h.hexdigest()


def conversations(rows: list[dict[str, Any]], fmt: str = "auto") -> list[dict[str, Any]]:
    """Log rows to conversations (this package's format), threads stitched, in time order."""
    calls = [call(r, detect(r) if fmt == "auto" else fmt) for r in rows]
    calls.sort(key=lambda c: c["time"] or "")
    threads: list[dict[str, Any]] = []
    open_: dict[str, tuple[int, int]] = {}  # key of messages + reply -> (thread, its length)
    lengths: set[int] = set()
    for c in calls:
        msgs = [m for m in c["messages"] if m.get("role") in ("system", "user", "assistant")]
        hit = next((open_.pop(k) for n in sorted(lengths, reverse=True) if n <= len(msgs) and (k := _key(msgs[:n])) in open_), None)
        if hit is None:
            i, new = len(threads), msgs
            threads.append({"id": c["id"], "model": c["model"], "timestamp": c["time"], "turns": [], "tags": dict(c["tags"])})
        else:
            i, new = hit[0], msgs[hit[1] :]
        t = threads[i]
        t["model"] = c["model"]  # the last call names the model
        # Assistant messages inside a prompt (few-shot examples, history from before the log) were
        # sent as input, not generated on this bill: `prompt` keeps pricing from counting them as replies.
        t["turns"] += [{"role": m["role"], "content": _text(m.get("content"))} | ({"prompt": True} if m["role"] == "assistant" else {}) for m in new]
        t["turns"].append({"role": "assistant", "content": c["reply"], "usage": c["usage"]})
        full = [*msgs, {"role": "assistant", "content": c["reply"]}]
        open_[_key(full)] = (i, len(full))
        lengths.add(len(full))
    return threads


def read(path: str | Path, fmt: str = "auto") -> list[dict[str, Any]]:
    """A .jsonl or .json (a list) log export, as conversations."""
    p = Path(path)
    text = p.read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if p.suffix == ".jsonl" else json.loads(text)
    if isinstance(rows, dict):  # an API response wrapping its rows
        rows = rows.get("data") or rows.get("rows") or []
    return conversations(rows, fmt)
