"""Gateway logs in, conversations out: LiteLLM, Helicone, or OpenAI or Anthropic request/response pairs.

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
    anthropic  {"request": <Messages API body>, "response": <Message>}, the same, for Claude.
               The top-level `system` becomes the first turn. Anthropic counts `input_tokens`
               net of the cache, so input is that plus cache reads and cache writes, and the
               writes (5-minute and 1-hour) are kept apart because they bill above input
    custom     any other JSON rows, read through a mapping file (--mapping) that names which
               field holds what; see `Mapping`

What the gateway knows about who sent a call (team, key alias, request tags, custom
properties) becomes the conversation's declared tags, which win over inferred ones. User ids
are not carried over: a tag names a team or a product, not a person.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMATS = ("litellm", "helicone", "openai", "anthropic", "custom")


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


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """A request's messages, an Anthropic top-level `system` (a string or text blocks) first."""
    system = _text(body.get("system"))
    return ([{"role": "system", "content": system}] if system else []) + (body.get("messages") or [])


def _anthropic_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Anthropic usage in this package's terms: input all of it, reads and writes of the cache apart."""
    read = int(usage.get("cache_read_input_tokens") or 0)
    split = usage.get("cache_creation") or {}  # the writes by TTL, where the API reports them
    w1h = int(split.get("ephemeral_1h_input_tokens") or 0)
    w = int(split.get("ephemeral_5m_input_tokens") or 0) if split else int(usage.get("cache_creation_input_tokens") or 0)
    return {
        "input_tokens": int(usage.get("input_tokens") or 0) + read + w + w1h,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cached_tokens": read,
        "cache_write_tokens": w,
        "cache_write_1h_tokens": w1h,
    }


def _usage(u: dict[str, Any]) -> dict[str, int]:
    """A provider's usage object, told apart by its keys: Anthropic's (input net of the cache), or
    OpenAI's (prompt/completion, or the Responses API's input/output, cached tokens included)."""
    if "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:
        return _anthropic_usage(u)
    return {
        "input_tokens": int(u.get("prompt_tokens") or u.get("input_tokens") or 0),
        "output_tokens": int(u.get("completion_tokens") or u.get("output_tokens") or 0),
        "cached_tokens": _cached(u),
    }


def _reply(response: Any) -> str:
    r = _json(response) or {}
    if isinstance(r, dict) and r.get("choices"):
        return _text(r["choices"][0].get("message", {}).get("content"))
    if isinstance(r, dict) and isinstance(r.get("content"), list):  # an Anthropic message
        return _text(r["content"])
    return r if isinstance(r, str) else ""


def _tags(pairs: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in pairs.items() if v not in (None, "") and not isinstance(v, dict | list)}


MISSING = object()


def _get(row: Any, path: str) -> Any:
    """The value at a dotted path ("request.messages", "choices.0.message"), JSON strings on the
    way parsed; MISSING where the path does not lead anywhere."""
    v = row
    for part in path.split("."):
        v = _json(v)
        if isinstance(v, dict) and part in v:
            v = v[part]
        elif isinstance(v, list) and part.lstrip("-").isdigit() and -len(v) <= int(part) < len(v):
            v = v[int(part)]
        else:
            return MISSING
    return _json(v)


class Mapping:
    """Where a custom log keeps each part of a call, as dotted paths into a row (a TOML file):

        model    = "model"                    # required
        messages = "request.messages"         # chat messages, or instead
        prompt   = "input"                    #   one user message's text
        system   = "request.system"           # optional: a system prompt kept apart
        reply    = "response"                 # required: text, or a ChatCompletion / Message
        id       = "trace_id"                 # optional: a hash of the row otherwise
        time     = "timestamp"                # optional: ISO or epoch seconds
        usage    = "response.usage"           # an OpenAI or Anthropic usage object, told apart by its keys
        input_tokens  = "tokens.in"           # or the counts one by one, which win over `usage`
        output_tokens = "tokens.out"
        cached_tokens = "tokens.cache_read"
        cache_write_tokens    = "tokens.cache_write"
        cache_write_1h_tokens = "tokens.cache_write_1h"
        input_excludes_cache  = false         # true when input_tokens leaves the cache out, as Anthropic's does
        tags = ["team", "metadata"]           # fields that become declared tags (a table spreads its keys)

    Map teams, products and environments into `tags`, not user ids: a tag names a team or a
    product, not a person.
    """

    PATHS = ("id", "model", "time", "messages", "prompt", "system", "reply", "usage", "input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens", "cache_write_1h_tokens")
    COUNTS = ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens", "cache_write_1h_tokens")

    def __init__(self, spec: dict[str, Any]):
        unknown = set(spec) - {*self.PATHS, "tags", "input_excludes_cache"}
        if unknown:
            raise ValueError(f"unknown mapping keys {sorted(unknown)}; the keys are {', '.join((*self.PATHS, 'tags', 'input_excludes_cache'))}")
        missing = [k for k in ("model", "reply") if k not in spec] + ([] if "messages" in spec or "prompt" in spec else ["messages or prompt"])
        if missing:
            raise ValueError(f"the mapping needs {', '.join(missing)}")
        self.paths = {k: str(spec[k]) for k in self.PATHS if k in spec}
        tags = spec.get("tags", [])
        self.tags = [tags] if isinstance(tags, str) else list(tags)
        self.input_excludes_cache = bool(spec.get("input_excludes_cache", False))

    @classmethod
    def load(cls, path: str | Path) -> Mapping:
        return cls(tomllib.loads(Path(path).read_text()))

    def get(self, row: dict[str, Any], key: str, default: Any = None) -> Any:
        v = _get(row, self.paths[key]) if key in self.paths else MISSING
        return default if v is MISSING or v is None else v

    def call(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.get(row, "model") is None:
            raise ValueError(f"no model at {self.paths['model']!r} in a row with keys {sorted(row)[:8]}")
        system = _text(self.get(row, "system"))
        prompt = self.get(row, "prompt")
        messages = self.get(row, "messages", []) if "messages" in self.paths else [{"role": "user", "content": _text(prompt)}] if prompt is not None else []
        usage = _usage(self.get(row, "usage", {}))
        counts = {k: int(self.get(row, k)) for k in self.COUNTS if self.get(row, k) is not None}
        if counts:
            usage = {k: 0 for k in self.COUNTS} | usage | counts
            if self.input_excludes_cache:
                usage["input_tokens"] += usage["cached_tokens"] + usage["cache_write_tokens"] + usage["cache_write_1h_tokens"]
        tags: dict[str, str] = {}
        for path in self.tags:
            if (v := _get(row, path)) is MISSING:
                continue
            tags |= _tags(v) if isinstance(v, dict) else _tags({path.rsplit(".", 1)[-1]: v})
        return {
            "id": str(self.get(row, "id") or hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]),
            "model": str(self.get(row, "model")),
            "time": _time(self.get(row, "time")),
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "reply": _reply(self.get(row, "reply", "")),
            "usage": usage,
            "tags": tags,
        }


def call(row: dict[str, Any], fmt: str, mapping: Mapping | None = None) -> dict[str, Any]:
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
            "messages": _messages(body),
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
    if fmt == "anthropic":
        req, response = _json(row["request"]), _json(row["response"])
        return {
            "id": str(response.get("id") or row.get("id")),
            "model": response.get("model") or req.get("model", ""),
            "time": _time(row.get("created_at")),
            "messages": _messages(req),
            "reply": _reply(response),
            "usage": _anthropic_usage(response.get("usage") or {}),
            "tags": _tags(row.get("metadata") or {}),  # the request's own metadata holds only a user id
        }
    if fmt == "custom":
        if mapping is None:
            raise ValueError("the custom format reads rows through a mapping file: pass --mapping")
        return mapping.call(row)
    raise ValueError(f"unknown log format {fmt!r}; one of {', '.join(FORMATS)}")


def detect(row: dict[str, Any]) -> str:
    if "request_body" in row or "request_id" in row:
        return "helicone"
    if "request" in row and "response" in row:
        response = _json(row["response"])
        return "anthropic" if isinstance(response, dict) and response.get("type") == "message" else "openai"
    if "messages" in row and "startTime" in row:
        return "litellm"
    raise ValueError(f"cannot tell the log format from keys {sorted(row)[:8]}; pass --format")


def _key(messages: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(f"{m.get('role')}\x00{_text(m.get('content'))}\x01".encode())
    return h.hexdigest()


def conversations(rows: list[dict[str, Any]], fmt: str = "auto", mapping: Mapping | None = None) -> list[dict[str, Any]]:
    """Log rows to conversations (this package's format), threads stitched, in time order. A
    mapping reads rows as the custom format (under "auto" too)."""
    if mapping is not None and fmt == "auto":
        fmt = "custom"
    calls = [call(r, detect(r) if fmt == "auto" else fmt, mapping) for r in rows]
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


def read(path: str | Path, fmt: str = "auto", mapping: str | Path | None = None) -> list[dict[str, Any]]:
    """A .jsonl or .json (a list) log export, as conversations; `mapping`, a custom format's TOML."""
    p = Path(path)
    text = p.read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if p.suffix == ".jsonl" else json.loads(text)
    if isinstance(rows, dict):  # an API response wrapping its rows
        rows = rows.get("data") or rows.get("rows") or []
    return conversations(rows, fmt, Mapping.load(mapping) if mapping else None)
