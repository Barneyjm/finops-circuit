"""Conversations in, one format out.

A conversation here is `{"id", "model", "timestamp", "language", "turns": [{"role", "content", "tokens"}]}`,
plus `expected_answers` in the hand-written samples, which the offline backend reads. WildChat
rows are converted on the way in, and only these fields are kept: WildChat also records a
country, a state, a hashed IP and browser headers per conversation, which a cost report has no
use for, so they never reach disk.

    tokenomics fetch --n 200                     # a sample of WildChat-4.8M into data/wildchat.jsonl
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROWS = "https://datasets-server.huggingface.co/rows?dataset=allenai/WildChat-4.8M&config=default&split=train&offset={offset}&length={length}"
WILDCHAT_ROWS = 3_199_860  # train split rows at the time of writing; offsets past the end come back empty


def from_wildchat(row: dict[str, Any]) -> dict[str, Any]:
    """A WildChat-4.8M row as a conversation: model, language, turns, measured output tokens."""
    turns = [{"role": t["role"], "content": t["content"], "tokens": t.get("token_counter") if t["role"] == "assistant" else None} for t in row["conversation"] if t.get("content")]
    return {
        "id": row["conversation_hash"],
        "model": row["model"],
        "timestamp": str(row.get("timestamp") or "")[:19] or None,
        "language": row.get("language"),
        "toxic": bool(row.get("toxic")),
        "turns": turns,
    }


def _get(url: str, tries: int = 8) -> dict[str, Any]:
    """GET JSON, waiting and retrying when the datasets server rate-limits (429) or hiccups (5xx)."""
    wait = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"user-agent": "llm-tokenomics"}), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == tries - 1:
                raise
            retry_after = e.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else wait)
            wait = min(wait * 2, 60.0)
    raise RuntimeError("unreachable")


def fetch_wildchat(n: int, seed: int = 0, batch: int = 10, pause: float = 0.3) -> list[dict[str, Any]]:
    """`n` conversations from random places in WildChat-4.8M, via Hugging Face's datasets
    server: no shard is downloaded. Small batches from many offsets, since neighbouring rows
    share a period and a model; a short pause between requests, and backoff when the server
    asks for it. ODC-BY; attribute AI2's WildChat when you publish from it."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    while len(out) < n:
        url = ROWS.format(offset=rng.randrange(WILDCHAT_ROWS - batch), length=min(batch, n - len(out)))
        out += [from_wildchat(x["row"]) for x in _get(url)["rows"] if x["row"].get("conversation")]
        time.sleep(pause)
    return out[:n]


def load(path: str | Path) -> list[dict[str, Any]]:
    """A .json conversation, a directory of them, or a .jsonl file of them."""
    p = Path(path)
    if p.is_dir():
        return [json.loads(f.read_text()) | {"file": f.name} for f in sorted(p.glob("*.json"))]
    if p.suffix == ".jsonl":
        return [json.loads(line) | {"file": f"{p.name}:{i + 1}"} for i, line in enumerate(p.read_text().splitlines()) if line.strip()]
    return [json.loads(p.read_text()) | {"file": p.name}]


def transcript(conv: dict[str, Any], max_chars: int = 6000) -> list[str]:
    """What the model reads: the turns as lines, `user:` and `assistant:`, long ones cut in the
    middle so a long conversation fits and its beginning and end both survive."""
    lines = []
    for t in conv["turns"]:
        text = " ".join(t["content"].split())
        if len(text) > 700:
            text = text[:450] + " [...] " + text[-200:]
        lines.append(f"{t['role']}: {text}")
    while sum(map(len, lines)) > max_chars and len(lines) > 4:
        del lines[len(lines) // 2]  # the middle of a long conversation says least about its purpose
    return lines
