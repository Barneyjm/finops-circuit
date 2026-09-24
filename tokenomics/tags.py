"""The `app` tag, set in code: which program sent a conversation.

A program calling a chat model wraps each input in the same fixed instructions ("Provide only
relevant keywords to facilitate an online search for: ..."), so its conversations open with the
same words and differ where the input goes. A conversation's template key is the opening of its
first message with the variable parts (numbers, quoted text, links, addresses) blanked; a key
that opens enough conversations in the set is one app. The rest are `adhoc`: people typing.

No model is asked. A fingerprint is a fact about the traffic, and it is what an app id would
say if the callers had set one.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

ADHOC = "adhoc"
KEY_WORDS = 8  # words of the opening that identify a template
MIN_CALLS = 3  # conversations sharing a key before it counts as an app

_VARIABLE = [
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"\"[^\"]*\"|'[^']*'|«[^»]*»|“[^”]*”"), "<quote>"),
    (re.compile(r"\d+([.,:/-]\d+)*"), "<n>"),
]


def template_key(conv: dict[str, Any]) -> str:
    text = _first(conv).lower()
    for pattern, stand_in in _VARIABLE:
        text = pattern.sub(stand_in, text)
    return " ".join(text.split()[:KEY_WORDS])


def _first(conv: dict[str, Any]) -> str:
    return next((t["content"] for t in conv["turns"] if t["role"] == "user"), "")


def app_ids(convs: list[dict[str, Any]], min_calls: int = MIN_CALLS) -> dict[str, str]:
    """Conversation id -> app id (`app-` and six hex digits of its template key), or `adhoc`.
    A key is an app when at least `min_calls` conversations open with it and they go on to say
    different things: a template wraps varying input. The same message sent many times ("hello!
    how are you today?") is a repeated request, not a program, and stays adhoc."""
    keys = {c["id"]: template_key(c) for c in convs}
    counts = Counter(k for k in keys.values() if k)
    variants: dict[str, set[str]] = {}
    for c in convs:
        variants.setdefault(keys[c["id"]], set()).add(" ".join(_first(c).split()).lower())
    is_app = {k for k, n in counts.items() if n >= min_calls and len(variants[k]) > 1}
    return {cid: (f"app-{hashlib.sha1(k.encode()).hexdigest()[:6]}" if k in is_app else ADHOC) for cid, k in keys.items()}


def app_labels(convs: list[dict[str, Any]], min_calls: int = MIN_CALLS) -> dict[str, str]:
    """App id -> its template key, so a report can say what each app is."""
    ids = app_ids(convs, min_calls)
    return {ids[c["id"]]: template_key(c) for c in convs if ids[c["id"]] != ADHOC}
