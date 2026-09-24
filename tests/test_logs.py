"""Gateway logs to conversations: thread stitching, measured usage, declared tags."""

import json

import pytest

from tokenomics.logs import conversations, detect
from tokenomics.pricing import cost

SYS = {"role": "system", "content": "You help the ops team with supplier questions."}
Q1 = {"role": "user", "content": "Which suppliers missed the March delivery window?"}
A1 = "Three did: Acme, Borealis and Crane."
Q2 = {"role": "user", "content": [{"type": "text", "text": "Draft a note to Crane about it."}]}
A2 = "Hello Crane team, ..."


def litellm(i, t, messages, reply, pt, ct, cached=0, team="ops"):
    return {
        "id": f"call-{i}",
        "model": "gpt-4o-2024-08-06",
        "startTime": t,
        "messages": messages,
        "response": {"choices": [{"message": {"role": "assistant", "content": reply}}], "usage": {"prompt_tokens": pt, "completion_tokens": ct, "prompt_tokens_details": {"cached_tokens": cached}}},
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "metadata": {"user_api_key_team_alias": team, "user_api_key_alias": "ops-bot", "user_api_key_user_id": "someone", "requester_metadata": {"environment": "production"}},
        "request_tags": ["supplier-desk"],
    }


def test_calls_of_one_thread_become_one_conversation_with_measured_usage():
    rows = [
        litellm(2, 1_700_000_060, [SYS, Q1, {"role": "assistant", "content": A1}, Q2], A2, 1400, 90, cached=1024),
        litellm(1, 1_700_000_000, [SYS, Q1], A1, 1200, 40),
        litellm(3, 1_700_000_100, [SYS, {"role": "user", "content": "Another question entirely."}], "Sure.", 900, 5),
    ]
    assert detect(rows[0]) == "litellm"
    convs = conversations(rows)
    assert len(convs) == 2
    c = convs[0]
    assert c["id"] == "call-1" and c["timestamp"] == "2023-11-14T22:13:20"
    assert [t["role"] for t in c["turns"]] == ["system", "user", "assistant", "user", "assistant"]
    assert c["turns"][3]["content"] == "Draft a note to Crane about it."
    assert c["tags"] == {"team": "ops", "api_key": "ops-bot", "environment": "production", "tag:supplier-desk": "true"}  # no user id
    spend = cost(c["model"], c["turns"])
    assert (spend.input_tokens, spend.cached_tokens, spend.output_tokens) == (2600, 1024, 130) and spend.input_measured


def test_few_shot_assistant_messages_are_input_not_replies():
    shots = [SYS, {"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "3+3"}]
    (c,) = conversations([litellm(1, 1_700_000_000, shots, "6", 60, 1)])
    assert [t.get("prompt", False) for t in c["turns"] if t["role"] == "assistant"] == [True, False]
    assert cost(c["model"], c["turns"]).output_tokens == 1


def test_helicone_and_openai_rows():
    helicone = {
        "request_id": "h1",
        "request_created_at": "2025-03-01T10:00:00Z",
        "request_model": "gpt-4o-mini",
        "request_body": json.dumps({"model": "gpt-4o-mini", "messages": [Q1]}),
        "response_body": {"choices": [{"message": {"content": A1}}]},
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "prompt_cache_read_tokens": 0,
        "request_properties": {"App": "supplier-desk", "Environment": "staging"},
        "request_user_id": "someone",
    }
    response = {"id": "chatcmpl-1", "model": "gpt-4.1-2025-04-14", "created": 1_740_823_200, "choices": [{"message": {"content": A1}}], "usage": {"prompt_tokens": 30, "completion_tokens": 12}}
    openai = {"request": {"model": "gpt-4.1", "messages": [Q1], "metadata": {"team": "ops"}}, "response": response}
    h, o = conversations([helicone]), conversations([openai])
    assert h[0]["tags"] == {"App": "supplier-desk", "Environment": "staging"} and h[0]["timestamp"] == "2025-03-01T10:00:00"
    assert o[0]["model"] == "gpt-4.1-2025-04-14" and o[0]["tags"] == {"team": "ops"}
    assert cost(o[0]["model"], o[0]["turns"]).usd == pytest.approx((30 * 2.00 + 12 * 8.00) / 1e6)
