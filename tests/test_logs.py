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


def anthropic(i, messages, reply, usage, system=None, **row):
    request = {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": messages, "metadata": {"user_id": "someone"}} | ({"system": system} if system else {})
    response = {"id": f"msg_{i}", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6", "content": [{"type": "text", "text": reply}], "stop_reason": "end_turn", "usage": usage}
    return {"request": request, "response": response} | row


def test_anthropic_pairs_count_cache_reads_and_writes_into_input_and_bill_writes_apart():
    system = [{"type": "text", "text": SYS["content"], "cache_control": {"type": "ephemeral"}}]
    rows = [
        anthropic(
            1,
            [Q1],
            A1,
            {"input_tokens": 20, "output_tokens": 40, "cache_creation_input_tokens": 1180, "cache_read_input_tokens": 0},
            system,
            created_at="2026-09-01T10:00:00Z",
            metadata={"team": "ops"},
        ),
        anthropic(
            2,
            [Q1, {"role": "assistant", "content": [{"type": "text", "text": A1}]}, Q2],
            A2,
            {
                "input_tokens": 30,
                "output_tokens": 90,
                "cache_read_input_tokens": 1180,
                "cache_creation_input_tokens": 190,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 190},
            },
            system,
            created_at="2026-09-01T10:01:00Z",
        ),
    ]
    assert detect(rows[0]) == "anthropic" and detect({"request": {}, "response": json.dumps(rows[0]["response"])}) == "anthropic"
    (c,) = conversations(rows)
    assert [t["role"] for t in c["turns"]] == ["system", "user", "assistant", "user", "assistant"]
    assert c["turns"][0]["content"] == SYS["content"] and c["timestamp"] == "2026-09-01T10:00:00"
    assert c["tags"] == {"team": "ops"}  # the request's user_id is not carried over
    spend = cost(c["model"], c["turns"])
    assert (spend.input_tokens, spend.cached_tokens, spend.cache_write_tokens, spend.cache_write_1h_tokens, spend.output_tokens) == (2600, 1180, 1180, 190, 130)
    # sonnet 4.6: $3 input, $0.30 cache read, $3.75 5-minute write, $6 1-hour write, $15 output
    assert spend.usd == pytest.approx((50 * 3.00 + 1180 * 0.30 + 1180 * 3.75 + 190 * 6.00 + 130 * 15.00) / 1e6)


def test_helicone_rows_for_claude_keep_the_system_prompt():
    row = {
        "request_id": "h2",
        "request_created_at": "2026-09-01T10:00:00Z",
        "request_model": "claude-haiku-4-5",
        "request_body": {"system": "Be brief.", "messages": [Q1]},
        "response_body": {"type": "message", "content": [{"type": "text", "text": A1}]},
        "prompt_tokens": 30,
        "completion_tokens": 12,
    }
    (c,) = conversations([row])
    assert [t["role"] for t in c["turns"]] == ["system", "user", "assistant"] and c["turns"][2]["content"] == A1


def test_custom_rows_through_a_mapping(tmp_path):
    from tokenomics.logs import Mapping, read

    spec = """
model = "llm.model"
system = "llm.system"
messages = "llm.history"
reply = "llm.output"
id = "trace"
time = "ts"
input_tokens = "cost.in"
output_tokens = "cost.out"
cached_tokens = "cost.cache_read"
cache_write_tokens = "cost.cache_write"
input_excludes_cache = true
tags = ["team", "labels"]
"""
    first = {
        "trace": "t1",
        "ts": 1_756_720_800,
        "team": "ops",
        "user": "someone",
        "labels": {"product": "supplier-desk"},
        "llm": {"model": "claude-haiku-4-5", "system": "Be brief.", "history": [Q1], "output": A1},
        "cost": {"in": 10, "out": 12, "cache_write": 1100},
    }
    second = first | {
        "trace": "t2",
        "ts": 1_756_720_860,
        "llm": first["llm"] | {"history": [Q1, {"role": "assistant", "content": A1}, Q2], "output": A2},
        "cost": {"in": 20, "out": 30, "cache_read": 1100},
    }
    (tmp_path / "map.toml").write_text(spec)
    (tmp_path / "logs.jsonl").write_text(json.dumps(second) + "\n" + json.dumps(first) + "\n")
    (c,) = read(tmp_path / "logs.jsonl", mapping=tmp_path / "map.toml")
    assert c["id"] == "t1" and c["timestamp"] == "2025-09-01T10:00:00"
    assert [t["role"] for t in c["turns"]] == ["system", "user", "assistant", "user", "assistant"]
    assert c["tags"] == {"team": "ops", "product": "supplier-desk"}  # only the mapped fields
    spend = cost(c["model"], c["turns"])
    assert (spend.input_tokens, spend.cached_tokens, spend.cache_write_tokens, spend.output_tokens) == (2230, 1100, 1100, 42)

    # a prompt/completion log with a provider usage object, told apart by its keys
    simple = Mapping({"model": "m", "prompt": "q", "reply": "a.choices.0.message.content", "usage": "a.usage"})
    row = {"m": "gpt-4.1", "q": "hi", "a": json.dumps({"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})}
    (o,) = conversations([row], mapping=simple)
    assert o["turns"][-1] == {"role": "assistant", "content": "hello", "usage": {"input_tokens": 5, "output_tokens": 2, "cached_tokens": 0}}
    assert len(o["id"]) == 16  # no id field: a hash of the row


def test_a_mapping_is_checked_on_load():
    from tokenomics.logs import Mapping

    with pytest.raises(ValueError, match="unknown mapping keys"):
        Mapping({"model": "m", "prompt": "p", "reply": "r", "modle": "x"})
    with pytest.raises(ValueError, match="needs reply, messages or prompt"):
        Mapping({"model": "m"})
    with pytest.raises(ValueError, match="pass --mapping"):
        conversations([{"x": 1}], "custom")
