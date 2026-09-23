"""The circuit, the pricing and the report, with the hand-written answers in samples/: no network."""

from pathlib import Path

import pytest

from finops import TAG_KEYS, analyze, app_ids, build_circuit, report
from finops.backends import FakeBackend
from finops.conversations import from_wildchat, load, transcript
from finops.pricing import cost, price_of

SAMPLES = {c["id"]: c for c in load(Path(__file__).resolve().parents[1] / "samples")}
APPS = app_ids(list(SAMPLES.values()))
V2 = build_circuit(v2=True)


def run(cid, circuit=V2):
    return analyze(SAMPLES[cid], FakeBackend(), circuit=circuit, app=APPS[cid])


def test_every_conversation_gets_every_tag_key():
    for cid in SAMPLES:
        assert set(run(cid).tags) == set(TAG_KEYS), cid


def test_two_stage_task_then_subtask():
    assert (run("02_sql_debug").tags["task"], run("02_sql_debug").tags["subtask"]) == ("code", "sql")
    assert (run("01_support_reply").tags["task"], run("01_support_reply").tags["subtask"]) == ("writing", "email_message")
    assert run("03_meeting_summary").tags == {**run("03_meeting_summary").tags, "task": "summarization", "subtask": "list_notes", "domain": "operations"}


def test_a_templated_program_is_one_app_and_people_are_adhoc():
    apps = {APPS[c] for c in ("11_keyword_app", "12_keyword_app", "13_keyword_app")}
    assert len(apps) == 1 and next(iter(apps)).startswith("app-")
    assert APPS["01_support_reply"] == "adhoc"
    f = run("11_keyword_app")
    assert f.tags["workload"] == "automated" and f.tags["task"] == "extraction" and f.tags["subtask"] == "keywords"
    assert any(a.startswith("downgrade") for a in f.actions) and any(a.startswith("cache") for a in f.actions)


def test_an_unsure_tag_is_untagged_and_the_subtask_is_not_asked():
    conv = dict(SAMPLES["02_sql_debug"])
    flat = {k: 1 / 12 for k in conv["expected_answers"]["task"]}
    conv["expected_answers"] = {**conv["expected_answers"], "task": flat}
    f = analyze(conv, FakeBackend(), circuit=V2)
    assert f.tags["task"] == "untagged" and f.tags["subtask"] == "untagged" and "subtask" not in f.audit["answers"]
    assert any(a.startswith("review") for a in f.actions)


def test_dev_test_traffic_and_non_business_spend():
    ping = run("08_test_ping")
    assert ping.tags["environment"] == "dev_test" and ping.business is False and any(a.startswith("policy") for a in ping.actions)
    for cid in ("05_homework", "06_roleplay", "07_jailbreak"):
        f = run(cid)
        assert f.business is False and f.savings_usd == pytest.approx(f.cost.usd), cid


def test_sensitive_data_is_held_and_not_downgraded():
    f = run("09_contract_clause")
    assert f.tags["data_class"] == "confidential" and any(a.startswith("hold") for a in f.actions)
    assert not any(a.startswith("downgrade") for a in f.actions)


def test_a_simple_task_on_a_premium_model_is_a_downgrade():
    f = run("01_support_reply")
    assert any(a.startswith("downgrade") for a in f.actions) and 0 < f.savings_usd < f.cost.usd
    assert not any(a.startswith("downgrade") for a in run("02_sql_debug").actions)  # a small model would not do


def test_a_long_conversation_is_flagged_for_resending_its_history():
    f = run("10_long_code_session")
    assert f.turns == 6 and any(a.startswith("trim context") for a in f.actions)
    assert f.cost.resent_usd / f.cost.usd >= 1 / 3
    assert not any(a.startswith("trim") for a in run("06_roleplay").actions)  # long too, but policy already covers it


def test_cost_resends_the_history_every_turn():
    turns = [{"role": "user", "content": "x" * 400, "tokens": None}, {"role": "assistant", "content": "y", "tokens": 50}] * 2
    c = cost("gpt-4-0314", turns)
    # first reply reads 100 tokens; the second reads 100 + 50 + 100, of which the first 100 were sent before
    assert c.input_tokens == 100 + 250 and c.output_tokens == 100 and c.output_measured == 100
    assert c.resent_usd == pytest.approx(100 * 30 / 1e6)
    assert c.usd == pytest.approx(350 * 30 / 1e6 + 100 * 60 / 1e6)
    assert price_of("gpt-4o-mini-2024-07-18") == (0.15, 0.60) and price_of("gpt-4o-2024-05-13") == (5.00, 15.00)


def test_report_groups_spend_by_any_tags_and_measures_coverage():
    findings = [run(cid) for cid in SAMPLES]
    r = report(findings, by=("task", "subtask"))
    assert r["conversations"] == 13 and r["spend_usd"] == pytest.approx(sum(f.cost.usd for f in findings), abs=1e-4)
    assert sum(g["usd"] for g in r["groups"].values()) == pytest.approx(r["spend_usd"], abs=1e-3)
    assert "code / sql" in r["groups"] and r["tag_coverage"]["task"] == 1.0 and r["fully_tagged_share"] == 1.0
    by_app = report(findings, by=("app",))
    assert any(k.startswith("app-") and v["conversations"] == 3 for k, v in by_app["groups"].items())
    assert 0 < r["savings_share"] < 1


def test_v1_backends_get_no_v2_questions():
    f = run("01_support_reply", circuit=build_circuit())
    assert "purpose" not in f.audit["answers"] and f.tags["task"] == "writing"


def test_wildchat_rows_keep_only_what_a_cost_review_needs():
    row = {
        "conversation_hash": "h",
        "model": "gpt-4-0314",
        "language": "English",
        "toxic": False,
        "country": "Somewhere",
        "state": "Somewhere",
        "hashed_ip": "abc",
        "header": {"user-agent": "x"},
        "conversation": [{"role": "user", "content": "hi", "token_counter": None, "country": "Somewhere"}, {"role": "assistant", "content": "hello", "token_counter": 3}],
    }
    conv = from_wildchat(row)
    assert set(conv) == {"id", "model", "language", "toxic", "turns"} and conv["turns"][1]["tokens"] == 3
    assert "Somewhere" not in str(conv) and "abc" not in str(conv)


def test_transcript_keeps_both_ends_of_a_long_message():
    lines = transcript({"turns": [{"role": "user", "content": "start " + "x" * 2000 + " end"}]})
    assert lines[0].startswith("user: start") and lines[0].endswith("end") and "[...]" in lines[0]


def test_the_same_message_repeated_is_not_an_app():
    hello = {"model": "gpt-4o-mini", "turns": [{"role": "user", "content": "hello! how are you today?"}, {"role": "assistant", "content": "Hi!"}]}
    convs = [{**hello, "id": f"h{i}"} for i in range(5)]
    assert set(app_ids(convs).values()) == {"adhoc"}


def test_declared_tags_win_and_the_circuit_fills_the_gaps():
    conv = {**SAMPLES["02_sql_debug"], "tags": {"environment": "dev_test", "app": "billing-service"}}
    f = analyze(conv, FakeBackend(), circuit=V2, app="adhoc")
    assert f.tags["environment"] == "dev_test" and f.tags["app"] == "billing-service"
    assert f.tag_source["environment"] == "declared" and f.tag_source["task"] == "inferred" and f.tags["task"] == "code"
    r = report([f])
    assert r["declared_share"]["environment"] == 1.0 and r["declared_share"]["task"] == 0.0
