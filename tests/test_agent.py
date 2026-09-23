"""The circuit, the pricing and the report, with the hand-written answers in samples/: no network."""

from pathlib import Path

import pytest

from finops import analyze, build_circuit, report
from finops.backends import FakeBackend
from finops.conversations import from_wildchat, load, transcript
from finops.pricing import cost, price_of

SAMPLES = {c["id"]: c for c in load(Path(__file__).resolve().parents[1] / "samples")}
V2 = build_circuit(v2=True)


def run(cid, circuit=V2):
    return analyze(SAMPLES[cid], FakeBackend(), circuit=circuit)


def test_a_simple_customer_reply_on_gpt4_is_a_downgrade():
    f = run("01_support_reply")
    assert f.value == "customer_facing" and f.business is True
    assert any(a.startswith("downgrade") for a in f.actions) and 0 < f.savings_usd < f.cost.usd


def test_hard_or_sensitive_work_is_not_downgraded():
    assert not any(a.startswith("downgrade") for a in run("02_sql_debug").actions)  # a small model would not do
    contract = run("09_contract_clause")
    assert contract.business is True and not any(a.startswith("downgrade") for a in contract.actions)
    assert any(a.startswith("hold") for a in contract.actions)  # sensitive: a person decides first


def test_non_business_spend_is_a_policy_question():
    for cid in ("05_homework", "06_roleplay", "07_jailbreak", "08_test_ping"):
        f = run(cid)
        assert f.business is False and any(a.startswith("policy") for a in f.actions), cid
        assert f.savings_usd == pytest.approx(f.cost.usd)


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


def test_report_adds_up():
    findings = [run(cid) for cid in SAMPLES]
    r = report(findings)
    assert r["conversations"] == 10 and r["spend_usd"] == pytest.approx(sum(f.cost.usd for f in findings), abs=1e-4)
    assert sum(v["usd"] for v in r["by_value"].values()) == pytest.approx(r["spend_usd"], abs=1e-3)
    assert set(r["savings_usd"]) <= {"policy", "downgrade"} and 0 < r["savings_share"] < 1


def test_v1_backends_get_no_v2_questions():
    f = run("01_support_reply", circuit=build_circuit())
    assert "functions" not in f.audit["answers"] and f.value == "customer_facing"


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
