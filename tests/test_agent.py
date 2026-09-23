"""The circuit, the pricing and the report, with the hand-written answers in samples/: no network."""

import json
from pathlib import Path

import pytest

from finops import analyze, app_ids, build_circuit, load_taxonomy, report, tag_keys
from finops.backends import FakeBackend
from finops.conversations import from_wildchat, load, transcript
from finops.pricing import cost, price_of

SAMPLES = {c["id"]: c for c in load(Path(__file__).resolve().parents[1] / "samples")}
APPS = app_ids(list(SAMPLES.values()))
TX = load_taxonomy()
V2 = build_circuit(TX, v2=True)


def run(cid, circuit=V2):
    return analyze(SAMPLES[cid], FakeBackend(), circuit=circuit, app=APPS[cid])


def test_every_conversation_gets_every_tag_key():
    for cid in SAMPLES:
        assert list(run(cid).tags) == tag_keys(TX), cid


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
    assert set(conv) == {"id", "model", "timestamp", "language", "toxic", "turns"} and conv["turns"][1]["tokens"] == 3
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


def test_the_dashboard_carries_the_data_and_no_conversation_text_by_default():
    from dataclasses import asdict

    from finops.html import document, render

    findings = [asdict(run(cid)) | {"first_message": SAMPLES[cid]["turns"][0]["content"]} for cid in SAMPLES]
    body = render(findings)
    assert "<title>LLM Spend Ledger</title>" in body and '"rows":' in body and "cracked screen" not in body
    with_text = render(findings, texts={f["id"]: f["first_message"] for f in findings})
    assert "cracked screen" in with_text and document(body).startswith("<!doctype html>")


def test_spend_by_month():
    a = analyze({**SAMPLES["01_support_reply"], "timestamp": "2024-05-02T10:00:00"}, FakeBackend(), circuit=V2)
    b = analyze({**SAMPLES["02_sql_debug"], "timestamp": "2024-05-20T09:00:00"}, FakeBackend(), circuit=V2)
    c = analyze({**SAMPLES["03_meeting_summary"], "timestamp": "2024-07-01T00:00:00"}, FakeBackend(), circuit=V2)
    months = report([a, b, c])["by_month"]
    assert list(months) == ["2024-05", "2024-07"] and months["2024-05"] == pytest.approx(round(a.cost.usd + b.cost.usd, 4))


CUSTOM = """
[tags.team]
question = "Which team would own this work?"
[tags.team.options]
growth = "Marketing and sales"
platform = "Engineering"
[tags.stage]
parent = "team"
question = "Which {parent} activity?"
[tags.stage.options.platform]
build = "Building"
run = "Running"
[tags.risk]
question = "How risky is the content?"
[tags.risk.options]
low = "Low"
high = "High"
[actions]
hold = { risk = ["high"] }
"""


def test_a_custom_taxonomy_defines_its_own_tags_children_and_actions(tmp_path):
    path = tmp_path / "tags.toml"
    path.write_text(CUSTOM)
    tx = load_taxonomy(path)
    conv = {
        **SAMPLES["02_sql_debug"],
        "tags": {"cost_center": "cc-4411"},
        "expected_answers": {**SAMPLES["02_sql_debug"]["expected_answers"], "team": {"growth": 0.1, "platform": 0.9}, "stage": {"build": 0.9, "run": 0.1}, "risk": {"low": 0.1, "high": 0.9}},
    }
    f = analyze(conv, FakeBackend(), taxonomy=tx, circuit=build_circuit(tx))
    assert f.tags == {"app": "adhoc", "team": "platform", "stage": "build", "risk": "high", "cost_center": "cc-4411"}
    assert f.tag_source["cost_center"] == "declared" and any(a.startswith("hold: risk=high") for a in f.actions)
    growth = analyze({**conv, "expected_answers": {**conv["expected_answers"], "team": {"growth": 0.9, "platform": 0.1}}}, FakeBackend(), taxonomy=tx, circuit=build_circuit(tx))
    assert growth.tags["stage"] == "n/a"  # growth has no stages: not applicable, not untagged


def test_a_broken_taxonomy_is_refused(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(CUSTOM.replace('hold = { risk = ["high"] }', 'hold = { risk = ["extreme"] }'))
    with pytest.raises(ValueError, match="extreme"):
        load_taxonomy(bad)


def test_focus_rows_carry_the_mandatory_columns_and_add_up():
    from finops.focus import COLUMNS, rows

    findings = [run(cid) for cid in SAMPLES]
    out = rows(findings)
    assert len(out) == 2 * len(findings)
    mandatory = ["BilledCost", "BillingAccountId", "BillingCurrency", "ChargeCategory", "ChargePeriodStart", "EffectiveCost", "ListCost", "PricingUnit", "ServiceCategory", "ServiceProviderName"]
    assert all(r[k] not in (None, "") for r in out for k in mandatory) and set(out[0]) <= set(COLUMNS)
    assert sum(r["BilledCost"] for r in out) == pytest.approx(sum(f.cost.usd for f in findings))
    assert sum(r["x_PotentialSavings"] for r in out) == pytest.approx(sum(f.savings_usd for f in findings))
    tags = json.loads(out[0]["Tags"])
    assert all(k.startswith("finops-circuit/") for k in tags) and "untagged" not in tags.values()
    assert out[0]["ServiceSubcategory"] == "Generative AI" and out[0]["PricingUnit"] == "1000000 Tokens"


PRICES = """
[settings]
currency = "EUR"
small_model = "tiny"

[models."tiny"]
input = 0.10
output = 0.40
cached_input = 0.05

[models."big"]
input = 2.00
output = 8.00
cached_input = 0.50
discount = 0.25

[models."big-nocache"]
input = 2.00
output = 8.00
"""


def prices(tmp_path, text=PRICES):
    from finops import load_prices

    (tmp_path / "prices.toml").write_text(text)
    return load_prices(tmp_path / "prices.toml")


def test_a_price_table_sets_prices_discount_currency_and_the_small_model(tmp_path):
    table = prices(tmp_path)
    assert table.of("big-2025-01-01").cached_input == 0.50 and table.of("big-nocache-x").cached_input is None  # longest prefix wins
    turns = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y", "tokens": 50}]
    c = cost("big", turns, table)
    assert c.usd == pytest.approx((100 * 2 + 50 * 8) / 1e6) and c.contracted_usd == pytest.approx(c.usd * 0.75) and c.currency == "EUR"
    assert c.usd_on_small_model == pytest.approx((100 * 0.10 + 50 * 0.40) / 1e6)
    with pytest.raises(ValueError, match="discount"):
        prices(tmp_path, PRICES.replace("discount = 0.25", "discount = 1.5"))
    with pytest.raises(KeyError, match="tinier"):
        prices(tmp_path, PRICES.replace('small_model = "tiny"', 'small_model = "tinier"').replace('[models."tiny"]', '[models."other"]'))


def test_measured_usage_is_used_as_is_and_cached_tokens_bill_at_the_cache_price(tmp_path):
    table = prices(tmp_path)
    turns = [
        {"role": "user", "content": "x" * 4000},
        {"role": "assistant", "content": "y", "usage": {"input_tokens": 1200, "output_tokens": 80, "cached_tokens": 0}},
        {"role": "user", "content": "z" * 40},
        {"role": "assistant", "content": "w", "usage": {"input_tokens": 1300, "output_tokens": 90, "cached_tokens": 1024}},
    ]
    c = cost("big", turns, table)
    assert (c.input_tokens, c.cached_tokens, c.output_tokens) == (2500, 1024, 170) and c.input_measured and c.output_measured == 170
    assert c.usd == pytest.approx(((2500 - 1024) * 2 + 1024 * 0.50 + 170 * 8) / 1e6)
    assert c.cached_usd == pytest.approx(1024 * 0.50 / 1e6)
    assert not cost("big", [{"role": "user", "content": "x" * 40}, {"role": "assistant", "content": "y"}], table).input_measured


def test_a_long_conversation_on_a_model_with_a_prompt_cache_is_told_to_cache(tmp_path):
    table = prices(tmp_path)
    conv = {**SAMPLES["10_long_code_session"], "model": "big"}
    f = analyze(conv, FakeBackend(), circuit=V2, app=APPS["10_long_code_session"], prices=table)
    assert any(a.startswith("prompt caching") for a in f.actions) and not any(a.startswith("trim") for a in f.actions)
    assert f.action_savings["prompt caching"] == pytest.approx(f.cost.resent_tokens * (2.00 - 0.50) / 1e6)
    assert f.savings_usd == pytest.approx(sum(f.action_savings.values())) and f.savings_usd < f.cost.usd
    nocache = analyze({**conv, "model": "big-nocache"}, FakeBackend(), circuit=V2, app=APPS["10_long_code_session"], prices=table)
    assert not any(a.startswith("prompt caching") for a in nocache.actions)  # no cache price: trim context is the fallback


def test_focus_bills_cached_input_on_its_own_row_after_the_discount(tmp_path):
    from finops.focus import rows

    table = prices(tmp_path)
    conv = {
        "id": "c1",
        "model": "big",
        "timestamp": "2024-05-01T10:00:00",
        "turns": [
            {"role": "user", "content": "Summarise the attached quarterly supplier report for the ops team."},
            {"role": "assistant", "content": "Summary.", "usage": {"input_tokens": 3000, "output_tokens": 200, "cached_tokens": 2048}},
        ],
    }
    f = analyze(conv, FakeBackend(), circuit=V2, prices=table)
    out = rows([f], prices=table)
    assert [r["SkuId"] for r in out] == ["big/input-tokens", "big/cached-input-tokens", "big/output-tokens"]
    assert [r["ConsumedQuantity"] for r in out] == [3000 - 2048, 2048, 200]
    assert sum(r["ListCost"] for r in out) == pytest.approx(f.cost.usd)
    assert sum(r["BilledCost"] for r in out) == pytest.approx(f.cost.contracted_usd) == pytest.approx(f.cost.usd * 0.75)
    assert out[1]["ListUnitPrice"] == 0.50 and out[1]["ContractedUnitPrice"] == pytest.approx(0.375) and out[0]["BillingCurrency"] == "EUR"
    assert out[0]["x_QuantityEstimated"] == "false"


def test_reprice_decides_again_from_the_saved_gates_without_the_model(tmp_path):
    from dataclasses import asdict

    from finops import reprice

    conv = {**SAMPLES["10_long_code_session"], "model": "big"}
    old = prices(tmp_path, PRICES.replace("cached_input = 0.50\n", ""))  # the same table before "big" got a cache price
    before = analyze(conv, FakeBackend(), circuit=V2, app=APPS["10_long_code_session"], prices=old)
    assert not any(a.startswith("prompt caching") for a in before.actions)
    table = prices(tmp_path)
    again = analyze(conv, FakeBackend(), circuit=V2, app=APPS["10_long_code_session"], prices=table)
    after = reprice(json.loads(json.dumps(asdict(before), default=str)), conv, prices=table)
    assert after.cost == again.cost and after.actions == again.actions and after.action_savings == pytest.approx(again.action_savings)
    assert after.tags == before.tags and after.audit == json.loads(json.dumps(before.audit, default=str))
