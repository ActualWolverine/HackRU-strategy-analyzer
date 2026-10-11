"""The AI explanation layer stays bounded by engine facts.

No network: these tests never call Gemini. They check the deterministic parts
(facts, template fallback, citation validation, indicator reference text) and
that the service falls back cleanly when the model fails.
"""

import json

import pytest

import backtest as bt
import explain
from explain import ExplanationService, TemplateExplainer, period_facts, strategy_facts, validate
from indicators import DESCRIPTIONS, describe_indicators
from regime import INDICATORS, describe_period
from strategies import STRATEGIES, universe_for

KEY = "sma_trend"


@pytest.fixture(scope="module")
def result():
    spec = STRATEGIES[KEY]
    params = {p["name"]: p["default"] for p in spec["params"]}
    return bt.backtest(spec["fn"], universe_for(KEY, params), spec["benchmark"], params,
                       published=spec.get("published")), spec, params


@pytest.fixture(scope="module")
def facts(result):
    r, spec, params = result
    return strategy_facts(r, spec, params)


def test_facts_contain_no_raw_series(facts):
    """The model gets structured facts only: no prices, equity curves, dates lists or trades."""
    blob = json.dumps(facts)
    assert len(blob) < 20_000
    for forbidden in ("strategy_equity", "benchmark_equity", '"dates"', "regimes", "close"):
        assert forbidden not in blob
    assert all(isinstance(v, str) for v in facts.values())


def test_facts_are_deterministic(result):
    r, spec, params = result
    assert strategy_facts(r, spec, params) == strategy_facts(r, spec, params)


def test_facts_quote_engine_numbers(result, facts):
    r, _, _ = result
    assert f"{r['metrics']['strategy']['cagr'] * 100:+.1f}%" in facts["cagr"]
    assert f"{r['alpha']['full']['t_stat']:.2f}" in facts["alpha_full"]
    top = r["conditions"]["outperform"][0]
    assert top["indicator"] in facts["outperform_1"] and top["direction"] in facts["outperform_1"]


def test_period_facts(result):
    r, spec, params = result
    stats = bt.period_stats(spec["fn"], universe_for(KEY, params), spec["benchmark"], params, "2008-09-01", "2009-06-30")
    conditions = describe_period(stats["start"], stats["end"])["indicators"]
    f = period_facts(stats, spec, params, conditions)
    assert "2008-09-02" in f["window"] and "2009-06-30" in f["window"]
    assert f"{stats['excess_return'] * 100:+.1f}%" in f["window_return"]
    assert stats["n_days"] == 209
    # Ordinals are well formed (no "22th").
    assert "th percentile" not in "".join(v for k, v in f.items() if k.startswith("condition_") and ("1st" in v or "2nd" in v or "3rd" in v))


def test_period_facts_include_alpha_inflections(result):
    r, spec, params = result
    stats = bt.period_stats(spec["fn"], universe_for(KEY, params), spec["benchmark"], params, "2018-06-01", "2020-12-31")
    f = period_facts(stats, spec, params, [])
    inf = stats["inflections"]
    assert f"beta {inf['beta']:.2f}" in f["alpha_definition"]
    assert inf["turning_points"], "2018-2020 has clear alpha turns for the SMA strategy"
    for t in inf["turning_points"]:
        assert t["date"] in f[f"inflection_{t['n']}"]
        assert f"inflection_{t['n']}_position" in f
    # The template gives each inflection its own cited section.
    out = TemplateExplainer().explain(explain.PERIOD_TASK, f, {})
    headings = [s["heading"] for s in out["sections"]]
    for t in inf["turning_points"]:
        assert any(h.startswith(t["date"]) for h in headings)


def test_period_stats_rejects_empty_window(result):
    _, spec, params = result
    with pytest.raises(ValueError):
        bt.period_stats(spec["fn"], universe_for(KEY, params), spec["benchmark"], params, "1990-01-01", "1990-01-02")


def test_template_is_deterministic_and_cites_only_real_facts(facts):
    t = TemplateExplainer()
    defs = explain.indicator_definitions(facts)
    a, b = t.explain(explain.STRATEGY_TASK, facts, defs), t.explain(explain.STRATEGY_TASK, facts, defs)
    assert a == b
    for section in a["sections"]:
        assert all(k in facts for k in section["evidence"])
    assert a["caveats"]


def test_validate_strips_unknown_citations(facts):
    fake = {"summary": "x", "caveats": ["y"], "sections": [
        {"heading": "ok", "body": "...", "basis": "fact", "evidence": ["cagr", "made_up_fact"]},
        {"heading": "empty", "body": "...", "basis": "fact", "evidence": ["not_real"]},
        {"heading": "ref", "body": "...", "basis": "reference", "evidence": []},
        {"heading": "weird", "body": "...", "basis": "truth", "evidence": []},
    ]}
    v = validate(fake, facts)
    assert v["sections"][0]["evidence"] == ["cagr"] and v["sections"][0]["unsupported_citations"] == 1
    assert v["sections"][1]["basis"] == "inference"  # fact section with nothing verifiable is downgraded
    assert v["sections"][2]["basis"] == "reference"  # reference text is left alone
    assert v["sections"][3]["basis"] == "inference"  # unknown basis values are normalised
    assert v["unsupported_citations"] == 2


class FailingExplainer:
    name = "gemini"
    model = "fake"

    def explain(self, task, facts, definitions):
        raise RuntimeError("simulated outage")


class CountingExplainer:
    name = "gemini"
    model = "fake"
    calls = 0

    def explain(self, task, facts, definitions):
        self.calls += 1
        return {"summary": "s", "caveats": ["c"], "sections": [
            {"heading": "h", "body": "b", "basis": "fact", "evidence": ["cagr", "bogus"]}]}


def test_service_falls_back_and_does_not_cache_fallback(facts):
    svc = ExplanationService(FailingExplainer())
    r = svc.explain(explain.STRATEGY_TASK, facts)
    assert r["provider"] == "template" and "simulated outage" in r["fallback_reason"]
    assert r["cached"] is False and svc._cache == {}


def test_service_validates_and_caches_model_output(facts):
    ex = CountingExplainer()
    svc = ExplanationService(ex)
    r1 = svc.explain(explain.STRATEGY_TASK, facts)
    r2 = svc.explain(explain.STRATEGY_TASK, facts)
    assert ex.calls == 1 and r2["cached"] is True
    assert r1["sections"][0]["evidence"] == ["cagr"] and r1["unsupported_citations"] == 1
    assert r1["provider"] == "gemini" and r1["model_latency_ms"] >= 0
    assert r1["facts"] == facts


def test_service_without_key_uses_template(facts, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    svc = ExplanationService.from_env()
    assert svc.provider == "template"
    assert svc.explain(explain.STRATEGY_TASK, facts)["provider"] == "template"


def test_every_indicator_has_reference_text():
    assert set(DESCRIPTIONS) == set(INDICATORS)
    rows = describe_indicators()
    assert [r["key"] for r in rows] == list(INDICATORS)
    for r in rows:
        for field in ("what", "read", "matters", "caveat", "label", "high", "low"):
            assert r[field], f"{r['key']} missing {field}"


def test_definitions_only_for_cited_indicators(facts):
    defs = explain.indicator_definitions(facts)
    assert defs and set(defs) <= set(INDICATORS)
    for key in defs:
        assert f"Indicator key: {key}." in " ".join(facts.values())
