"""'Should have worked but didn't': deterministic, and behaves on hand-built histories."""

import json

import numpy as np
import pandas as pd
import pytest

from expectations import expectation_check, favorability_score, favorable_share
from regime import compare_conditions

WINDOW = 63


def _history(n_days: int, seed: int = 0):
    """One indicator drives the labels: high volatility -> outperform, low -> underperform.

    The labels are cut from the indicator's 63-day centred mean, the same
    smoothing the score uses, so the score can track them.
    """
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    rng = np.random.default_rng(seed)
    vol = pd.Series(np.abs(rng.normal(0.15, 0.05, n_days)), index=idx).rolling(21, min_periods=1).mean()
    state = pd.DataFrame({"realized_vol_20d": vol}, index=idx)
    driver = vol.rolling(WINDOW, center=True, min_periods=WINDOW // 2).mean()
    lo, hi = driver.quantile([1 / 3, 2 / 3])
    labels = pd.Series("match", index=idx, dtype=object)
    labels[driver >= hi] = "outperform"
    labels[driver <= lo] = "underperform"
    return idx, state, labels


def _returns(labels: pd.Series) -> pd.Series:
    return pd.Series(np.where(labels == "outperform", 0.002, np.where(labels == "underperform", -0.002, 0.0)), index=labels.index)


def _check(state, labels, published=None):
    conditions = compare_conditions(labels.eq("outperform"), state)
    benchmark = pd.Series(0.0, index=labels.index)
    weights = pd.DataFrame({"SPY": 1.0}, index=labels.index)
    return expectation_check(labels, _returns(labels), benchmark, weights, state, conditions, published, WINDOW)


def test_perfect_conditions_rarely_fail():
    idx, state, labels = _history(2500)
    out = _check(state, labels)
    assert out["available"]
    assert out["score_indicators"][0]["indicator"] == "realized_vol_20d"
    assert out["hit_rate"] > 0.8 and out["lift"] > 2
    assert out["failure_days"] < 0.05 * out["favorable_days"]
    assert out["verdict"]["kind"] in ("rare_failures", "marginal")
    assert out["what_differed"] == []  # the only indicator is inside the score


def test_edge_that_fades_is_reported_as_later():
    """Conditions keep saying 'favorable' but the strategy mostly stops working half way: failures cluster late."""
    idx, state, labels = _history(4000)
    rng = np.random.default_rng(7)
    late = (idx >= idx[2000]) & (labels == "outperform").to_numpy() & (rng.random(len(idx)) < 0.7)
    faded = labels.copy()
    faded[late] = "underperform"
    out = _check(state, faded, published=idx[2000].year)
    assert out["available"]
    assert out["date_effect"] > 0.6
    assert out["verdict"]["flags"] == ["later"] or "later" in out["verdict"]["flags"]
    assert "measured" not in out["verdict"]["flags"]
    assert out["verdict"]["kind"] == "later"
    pub = out["by_publication"]
    assert pub["before"]["hit_rate"] > 0.8 and pub["after"]["hit_rate"] < 0.5
    assert pub["after"]["miss_rate"] > pub["before"]["miss_rate"]
    rated = [e for e in out["by_era"] if e["rated"]]
    assert rated[0]["hit_rate"] > rated[-1]["hit_rate"]
    # Every listed failure period lies in the second half and underperformed.
    assert out["failures"] and all(f["start"] >= idx[2000].strftime("%Y-%m-%d") and f["excess_return"] < 0 for f in out["failures"])


def test_ignored_indicator_that_explains_failures_is_measured():
    """A second indicator decides success vs failure only under favorable conditions.

    Its overall link to outperformance is too weak to enter the score, so it
    is exactly the kind of missing interaction the contrast should surface.
    """
    idx, state, labels = _history(4000, seed=2)
    rng = np.random.default_rng(5)
    spread = pd.Series(rng.normal(2.0, 0.5, len(idx)), index=idx).rolling(63, min_periods=1).mean()
    state = state.assign(credit_spread=spread)
    wide = (spread > spread.median()).to_numpy()
    flip = wide & (labels == "outperform").to_numpy() & (rng.random(len(idx)) < 0.4)
    broken = labels.copy()
    broken[flip] = "underperform"
    out = _check(state, broken)
    assert out["available"]
    assert all(i["indicator"] != "credit_spread" for i in out["score_indicators"])
    top = out["what_differed"][0]
    assert top["indicator"] == "credit_spread" and top["effect"] > 0.4
    assert out["verdict"]["kind"] == "measured"


def test_no_notable_indicator_means_no_signal():
    idx, state, labels = _history(2500)
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.choice(["outperform", "match", "underperform"], len(idx)), index=idx)
    out = _check(state, noise)
    assert not out["available"]
    assert out["verdict"]["kind"] == "no_signal"


def test_universe_restricts_the_contrast():
    idx, state, labels = _history(1000)
    mask = labels.eq("outperform")
    universe = labels.ne("match")
    rows = compare_conditions(mask, state, universe=universe)
    plain = compare_conditions(mask, state)
    # Outperform vs underperform only: a sharper contrast than outperform vs everything else.
    assert rows[0]["effect"] > plain[0]["effect"] > 0
    assert rows[0]["n_inside"] == int(mask.sum())


def test_score_is_bounded_and_smoothed():
    idx, state, labels = _history(1000)
    conditions = compare_conditions(labels.eq("outperform"), state)
    score, used = favorability_score(state, conditions, WINDOW)
    assert used and score.dropna().between(-1, 1).all()
    assert score.diff().abs().max() < 0.2  # a 63-day mean cannot jump


def test_favorable_share_inside_a_window():
    idx, state, labels = _history(1000)
    conditions = compare_conditions(labels.eq("outperform"), state)
    score, _ = favorability_score(state, conditions, WINDOW)
    cut = float(score.quantile(2 / 3))
    out = favorable_share(score, labels, cut, idx[100], idx[300])
    assert out["available"] and 0 <= out["favorable_share"] <= 1
    assert out["favorable_and_underperform_share"] <= out["favorable_share"]


def test_deterministic_and_json_safe():
    idx, state, labels = _history(1500)
    a, b = _check(state, labels), _check(state, labels)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert "NaN" not in json.dumps(a)  # browsers reject NaN in JSON
