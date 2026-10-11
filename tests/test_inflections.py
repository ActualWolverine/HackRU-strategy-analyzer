"""Alpha inflection detection: deterministic, and behaves on hand-checkable series."""

import numpy as np
import pandas as pd
import pytest

from inflections import describe_inflections, drop_short_legs, turning_points


def test_simple_peak_and_trough():
    # Up 10, down 10, up 10 (cumulative): one peak, one trough.
    x = np.concatenate([np.linspace(0, 0.10, 11), np.linspace(0.09, 0.0, 10), np.linspace(0.01, 0.10, 10)])
    assert turning_points(x, 0.05) == [(10, "peak"), (20, "trough")]


def test_small_wiggles_are_ignored():
    x = np.cumsum(np.tile([0.004, -0.003], 50))  # steady drift up with tiny reversals
    assert turning_points(x, 0.01) == []


def test_unconfirmed_final_extreme_not_reported():
    x = np.concatenate([np.linspace(0, 0.10, 11), np.linspace(0.099, 0.095, 5)])  # tops out, barely dips
    assert turning_points(x, 0.05) == []


def test_window_start_is_never_a_turn():
    x = np.linspace(0.0, -0.10, 11)  # falls from the first day
    assert turning_points(x, 0.02) == []


def test_turns_alternate():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.01, 2000))
    kinds = [k for _, k in turning_points(x, 0.03)]
    assert kinds and all(a != b for a, b in zip(kinds, kinds[1:]))


def test_short_legs_dropped_in_pairs():
    turns = [(10, "peak"), (12, "trough"), (40, "peak"), (60, "trough")]
    assert drop_short_legs(turns, 5) == [(40, "peak"), (60, "trough")]


@pytest.fixture
def synthetic():
    """A strategy that tracks the market with beta 1, then sits in cash, then re-enters."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    m = pd.Series(np.r_[np.full(40, 0.002), np.full(40, -0.004), np.full(40, 0.003)], index=idx)
    w = pd.DataFrame({"SPY": np.r_[np.ones(40), np.zeros(40), np.ones(40)]}, index=idx)
    s = w["SPY"].shift(1).fillna(0) * m
    rf = pd.Series(0.0, index=idx)
    state = pd.DataFrame({"realized_vol_20d": np.r_[np.full(40, 0.1), np.full(80, 0.4)]}, index=idx)
    return s, m, rf, w, state


def test_describe_inflections_on_synthetic(synthetic):
    s, m, rf, w, state = synthetic
    out = describe_inflections(s, m, rf, beta=0.5, weights=w, state=state.reindex(columns=["realized_vol_20d"]))
    # Alpha vs beta 0.5 (positions earn the next day's return):
    #   day 0 not yet invested: -0.5 * 0.002;   days 1-39 invested: +0.001 each
    #   day 40 still invested into the selloff: -0.004 + 0.002;   days 41-79 cash: +0.002 each
    #   day 80 not yet re-invested: -0.5 * 0.003;   days 81-119 invested: +0.0015 each
    # Alpha rises throughout apart from one-day dips, so no turning point.
    assert out["turning_points"] == []
    expected = -0.001 + 39 * 0.001 - 0.002 + 39 * 0.002 - 0.0015 + 39 * 0.0015
    assert out["window_alpha"] == pytest.approx(expected, rel=1e-9)

def test_turns_line_up_with_position_changes():
    """With beta 1.5, alpha falls while invested in a rally and rises while in cash during a selloff."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    m = pd.Series(np.r_[np.full(40, 0.006), np.full(40, -0.004), np.full(40, 0.003)], index=idx)
    w = pd.DataFrame({"SPY": np.r_[np.ones(40), np.zeros(40), np.ones(40)]}, index=idx)
    s = w["SPY"].shift(1).fillna(0) * m
    rf = pd.Series(0.0, index=idx)
    state = pd.DataFrame({"realized_vol_20d": np.r_[np.full(40, 0.1), np.full(80, 0.4)]}, index=idx)

    out = describe_inflections(s, m, rf, beta=1.5, weights=w, state=state)
    turns = out["turning_points"]
    assert [t["kind"] for t in turns] == ["trough", "peak"]
    # Positions earn the next day's return. Trough: day 39, the last day alpha
    # falls (day 40's return is earned while still invested, but in a selloff,
    # which beta 1.5 counts as a gain). Peak: day 79, the last selloff day in
    # cash; on day 80 the market rallies while the strategy is still in cash.
    assert turns[0]["date"] == idx[39].strftime("%Y-%m-%d")
    assert turns[1]["date"] == idx[79].strftime("%Y-%m-%d")
    sale = [x for x in turns[0]["trades_near"] if x["net_after"] == 0.0]
    assert sale and sale[0]["date"] == idx[40].strftime("%Y-%m-%d")
    assert turns[0]["leg_before"]["alpha"] < 0 < turns[0]["leg_after"]["alpha"]
    # Single asset that is the market itself: alpha is all timing, no selection.
    for leg in (turns[0]["leg_before"], turns[0]["leg_after"]):
        assert leg["alpha_timing"] == pytest.approx(leg["alpha"], abs=1e-12)
        assert leg["alpha_selection"] == pytest.approx(0.0, abs=1e-12)
    # The volatility regime change at day 40 is the reported market shift.
    assert turns[0]["market_shifts"][0]["indicator"] == "realized_vol_20d"


def test_timing_plus_selection_is_alpha_with_two_assets():
    """Holding a non-market asset: selection = alpha - timing, and both are exact."""
    idx = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0.0005, 0.01, 60), index=idx)
    other = pd.Series(rng.normal(0.0, 0.01, 60), index=idx)
    rf = pd.Series(0.0001, index=idx)
    w = pd.DataFrame({"SPY": np.r_[np.full(30, 0.6), np.full(30, 0.2)], "GLD": np.r_[np.full(30, 0.4), np.full(30, 0.7)]}, index=idx)
    held = w.shift(1).fillna(0.0)
    s = held["SPY"] * m + held["GLD"] * other + (1 - held.sum(axis=1)) * rf
    beta = 0.5
    out = describe_inflections(s, m, rf, beta, w, pd.DataFrame(index=idx))
    alpha = (s - rf) - beta * (m - rf)
    timing = (held.sum(axis=1) - beta) * (m - rf)
    selection = held["GLD"] * (other - rf) + held["SPY"] * (m - rf) - held.sum(axis=1) * (m - rf)
    assert out["window_alpha"] == pytest.approx(alpha.sum(), rel=1e-12)
    assert out["window_alpha_timing"] == pytest.approx(timing.sum(), rel=1e-12)
    assert out["window_alpha"] - out["window_alpha_timing"] == pytest.approx(selection.sum(), rel=1e-9)


def test_describe_inflections_is_deterministic(synthetic):
    s, m, rf, w, state = synthetic
    a = describe_inflections(s, m, rf, 1.5, w, state)
    b = describe_inflections(s, m, rf, 1.5, w, state)
    assert a == b
