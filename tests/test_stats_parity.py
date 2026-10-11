"""stats.py must reproduce statsmodels exactly.

statsmodels is a dev-only dependency (requirements-dev.txt); the deployed app
uses the numpy implementations in stats.py. Skipped if statsmodels is absent.
"""

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")
from statsmodels.tsa.stattools import coint  # noqa: E402

import backtest as bt  # noqa: E402
from data_loader import get_closes  # noqa: E402
from stats import engle_granger, ols_hac  # noqa: E402
from strategies import STRATEGIES, universe_for  # noqa: E402


def strategy_returns(key):
    spec = STRATEGIES[key]
    params = {p["name"]: p["default"] for p in spec["params"]}
    tickers = universe_for(key, params)
    closes = get_closes(sorted(set(tickers) | {"SPY"})).dropna()
    rf = bt.risk_free_daily(closes.index)
    w = spec["fn"](closes[tickers], **params).reindex(closes.index).fillna(0.0)
    return bt.run(w, closes, rf), closes["SPY"].pct_change(), rf


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_hac_regression_matches_statsmodels(key):
    s, m, rf = strategy_returns(key)
    df = pd.DataFrame({"s": s - rf, "m": m - rf}).dropna()
    ref = sm.OLS(df["s"], sm.add_constant(df["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    params, tvalues = ols_hac(df["s"].to_numpy(), np.column_stack([np.ones(len(df)), df["m"].to_numpy()]), 10)
    np.testing.assert_allclose(params, ref.params.to_numpy(), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(tvalues, ref.tvalues.to_numpy(), rtol=1e-9)


PAIRS = [("XLY", "XLV"), ("QQQ", "XLK"), ("IVV", "SPY"), ("EFA", "EEM"), ("TLT", "IEF"), ("XLU", "XLB")]


@pytest.mark.parametrize("y,x", PAIRS)
def test_engle_granger_matches_statsmodels(y, x):
    c = np.log(get_closes([y, x]).dropna())
    t_ref, p_ref, _ = coint(c[y], c[x])
    t, p = engle_granger(c[y], c[x])
    if np.isinf(t_ref):
        assert np.isinf(t) and p == p_ref
    else:
        assert t == pytest.approx(t_ref, rel=1e-9)
        assert p == pytest.approx(p_ref, rel=1e-9, abs=1e-12)


def test_engle_granger_matches_on_short_and_random_series():
    rng = np.random.default_rng(0)
    for n in (60, 250, 1000):
        x = np.cumsum(rng.normal(size=n))
        for y in (x + rng.normal(size=n), np.cumsum(rng.normal(size=n))):  # cointegrated, then not
            t_ref, p_ref, _ = coint(y, x)
            t, p = engle_granger(y, x)
            assert t == pytest.approx(t_ref, rel=1e-9)
            assert p == pytest.approx(p_ref, rel=1e-9, abs=1e-12)
