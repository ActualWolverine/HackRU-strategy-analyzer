"""Cash reconciliation: the vectorized engine must agree with explicit bookkeeping.

`backtest.run()` computes returns as weighted sums. This test re-runs every
strategy through an independent ledger that holds actual share counts and a
cash balance: it buys and sells shares at each close, credits short-sale
proceeds to cash, accrues interest on cash (or pays it on a negative balance,
i.e. leverage), and pays transaction costs out of equity. The two equity
curves must match.
"""

import numpy as np
import pandas as pd
import pytest

import backtest as bt
from data_loader import get_closes
from strategies import STRATEGIES, universe_for

START_EQUITY = 100.0
COSTS_BPS = [0.0, 10.0]


def default_params(key: str) -> dict:
    return {p["name"]: p["default"] for p in STRATEGIES[key]["params"]}


def inputs(key: str):
    params = default_params(key)
    tickers = universe_for(key, params)
    closes = get_closes(sorted(set(tickers))).dropna()
    weights = STRATEGIES[key]["fn"](closes[tickers], **params).reindex(closes.index).fillna(0.0)
    rf = bt.risk_free_daily(closes.index)
    return weights, closes, rf


def ledger(weights: pd.DataFrame, closes: pd.DataFrame, rf: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Share-and-cash bookkeeping, one day at a time.

    Each day: cash accrues interest; positions are marked at today's close;
    trading cost is charged on the turnover from the drifted (pre-trade)
    weights to today's targets; then shares are set so each position is
    target_weight x post-cost equity, and cash is whatever is left.
    """
    prices = closes[weights.columns].to_numpy()
    targets = weights.to_numpy()
    rates = rf.to_numpy()
    shares = np.zeros(prices.shape[1])
    cash = START_EQUITY
    rows = []
    for t in range(len(prices)):
        cash *= 1 + rates[t]
        position_value = shares * prices[t]
        equity_pre = cash + position_value.sum()

        turnover = np.abs(targets[t] - position_value / equity_pre).sum()
        cost = turnover * cost_bps / 10_000 * equity_pre
        equity = equity_pre - cost

        shares = targets[t] * equity / prices[t]
        cash = equity - (shares * prices[t]).sum()
        rows.append((cash, (shares * prices[t]).sum(), cost, equity))

        # Bookkeeping invariant: equity is exactly cash plus marked positions.
        assert np.isclose(equity, cash + (shares * prices[t]).sum(), rtol=1e-12, atol=1e-9)
    return pd.DataFrame(rows, index=weights.index, columns=["cash", "positions", "cost", "equity"])


def engine_equity(weights, closes, rf, cost_bps) -> pd.Series:
    return START_EQUITY * (1 + bt.run(weights, closes, rf, cost_bps)).cumprod()


@pytest.mark.parametrize("cost_bps", COSTS_BPS)
@pytest.mark.parametrize("key", list(STRATEGIES))
def test_engine_matches_ledger(key, cost_bps):
    weights, closes, rf = inputs(key)
    book = ledger(weights, closes, rf, cost_bps)
    engine = engine_equity(weights, closes, rf, cost_bps)
    np.testing.assert_allclose(engine.to_numpy(), book["equity"].to_numpy(), rtol=1e-9)
    if cost_bps:
        assert book["cost"].sum() > 0, f"{key}: costs were configured but nothing was charged"


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_backtest_equity_matches_ledger(key):
    """The equity curve the app shows (trimmed to the first trade, rebased to 100) reconciles too."""
    spec = STRATEGIES[key]
    params = default_params(key)
    result = bt.backtest(spec["fn"], universe_for(key, params), spec["benchmark"], params)

    # backtest() loads the strategy's tickers plus the benchmark, and drops dates
    # where any are missing, so rebuild exactly that input.
    market = spec["benchmark"] if spec["benchmark"] != "CASH" else "SPY"
    tickers = universe_for(key, params)
    closes = get_closes(sorted(set(tickers) | {market})).dropna()
    weights = spec["fn"](closes[tickers], **params).reindex(closes.index).fillna(0.0)
    first = weights.abs().sum(axis=1).gt(0).idxmax()
    rf = bt.risk_free_daily(closes.index)

    book = ledger(weights.loc[first:], closes.loc[first:], rf.loc[first:], 0.0)
    assert result["dates"] == book.index.strftime("%Y-%m-%d").tolist()
    # The app rounds equity to 3 decimals.
    np.testing.assert_allclose(result["strategy_equity"], book["equity"].round(3).to_numpy(), atol=1e-3)


def test_leverage_and_shorts_hit_cash():
    """Spot-check that the ledger's cash really moves the way the strategies imply."""
    # Vol-managed borrows above 1x: cash goes negative on those days.
    weights, closes, rf = inputs("vol_managed")
    book = ledger(weights, closes, rf, 0.0)
    levered = weights.sum(axis=1) > 1
    assert levered.any() and (book.loc[levered, "cash"] < 0).all()

    # Bollinger shorts the upper band at -1x: the sale proceeds land in cash,
    # so cash is equity plus the short's value, i.e. 2x equity at the close.
    weights, closes, rf = inputs("bollinger_reversion")
    book = ledger(weights, closes, rf, 0.0)
    short = weights.sum(axis=1) == -1
    assert short.any()
    np.testing.assert_allclose(book.loc[short, "cash"], 2 * book.loc[short, "equity"], rtol=1e-12)
    np.testing.assert_allclose(book.loc[short, "positions"], -book.loc[short, "equity"], rtol=1e-12)


def test_reconciliation_catches_missing_interest():
    """Control: an engine that forgot to pay interest on cash must fail to reconcile."""
    weights, closes, rf = inputs("sma_trend")
    book = ledger(weights, closes, rf, 0.0)
    broken = START_EQUITY * (1 + bt.run(weights, closes, rf=None)).cumprod()
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(broken.to_numpy(), book["equity"].to_numpy(), rtol=1e-9)


def test_reconciliation_catches_naive_turnover():
    """Control: the old cost formula (|w_t - w_{t-1}|, ignoring price drift) must fail to reconcile."""
    weights, closes, rf = inputs("risk_parity")
    cost_bps = 10.0
    book = ledger(weights, closes, rf, cost_bps)
    gross = bt.run(weights, closes, rf)
    naive_turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    naive = START_EQUITY * ((1 + gross) * (1 - naive_turnover * cost_bps / 10_000)).cumprod()
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(naive.to_numpy(), book["equity"].to_numpy(), rtol=1e-9)
