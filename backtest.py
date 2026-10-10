"""Backtest engine.

A strategy is any callable `(closes: DataFrame, **params) -> DataFrame` that
returns a weights table (same index as `closes`, one column per ticker). A
weight is the fraction of capital in that ticker at the close of each day:
+1 is fully long, -1 fully short, 0 flat. The engine applies the weights to the
*next* day's returns, so a strategy can only trade on information it has at
the close. Nothing is precomputed; every run is from scratch.
"""

import numpy as np
import pandas as pd

from data_loader import get_closes, load_market_state
from regime import compare_conditions

TRADING_DAYS = 252
LABEL_WINDOW = 63  # ~3 months: rolling window used to judge relative performance
MIN_RUN = 10       # regimes shorter than this (days) are absorbed into their neighbours
MIN_EXCESS = 0.005 # excess return over the window below this (abs) always counts as "match"


def run(weights: pd.DataFrame, closes: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """Daily strategy returns from a weights table.

    Weights at the close of day t earn the return from t to t+1. Transaction
    cost is charged per unit of turnover.
    """
    returns = closes[weights.columns].pct_change()
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * cost_bps / 10_000


def metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
        "volatility": float(vol),
        "sharpe": float(returns.mean() * TRADING_DAYS / vol) if vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
    }


def label_relative_performance(strategy: pd.Series, benchmark: pd.Series, window: int = LABEL_WINDOW) -> pd.Series:
    """Label each day outperform / match / underperform.

    Uses the strategy's excess return over the benchmark across a window
    centred on the day, so a day's label describes the regime it sits in. The
    top and bottom terciles of that excess return get the extreme labels, so
    every strategy has three populated buckets to compare. Days whose excess
    is within +-MIN_EXCESS are always "match": a strategy that simply holds
    the benchmark part of the time has exact zeros on those days, which would
    otherwise collapse the tercile cut-points onto zero.
    """
    log_excess = np.log1p(strategy) - np.log1p(benchmark)
    rolling = log_excess.rolling(window, center=True).sum()
    lo, hi = rolling.quantile([1 / 3, 2 / 3])
    labels = pd.Series("match", index=rolling.index)
    labels[(rolling <= lo) & (rolling < -MIN_EXCESS)] = "underperform"
    labels[(rolling >= hi) & (rolling > MIN_EXCESS)] = "outperform"
    labels[rolling.isna()] = None
    return smooth_labels(labels)


def smooth_labels(labels: pd.Series, min_run: int = MIN_RUN) -> pd.Series:
    """Absorb runs shorter than `min_run` into the preceding regime.

    Tercile labels flicker day to day; without this the chart shading is
    hundreds of 2px stripes. Repeats until no short runs remain.
    """
    labels = labels.copy()
    for _ in range(10):
        valid = labels.notna()
        run_id = labels.ne(labels.shift()).cumsum()
        run_len = labels.groupby(run_id).transform("size")
        short = valid & (run_len < min_run)
        if not short.any():
            break
        labels[short] = None
        labels[valid] = labels[valid].ffill()
    return labels


def label_runs(labels: pd.Series) -> list[dict]:
    """Collapse a per-day label series into contiguous [start, end] runs."""
    runs = []
    labels = labels.dropna()
    if labels.empty:
        return runs
    change = labels.ne(labels.shift()).cumsum()
    for _, grp in labels.groupby(change):
        runs.append({
            "label": grp.iloc[0],
            "start": grp.index[0].strftime("%Y-%m-%d"),
            "end": grp.index[-1].strftime("%Y-%m-%d"),
        })
    return runs


def backtest(strategy, tickers: list[str], benchmark: str, params: dict, start=None, end=None) -> dict:
    """Run a strategy, benchmark it, label regimes, and describe each regime's conditions."""
    needed = set(tickers) | ({benchmark} if benchmark != "CASH" else set())
    closes = get_closes(sorted(needed), start, end).dropna()
    weights = strategy(closes[tickers], **params)
    weights = weights.reindex(closes.index).fillna(0.0)

    strat_returns = run(weights, closes)
    # "CASH" benchmarks absolute return (zero), which suits market-neutral strategies.
    bench_returns = (closes[benchmark].pct_change() if benchmark != "CASH"
                     else pd.Series(0.0, index=closes.index)).fillna(0.0)

    # Start the comparison on the first day the strategy actually holds something.
    first = weights.abs().sum(axis=1).gt(0).idxmax()
    strat_returns, bench_returns = strat_returns.loc[first:], bench_returns.loc[first:]
    strat_equity = 100 * (1 + strat_returns).cumprod()
    bench_equity = 100 * (1 + bench_returns).cumprod()

    labels = label_relative_performance(strat_returns, bench_returns)
    state = load_market_state()
    conditions = {
        lab: compare_conditions(labels.eq(lab), state) for lab in ("outperform", "underperform")
    }

    dates = strat_equity.index.strftime("%Y-%m-%d").tolist()
    return {
        "dates": dates,
        "strategy_equity": strat_equity.round(3).tolist(),
        "benchmark_equity": bench_equity.round(3).tolist(),
        "benchmark": benchmark,
        "metrics": {"strategy": metrics(strat_returns), "benchmark": metrics(bench_returns)},
        "label_window": LABEL_WINDOW,
        "regimes": label_runs(labels),
        "regime_days": labels.value_counts().to_dict(),
        "conditions": conditions,
        "diagnostics": getattr(weights, "attrs", {}).get("diagnostics", {}),
    }
