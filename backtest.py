"""Backtest engine.

A strategy is any callable `(closes: DataFrame, **params) -> DataFrame` that
returns a weights table (same index as `closes`, one column per ticker). A
weight is the fraction of capital in that ticker at the close of each day:
+1 is fully long, -1 fully short, 0 flat. The engine applies the weights to the
*next* day's returns, so a strategy can only trade on information it has at
the close. Nothing is precomputed; every run is from scratch.

Capital not deployed (1 - sum of weights) earns the risk-free rate, and
leverage (sum of weights > 1) pays it, so a strategy that sits in cash is
credited with T-bill-like interest rather than zero.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data_loader import get_closes, get_macro_daily, load_market_state
from regime import compare_conditions

TRADING_DAYS = 252
LABEL_WINDOW = 63  # ~3 months: rolling window used to judge relative performance
MIN_RUN = 10       # regimes shorter than this (days) are absorbed into their neighbours
MIN_EXCESS = 0.005 # excess return over the window below this (abs) always counts as "match"
MIN_ALPHA_YEARS = 2  # don't report alpha for a sub-period shorter than this
SIGNIFICANT_T = 3.0  # Harvey-Liu-Zhu (2016) bar for a finding chosen from many candidates


def risk_free_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free return from the effective Fed funds rate (FRED DFF)."""
    dff = get_macro_daily()["DFF"].reindex(index).ffill().bfill()
    return (1 + dff / 100) ** (1 / TRADING_DAYS) - 1


def run(weights: pd.DataFrame, closes: pd.DataFrame, rf: pd.Series | None = None, cost_bps: float = 0.0) -> pd.Series:
    """Daily strategy returns from a weights table.

    Weights at the close of day t earn the return from t to t+1. The
    undeployed balance (1 - net weight) earns `rf`.

    Transaction cost is charged at each close on turnover: the change from
    the pre-trade weights (yesterday's holdings after today's price moves)
    to today's targets, as a fraction of pre-trade equity. Costs reduce
    equity before the new targets are applied.
    """
    returns = closes[weights.columns].pct_change().fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    if rf is not None:
        gross += (1 - held.sum(axis=1)) * rf
    if not cost_bps:
        return gross
    # Holdings drift with prices between rebalances, so what's actually traded
    # is the gap between today's target and the drifted weight, not w_t - w_{t-1}.
    drifted = held.mul(1 + returns).div(1 + gross, axis=0)
    turnover = (weights - drifted).abs().sum(axis=1)
    return (1 + gross) * (1 - turnover * cost_bps / 10_000) - 1


def metrics(returns: pd.Series, rf: pd.Series) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    excess = returns - rf.reindex(returns.index)
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    excess_vol = excess.std() * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
        "volatility": float(vol),
        "sharpe": float(excess.mean() * TRADING_DAYS / excess_vol) if excess_vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
    }


def alpha_stats(strategy: pd.Series, market: pd.Series, rf: pd.Series) -> dict | None:
    """CAPM regression of strategy excess returns on market excess returns.

    alpha is the annualized intercept: return not explained by market
    exposure. The t-stat uses Newey-West standard errors because daily
    strategy returns are autocorrelated and heteroskedastic.
    """
    df = pd.DataFrame({"s": strategy - rf, "m": market - rf}).dropna()
    if len(df) < MIN_ALPHA_YEARS * TRADING_DAYS:
        return None
    fit = sm.OLS(df["s"], sm.add_constant(df["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    alpha = float(fit.params["const"] * TRADING_DAYS)
    residual_vol = float(fit.resid.std() * np.sqrt(TRADING_DAYS))
    return {
        "alpha": alpha,
        "t_stat": float(fit.tvalues["const"]),
        "beta": float(fit.params["m"]),
        "info_ratio": alpha / residual_vol if residual_vol > 0 else float("nan"),
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
        "years": round(len(df) / TRADING_DAYS, 1),
    }


def publication_test(strategy: pd.Series, market: pd.Series, rf: pd.Series, published: int | None) -> dict:
    """Alpha over the full sample, and before vs. after the strategy was published.

    Anomalies that were real but got arbitraged away show alpha before
    publication and much less after; McLean & Pontiff (2016) find average
    decay of roughly half. Periods shorter than MIN_ALPHA_YEARS report None.
    """
    result = {"full": alpha_stats(strategy, market, rf), "published": published, "before": None, "after": None}
    if published is not None:
        cut = pd.Timestamp(f"{published}-01-01")
        result["before"] = alpha_stats(strategy[strategy.index < cut], market, rf)
        result["after"] = alpha_stats(strategy[strategy.index >= cut], market, rf)
    return result


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


def period_stats(strategy, tickers: list[str], benchmark: str, params: dict, start: str, end: str) -> dict:
    """Deterministic summary of a strategy's behaviour inside one date window.

    Recomputes the full backtest (so the window sees the same warm-up and the
    same regime labels as the chart), then slices. Input to the AI explainer;
    contains no price or trade series.
    """
    market = benchmark if benchmark != "CASH" else "SPY"
    closes = get_closes(sorted(set(tickers) | {market})).dropna()
    rf = risk_free_daily(closes.index)
    weights = strategy(closes[tickers], **params).reindex(closes.index).fillna(0.0)
    strat = run(weights, closes, rf)
    bench = closes[market].pct_change().fillna(0.0) if benchmark != "CASH" else rf
    first = weights.abs().sum(axis=1).gt(0).idxmax()
    labels = label_relative_performance(strat.loc[first:], bench.loc[first:])

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    window = (strat.index >= lo) & (strat.index <= hi) & (strat.index >= first)
    if window.sum() < 2:
        raise ValueError("selected period has fewer than two trading days of backtest data")
    s, b, w = strat[window], bench[window], weights[window]

    def total(r: pd.Series) -> float:
        return float((1 + r).prod() - 1)

    def max_dd(r: pd.Series) -> float:
        eq = (1 + r).cumprod()
        return float((eq / eq.cummax() - 1).min())

    gross = w.abs().sum(axis=1)
    share = labels.reindex(s.index).value_counts(normalize=True) * 100
    return {
        "start": s.index[0].strftime("%Y-%m-%d"), "end": s.index[-1].strftime("%Y-%m-%d"),
        "n_days": int(window.sum()), "benchmark": benchmark,
        "strategy_return": total(s), "benchmark_return": total(b), "excess_return": total(s) - total(b),
        "strategy_max_drawdown": max_dd(s), "benchmark_max_drawdown": max_dd(b),
        "strategy_volatility": float(s.std() * np.sqrt(TRADING_DAYS)),
        "benchmark_volatility": float(b.std() * np.sqrt(TRADING_DAYS)),
        "avg_gross_exposure": float(gross.mean()),
        "days_invested_pct": float((gross > 0).mean() * 100),
        "position_changes": int((w.diff().abs().sum(axis=1) > 1e-9).sum()),
        "best_day": float(s.max()), "worst_day": float(s.min()),
        "regime_share": {k: float(v) for k, v in share.items()},
    }


def backtest(strategy, tickers: list[str], benchmark: str, params: dict, start=None, end=None,
             published: int | None = None, describe: bool = True) -> dict:
    """Run a strategy, benchmark it, measure alpha, label regimes, and describe each regime's conditions.

    Alpha is measured against the benchmark, or against SPY when the
    benchmark is cash (alpha needs a market to regress on). `describe=False`
    skips the regime labelling and conditions analysis, for bulk runs.
    """
    market = benchmark if benchmark != "CASH" else "SPY"
    closes = get_closes(sorted(set(tickers) | {market}), start, end).dropna()
    rf = risk_free_daily(closes.index)
    weights = strategy(closes[tickers], **params)
    weights = weights.reindex(closes.index).fillna(0.0)

    strat_returns = run(weights, closes, rf)
    market_returns = closes[market].pct_change().fillna(0.0)
    # "CASH" benchmarks the risk-free rate, which suits market-neutral strategies.
    bench_returns = market_returns if benchmark != "CASH" else rf

    # Start the comparison on the first day the strategy actually holds something.
    first = weights.abs().sum(axis=1).gt(0).idxmax()
    strat_returns, bench_returns = strat_returns.loc[first:], bench_returns.loc[first:]
    market_returns, rf = market_returns.loc[first:], rf.loc[first:]

    result = {
        "benchmark": benchmark,
        "market": market,
        "metrics": {"strategy": metrics(strat_returns, rf), "benchmark": metrics(bench_returns, rf)},
        "alpha": publication_test(strat_returns, market_returns, rf, published),
        "diagnostics": getattr(weights, "attrs", {}).get("diagnostics", {}),
    }
    if not describe:
        return result

    labels = label_relative_performance(strat_returns, bench_returns)
    state = load_market_state()
    strat_equity = 100 * (1 + strat_returns).cumprod()
    bench_equity = 100 * (1 + bench_returns).cumprod()
    return {
        **result,
        "dates": strat_equity.index.strftime("%Y-%m-%d").tolist(),
        "strategy_equity": strat_equity.round(3).tolist(),
        "benchmark_equity": bench_equity.round(3).tolist(),
        "label_window": LABEL_WINDOW,
        "regimes": label_runs(labels),
        "regime_days": labels.value_counts().to_dict(),
        "conditions": {
            lab: compare_conditions(labels.eq(lab), state) for lab in ("outperform", "underperform")
        },
    }
