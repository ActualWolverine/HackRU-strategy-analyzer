"""Describe what was different about a set of trading days.

The core primitive is `compare_conditions(mask)`: for every market-state
indicator it contrasts the days where `mask` is True against all other days
and ranks the indicators by how unusual the selected days were. It is used
both for a hand-selected chart region and, later, for the day-labels a
backtest produces (outperform / match / underperform).
"""

import numpy as np
import pandas as pd

from data_loader import load_market_state

# Display metadata, keyed by column name in market_state.parquet.
INDICATORS = {
    "realized_vol_20d":     {"label": "Realized volatility (20d)",  "fmt": "pct",   "high": "volatile",         "low": "calm"},
    "efficiency_ratio_60d": {"label": "Efficiency ratio (60d)",     "fmt": "ratio", "high": "trending cleanly", "low": "choppy"},
    "autocorr_60d":         {"label": "Return autocorrelation (60d)","fmt": "ratio", "high": "momentum-driven",  "low": "mean-reverting"},
    "trend_200d":           {"label": "Distance from 200d average", "fmt": "pct",   "high": "extended uptrend", "low": "below trend"},
    "drawdown_52w":         {"label": "Drawdown from 52w high",     "fmt": "pct",   "high": "near highs",       "low": "deep drawdown"},
    "momentum_12_1":        {"label": "12-1 month momentum",        "fmt": "pct",   "high": "stronger momentum", "low": "weaker momentum",
                             "formula": "Close(t-21) / Close(t-252) - 1; 21 and 252 are trading-day approximations."},
    "sector_breadth_200d":  {"label": "Sectors above 200d average", "fmt": "pct",   "high": "broad participation", "low": "narrow participation",
                             "formula": "Count(sector close > its 200-day average) / count(sectors with valid data)."},
    "adx_14d":              {"label": "Trend strength (ADX 0-100)", "fmt": "ratio", "high": "stronger trend", "low": "weaker trend",
                             "formula": "ADX = Wilder-smoothed DX; DX = 100 * abs(+DI - -DI) / (+DI + -DI), with +DI/-DI derived from 14-day-smoothed directional movement and true range."},
    "sector_correlation_60d": {"label": "Sector correlation (60d)", "fmt": "ratio", "high": "sectors moving together", "low": "sectors more independent",
                             "formula": "Mean of rolling 60-day Pearson correlations for all pairs of daily sector returns."},
    "yield_curve_10y2y":     {"label": "Treasury yield curve (10y-2y)", "fmt": "level", "high": "steeper yield curve", "low": "flatter or inverted yield curve",
                             "formula": "10-year Treasury yield (DGS10) - 2-year Treasury yield (DGS2), in percentage points; FRED series T10Y2Y."},
    "vix_term_structure":   {"label": "VIX / VIX3M",                "fmt": "ratio", "high": "fear (inverted)",  "low": "complacent"},
    "credit_spread":        {"label": "Credit spread (Baa - 10Y)",  "fmt": "level", "high": "credit stress",    "low": "easy credit"},
    "fed_funds_3m_change":  {"label": "Fed funds 3-month change",   "fmt": "level", "high": "tightening",       "low": "easing"},
}

# Rank-biserial effect size thresholds (|r|). Below "small" the indicator is reported as typical.
EFFECT_BANDS = [(0.6, "very different"), (0.4, "different"), (0.2, "slightly different"), (0.0, "typical")]


def rank_biserial(inside: pd.Series, outside: pd.Series) -> float:
    """Effect size in [-1, 1]: P(inside > outside) - P(inside < outside).

    Equivalent to 2*AUC - 1 from the Mann-Whitney U statistic. Robust to the
    heavy tails these indicators have, unlike a mean difference.
    """
    n_in, n_out = len(inside), len(outside)
    if n_in == 0 or n_out == 0:
        return float("nan")
    ranks = pd.concat([inside, outside]).rank()
    u = ranks.iloc[:n_in].sum() - n_in * (n_in + 1) / 2
    return float(2 * u / (n_in * n_out) - 1)


def compare_conditions(mask: pd.Series, state: pd.DataFrame | None = None,
                       universe: pd.Series | None = None) -> list[dict]:
    """Contrast the days where `mask` is True against the rest of history.

    With `universe`, the contrast is against the other days inside `universe`
    only (e.g. failures vs. successes among favorable days). Returns one dict
    per indicator, sorted by |effect| descending.
    """
    if state is None:
        state = load_market_state()
    mask = mask.reindex(state.index, fill_value=False).astype(bool)
    universe = (pd.Series(True, index=state.index) if universe is None
                else universe.reindex(state.index, fill_value=False).astype(bool))

    rows = []
    for col, meta in INDICATORS.items():
        if col not in state.columns:
            continue
        series = state[col].dropna()
        m = mask.reindex(series.index, fill_value=False)
        u = universe.reindex(series.index, fill_value=False)
        inside, outside = series[m & u], series[~m & u]
        if len(inside) < 5:
            continue
        effect = rank_biserial(inside, outside)
        inside_median = float(inside.median())
        band = next(name for threshold, name in EFFECT_BANDS if abs(effect) >= threshold)
        rows.append({
            "indicator": col,
            "label": meta["label"],
            "fmt": meta["fmt"],
            "formula": meta.get("formula", ""),
            "n_inside": int(len(inside)),
            "inside_median": inside_median,
            "baseline_median": float(outside.median()),
            "inside_p10": float(inside.quantile(0.1)),
            "inside_p90": float(inside.quantile(0.9)),
            # Where the selected period's typical value sits in all of history.
            "percentile": float((series < inside_median).mean() * 100),
            "effect": effect,
            "band": band,
            "direction": meta["high"] if effect > 0 else meta["low"],
        })
    rows.sort(key=lambda r: abs(r["effect"]), reverse=True)
    return rows


def describe_period(start: str, end: str) -> dict:
    """Conditions for a date range, for the region-select UI."""
    state = load_market_state()
    mask = (state.index >= pd.Timestamp(start)) & (state.index <= pd.Timestamp(end))
    mask = pd.Series(mask, index=state.index)
    return {
        "start": start,
        "end": end,
        "n_days": int(mask.sum()),
        "indicators": compare_conditions(mask, state),
    }
