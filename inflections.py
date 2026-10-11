"""Inflection points in a strategy's alpha.

Daily alpha is the return left after removing market exposure, using the same
CAPM fit as the alpha table: alpha_t = (r_t - rf_t) - beta * (m_t - rf_t), with
the strategy's full-sample beta. Summed over the full backtest it reproduces
the reported alpha, so a turn in cumulative alpha is a turn in the number the
user is reading. Raw strategy-minus-benchmark would mostly show beta instead:
a strategy sitting in cash trails a rising market without losing any alpha.

Inside a selected window, cumulative alpha is split into legs by a zigzag
filter: a peak or trough counts as a turning point only once the line has
reversed from it by at least `min_swing`. Small wiggles are ignored, legs
shorter than MIN_LEG_DAYS are merged away, and the last extreme is not
reported unless a reversal confirmed it.

For each turning point the engine records, deterministically:
* the leg before and after it (alpha earned, strategy and market returns),
  with alpha split exactly into two parts (cash earns the risk-free rate, so
  strategy excess return = sum_i held_i * (r_i - rf)):
    timing    = (net exposure - beta) * (m - rf)   more or less market than beta assumes
    selection = sum_i held_i * (r_i - rf) - net exposure * (m - rf)   assets that beat or lagged the market
  timing + selection = alpha, before transaction costs (the app runs without costs),
* the strategy's net exposure just before vs. just after,
* position changes within a few trading days of the turn,
* the market-state indicators that shifted most around the turn, measured as
  the change in their 10-day average before vs. after, in units of the
  indicator's full-history standard deviation.

This is descriptive: "after" uses data after the turn on purpose, to describe
what changed. Nothing here feeds back into a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime import INDICATORS

MAX_TURNS = 5        # report at most this many turning points (the largest swings)
MIN_SWING = 0.01     # never smaller than a 1% reversal in relative performance
SWING_FRACTION = 0.2 # ... and at least 20% of the window's full relative range
MIN_LEG_DAYS = 5     # a peak and trough closer than this are noise; both are dropped
TRADE_WINDOW = 5     # trading days either side of a turn to look for position changes
MIN_EXPOSURE_CHANGE = 0.1  # net exposure change that counts as a trade on its own
STATE_WINDOW = 10    # trading days averaged before/after for indicator shifts
TOP_SHIFTS = 3       # indicators reported per turning point


def turning_points(x: np.ndarray, min_swing: float) -> list[tuple[int, str]]:
    """Zigzag turning points of `x`: [(index, "peak" | "trough")], chronological.

    A point is a peak (trough) if the series later falls (rises) from it by at
    least `min_swing` before making a new high (low). The window's first point
    and an unconfirmed final extreme are never reported.
    """
    n = len(x)
    if n < 3:
        return []
    turns: list[tuple[int, str]] = []
    trend, ext, hi, lo = 0, 0, 0, 0
    for i in range(1, n):
        if trend == 0:
            if x[i] > x[hi]:
                hi = i
            if x[i] < x[lo]:
                lo = i
            if x[hi] - x[lo] >= min_swing:
                # Direction is set by whichever extreme came last. The earlier
                # one is a turn only if the line also moved into it from the
                # window's start by at least min_swing; otherwise it is just
                # the start of the first leg.
                if hi > lo:
                    trend, ext = 1, hi
                    if lo > 0 and x[0] - x[lo] >= min_swing:
                        turns.append((lo, "trough"))
                else:
                    trend, ext = -1, lo
                    if hi > 0 and x[hi] - x[0] >= min_swing:
                        turns.append((hi, "peak"))
        elif trend == 1:
            if x[i] > x[ext]:
                ext = i
            elif x[ext] - x[i] >= min_swing:
                turns.append((ext, "peak"))
                trend, ext = -1, i
        else:
            if x[i] < x[ext]:
                ext = i
            elif x[i] - x[ext] >= min_swing:
                turns.append((ext, "trough"))
                trend, ext = 1, i
    return turns


def drop_short_legs(turns: list[tuple[int, str]], min_days: int) -> list[tuple[int, str]]:
    """Remove peak/trough pairs closer than `min_days`; removing both keeps peaks and troughs alternating."""
    turns = list(turns)
    changed = True
    while changed:
        changed = False
        for k in range(len(turns) - 1):
            if turns[k + 1][0] - turns[k][0] < min_days:
                del turns[k: k + 2]
                changed = True
                break
    return turns


def _leg(s: pd.Series, m: pd.Series, a: pd.Series, timing: pd.Series | None, i0: int, i1: int) -> dict:
    """Over (i0, i1], from one pivot's close to the next: alpha earned, its split, and raw returns."""
    sl = slice(i0 + 1, i1 + 1)
    leg = {"start": s.index[i0].strftime("%Y-%m-%d"), "end": s.index[i1].strftime("%Y-%m-%d"),
           "days": int(i1 - i0), "alpha": float(a.iloc[sl].sum()),
           "strategy_return": float((1 + s.iloc[sl]).prod() - 1),
           "market_return": float((1 + m.iloc[sl]).prod() - 1)}
    if timing is not None:
        leg["alpha_timing"] = float(timing.iloc[sl].sum())
        leg["alpha_selection"] = leg["alpha"] - leg["alpha_timing"]
    return leg


def _trades_near(weights: pd.DataFrame, at: pd.Timestamp) -> list[dict]:
    pos = weights.index.get_loc(at)
    lo, hi = max(1, pos - TRADE_WINDOW), min(len(weights) - 1, pos + TRADE_WINDOW)
    trades = []
    for i in range(lo, hi + 1):
        before, after = weights.iloc[i - 1], weights.iloc[i]
        # A trade is a real change of position: an asset entered, exited or
        # flipped side, or net exposure moved materially. Daily re-weighting
        # (vol targeting, inverse-vol sizing) is not reported.
        flipped = np.sign(before) != np.sign(after)
        net_move = abs(after.sum() - before.sum()) >= MIN_EXPOSURE_CHANGE
        if flipped.any() or net_move:
            delta = after - before
            changed = delta.abs().sort_values(ascending=False).index[:3]
            trades.append({
                "date": weights.index[i].strftime("%Y-%m-%d"),
                "offset_days": int(i - pos),
                "net_before": float(before.sum()), "net_after": float(after.sum()),
                "changes": {t: [float(before[t]), float(after[t])] for t in changed},
            })
    return trades


def _state_shifts(state: pd.DataFrame, at: pd.Timestamp) -> list[dict]:
    pos = state.index.searchsorted(at)
    before = state.iloc[max(0, pos - STATE_WINDOW): pos]
    after = state.iloc[pos + 1: pos + 1 + STATE_WINDOW]
    if before.empty or after.empty:
        return []
    std = state.std()
    shifts = []
    for col, meta in INDICATORS.items():
        if col not in state.columns:
            continue
        b, a, sd = before[col].mean(), after[col].mean(), std[col]
        if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(sd) and sd > 0):
            continue
        shifts.append({"indicator": col, "label": meta["label"], "fmt": meta["fmt"],
                       "before": float(b), "after": float(a), "change_std": float((a - b) / sd),
                       "direction": meta["high"] if a > b else meta["low"]})
    shifts.sort(key=lambda r: abs(r["change_std"]), reverse=True)
    return shifts[:TOP_SHIFTS]


def describe_inflections(strategy: pd.Series, market: pd.Series, rf: pd.Series, beta: float,
                         weights: pd.DataFrame, state: pd.DataFrame) -> dict:
    """Turning points of cumulative alpha within the given (already sliced) window.

    `weights` are the strategy's target weights for the whole backtest; the
    weight held on day t is the target set at the previous close.
    """
    alpha = ((strategy - rf) - beta * (market - rf)).fillna(0.0)
    held_net = weights.sum(axis=1).shift(1).reindex(strategy.index).fillna(0.0)
    timing = ((held_net - beta) * (market - rf)).fillna(0.0)
    x = alpha.cumsum().to_numpy()
    span = float(x.max() - x.min()) if len(x) else 0.0
    min_swing = max(MIN_SWING, SWING_FRACTION * span)
    turns = drop_short_legs(turning_points(x, min_swing), MIN_LEG_DAYS)

    # Keep the largest reversals, then restore chronological order.
    def swing(k: int) -> float:
        nxt = turns[k + 1][0] if k + 1 < len(turns) else len(x) - 1
        return abs(x[nxt] - x[turns[k][0]])
    keep = sorted(sorted(range(len(turns)), key=swing, reverse=True)[:MAX_TURNS])
    turns = [turns[k] for k in keep]

    pivots = [0] + [i for i, _ in turns] + [len(x) - 1]
    out = []
    for n, (i, kind) in enumerate(turns, 1):
        at = strategy.index[i]
        w_net = weights.sum(axis=1)
        pos = weights.index.get_loc(at)
        out.append({
            "n": n, "date": at.strftime("%Y-%m-%d"), "kind": kind,
            "leg_before": _leg(strategy, market, alpha, timing, pivots[n - 1], i),
            "leg_after": _leg(strategy, market, alpha, timing, i, pivots[n + 1]),
            "net_exposure_before": float(w_net.iloc[max(0, pos - TRADE_WINDOW): pos + 1].mean()),
            "net_exposure_after": float(w_net.iloc[pos + 1: pos + 1 + TRADE_WINDOW].mean())
                                  if pos + 1 < len(w_net) else float(w_net.iloc[pos]),
            "trades_near": _trades_near(weights, at),
            "market_shifts": _state_shifts(state, at),
        })
    return {"beta": float(beta), "window_alpha": float(alpha.sum()),
            "window_alpha_timing": float(timing.sum()), "min_swing": min_swing,
            "turning_points": out}
