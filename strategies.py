"""Built-in strategies.

Each strategy is a function `(closes, **params) -> weights` (see backtest.py)
plus an entry in STRATEGIES describing its universe, benchmark and tunable
parameters so the UI can render a form for it. Extra information worth showing
(cointegration test, half-life, ...) goes in `weights.attrs["diagnostics"]`.
"""

import numpy as np
import pandas as pd
from stats import engle_granger

TRADING_DAYS = 252
MAX_NET_EXPOSURE = 0.5  # pairs: largest |long - short| allowed, as a fraction of gross


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ou_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion, in days, from an AR(1) fit of the spread.

    Models the spread as Ornstein-Uhlenbeck: d s = -lambda (s - mu) dt + noise.
    Discretised, delta s_t = a + b s_{t-1}; lambda = -b and half-life = ln 2 / lambda.
    """
    s = spread.dropna()
    lagged, delta = s.shift(1).iloc[1:], s.diff().iloc[1:]
    b = np.polyfit(lagged, delta, 1)[0]
    return float(np.log(2) / -b) if b < 0 else float("inf")


def zscore_to_weights(z: pd.Series, beta: pd.Series, entry: float, exit: float) -> pd.DataFrame:
    """Turn a spread z-score into dollar-neutral long/short weights.

    Spread = y - beta * x. Short the spread (short y, long x) when z > entry,
    long it when z < -entry, exit when z crosses back through +-exit. Gross
    exposure is 1.
    """
    side = pd.Series(np.nan, index=z.index)
    side[z > entry] = -1.0
    side[z < -entry] = 1.0
    # Hold until the spread crosses the exit level.
    flat = (z.abs() <= exit) | ((z.shift(1) > 0) & (z < 0)) | ((z.shift(1) < 0) & (z > 0))
    side[flat & side.isna()] = 0.0
    side = side.ffill().fillna(0.0)
    # Weights are hedge-ratio proportional (w_x / w_y = beta), so net exposure
    # is side * (1 - beta) / (1 + beta). A hedge ratio far from 1, or negative,
    # turns the "pair" into a one-sided directional bet. Trade only while the
    # net exposure would stay within MAX_NET_EXPOSURE of gross, which for the
    # default 0.5 means beta in [1/3, 3]; otherwise stay flat.
    lo, hi = (1 - MAX_NET_EXPOSURE) / (1 + MAX_NET_EXPOSURE), (1 + MAX_NET_EXPOSURE) / (1 - MAX_NET_EXPOSURE)
    tradable = (beta >= lo) & (beta <= hi)
    side = side.where(tradable, 0.0)

    b = beta.where(tradable, 0.0)
    w_y = side / (1 + b)
    w_x = -side * b / (1 + b)
    return pd.DataFrame({"y": w_y, "x": w_x})


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def pairs_stat_arb(closes, y: str, x: str, lookback: int = 252, entry_z: float = 2.0, exit_z: float = 0.0):
    """Mean-revert the spread between two cointegrated ETFs.

    Hedge ratio from a rolling OLS of log(y) on log(x); z-score of the spread
    over the same window. Reports the Engle-Granger cointegration test and
    the OU half-life on the full sample as diagnostics.
    """
    ly, lx = np.log(closes[y]), np.log(closes[x])

    # Rolling OLS slope/intercept: beta = cov / var.
    cov = ly.rolling(lookback).cov(lx)
    var = lx.rolling(lookback).var()
    beta = cov / var
    alpha = ly.rolling(lookback).mean() - beta * lx.rolling(lookback).mean()
    spread = ly - beta * lx - alpha
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    w = zscore_to_weights(z, beta, entry_z, exit_z)
    weights = pd.DataFrame({y: w["y"], x: w["x"]}, index=closes.index)

    # Full-sample diagnostics (descriptive, not used for trading decisions).
    t_stat, p_value = engle_granger(ly, lx)
    full_beta = np.polyfit(lx, ly, 1)[0]
    full_spread = ly - full_beta * lx
    weights.attrs["diagnostics"] = {
        "Engle-Granger p-value": round(float(p_value), 4),
        "Cointegrated at 5%": bool(p_value < 0.05),
        "Full-sample hedge ratio": round(float(full_beta), 3),
        "OU half-life (days)": round(ou_half_life(full_spread), 1),
    }
    return weights


def kalman_pairs(closes, y: str, x: str, delta: float = 1e-5, z_window: int = 63, entry_z: float = 2.0, exit_z: float = 0.0):
    """Pairs trade with a Kalman-filtered, time-varying hedge ratio.

    State = [alpha, beta] following a random walk; observation log(y) = alpha
    + beta * log(x) + noise. The filter's innovation (observed minus predicted
    log(y)) is the spread; it is z-scored against its own trailing std. The
    observation noise is learned online from the innovations. `delta` sets how
    fast the hedge ratio is allowed to drift.
    """
    ly, lx = np.log(closes[y]).to_numpy(), np.log(closes[x]).to_numpy()
    n = len(ly)
    state = np.array([0.0, 1.0])            # [alpha, beta]
    P = np.eye(2)                           # state covariance
    Q = delta / (1 - delta) * np.eye(2)     # process noise (how fast beta may drift)
    R = 1e-3                                # observation noise, adapted below

    betas, innovations = np.empty(n), np.empty(n)
    for t in range(n):
        H = np.array([1.0, lx[t]])
        P = P + Q                           # predict
        e = ly[t] - H @ state
        S = H @ P @ H + R
        K = P @ H / S                       # update
        state = state + K * e
        P = P - np.outer(K, H) @ P
        R = 0.99 * R + 0.01 * e * e         # slow adaptation of observation noise
        betas[t], innovations[t] = state[1], e

    beta = pd.Series(betas, index=closes.index)
    spread = pd.Series(innovations, index=closes.index)
    z = spread / spread.rolling(z_window).std()
    z.iloc[:TRADING_DAYS] = 0.0             # burn-in while the filter settles

    w = zscore_to_weights(z, beta, entry_z, exit_z)
    weights = pd.DataFrame({y: w["y"], x: w["x"]}, index=closes.index)
    weights.attrs["diagnostics"] = {
        "Hedge ratio (latest)": round(float(beta.iloc[-1]), 3),
        "Hedge ratio range": f"{beta.iloc[TRADING_DAYS:].min():.2f} to {beta.iloc[TRADING_DAYS:].max():.2f}",
        "Days with |z| > entry": int((z.abs() > entry_z).sum()),
    }
    return weights


def vol_managed(closes, asset: str, target_vol: float = 0.15, vol_window: int = 21, max_leverage: float = 2.0):
    """Moreira & Muir volatility-managed exposure: weight = target variance / realized variance."""
    returns = closes[asset].pct_change()
    realized_var = returns.rolling(vol_window).var() * TRADING_DAYS
    w = (target_vol ** 2 / realized_var).clip(upper=max_leverage).fillna(0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {
        "Average exposure": round(float(w[w > 0].mean()), 2),
        "Days at max leverage": int((w >= max_leverage).sum()),
    }
    return weights


def cross_sectional_momentum(closes, lookback: int = 126, top_n: int = 2, vol_window: int = 20):
    """Hold the top-N assets by trailing return, sized by inverse volatility, rebalanced daily."""
    returns = closes.pct_change()
    momentum = closes / closes.shift(lookback) - 1
    vol = returns.rolling(vol_window).std()

    rank = momentum.rank(axis=1, ascending=False)
    selected = (rank <= top_n) & momentum.notna()
    inv_vol = (1 / vol).where(selected)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    weights.attrs["diagnostics"] = {
        "Average holdings": round(float(selected.sum(axis=1).mean()), 2),
    }
    return weights


def sma_trend(closes, asset: str, window: int = 200, monthly: int = 1):
    """Faber's trend filter: hold the asset while its price is above its N-day simple
    moving average, otherwise hold cash. With `monthly`, the rule is only checked on
    the last trading day of each month and held until the next, as in Faber (2007)."""
    c = closes[asset]
    sma = c.rolling(window).mean()
    signal = (c > sma).astype(float).where(sma.notna())
    if monthly:
        month = pd.Series(c.index.to_period("M"), index=c.index)
        month_end = month.ne(month.shift(-1))
        signal = signal.where(month_end).ffill()
    w = signal.fillna(0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{w.mean():.0%}", "Switches": int((w != w.shift()).sum() - 1)}
    return weights


def sma_crossover(closes, asset: str, fast: int = 50, slow: int = 200, allow_short: int = 0):
    """Trend following: long when the fast moving average is above the slow one."""
    c = closes[asset]
    signal = np.sign(c.rolling(fast).mean() - c.rolling(slow).mean())
    w = signal.where(signal > 0, -1.0 if allow_short else 0.0).where(signal.notna(), 0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{(w != 0).mean():.0%}", "Crossovers": int((w != w.shift()).sum())}
    return weights


def time_series_momentum(closes, lookback: int = 252, target_vol: float = 0.10, vol_window: int = 63):
    """Moskowitz-Ooi-Pedersen: in each asset, go long if its trailing return is positive,
    short if negative, sized so each position targets the same volatility."""
    returns = closes.pct_change()
    sign = np.sign(closes / closes.shift(lookback) - 1)
    vol = returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    weights = (sign * target_vol / vol / closes.shape[1]).fillna(0.0)
    weights.attrs["diagnostics"] = {"Average gross exposure": round(float(weights.abs().sum(axis=1).mean()), 2),
                                    "Share of long positions": f"{(sign > 0).mean().mean():.0%}"}
    return weights


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_mean_reversion(closes, asset: str, period: int = 2, entry: float = 10.0, exit_ma: int = 5, trend_ma: int = 200):
    """Connors RSI(2): buy oversold dips in an uptrend, sell when price closes back above a short MA."""
    c = closes[asset]
    r = rsi(c, period)
    uptrend = c > c.rolling(trend_ma).mean()
    side = pd.Series(np.nan, index=c.index)
    side[(r < entry) & uptrend] = 1.0
    side[c > c.rolling(exit_ma).mean()] = 0.0
    w = side.ffill().fillna(0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{(w > 0).mean():.0%}", "Trades": int(((w > 0) & (w.shift() == 0)).sum())}
    return weights


def bollinger_reversion(closes, asset: str, window: int = 20, k: float = 2.0, allow_short: int = 1):
    """Fade moves outside the Bollinger bands; exit when price crosses back through the middle band."""
    c = closes[asset]
    mid = c.rolling(window).mean()
    band = k * c.rolling(window).std()
    side = pd.Series(np.nan, index=c.index)
    side[c < mid - band] = 1.0
    if allow_short:
        side[c > mid + band] = -1.0
    crossed = np.sign(c - mid) != np.sign((c - mid).shift(1))
    side[crossed & side.isna()] = 0.0
    w = side.ffill().fillna(0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{(w != 0).mean():.0%}", "Trades": int(((w != 0) & (w.shift() == 0)).sum())}
    return weights


def vix_contango_filter(closes, asset: str = "SPY", max_ratio: float = 1.0):
    """Hold the asset only while the VIX curve is in contango (VIX / VIX3M below the threshold).

    An inverted curve means the market is pricing near-term panic; stepping
    aside then is a classic volatility-regime filter. Data starts 2006 (VIX3M).
    """
    ratio = closes["^VIX"] / closes["^VIX3M"]
    w = (ratio < max_ratio).astype(float)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{w.mean():.0%}", "Days inverted": int((w == 0).sum())}
    return weights


def donchian_breakout(closes, asset: str, entry_window: int = 252, exit_window: int = 126):
    """Turtle-style breakout: buy a new N-day high, sell on an M-day low."""
    c = closes[asset]
    high = c.rolling(entry_window).max().shift(1)
    low = c.rolling(exit_window).min().shift(1)
    side = pd.Series(np.nan, index=c.index)
    side[c > high] = 1.0
    side[c < low] = 0.0
    w = side.ffill().fillna(0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{(w > 0).mean():.0%}", "Breakouts": int(((w > 0) & (w.shift() == 0)).sum())}
    return weights


def buy_the_dip(closes, asset: str, entry_drawdown: float = 0.10, exit_drawdown: float = 0.02):
    """Buy when the asset is more than `entry_drawdown` below its 52-week high; sell once it
    has recovered to within `exit_drawdown` of the high. Works in V-shaped recoveries, bleeds in long bears."""
    c = closes[asset]
    dd = c / c.rolling(TRADING_DAYS, min_periods=1).max() - 1
    side = pd.Series(np.nan, index=c.index)
    side[dd < -entry_drawdown] = 1.0
    side[dd > -exit_drawdown] = 0.0
    w = side.ffill().fillna(0.0)
    weights = pd.DataFrame({asset: w}, index=closes.index)
    weights.attrs["diagnostics"] = {"Time in market": f"{(w > 0).mean():.0%}", "Dips bought": int(((w > 0) & (w.shift() == 0)).sum())}
    return weights


def risk_parity(closes, vol_window: int = 63):
    """Equal risk contribution (inverse-volatility) across the universe, rebalanced daily."""
    vol = closes.pct_change().rolling(vol_window).std()
    inv = 1 / vol
    weights = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
    weights.attrs["diagnostics"] = {f"Avg weight {t}": f"{weights[t].mean():.0%}" for t in closes.columns}
    return weights


def sector_rotation(closes, lookback: int = 126, top_n: int = 3):
    """Hold the top-N S&P sectors by trailing return, equal weight, rebalanced daily."""
    momentum = closes / closes.shift(lookback) - 1
    rank = momentum.rank(axis=1, ascending=False)
    selected = (rank <= top_n) & momentum.notna()
    weights = selected.div(selected.sum(axis=1), axis=0).fillna(0.0)
    held = selected.sum()
    weights.attrs["diagnostics"] = {"Most held": held.idxmax(), "Least held": held.idxmin()}
    return weights


def dual_momentum(closes, lookback: int = 252, risk_on: str = "SPY", alt: str = "EFA", safe: str = "IEF"):
    """Antonacci's dual momentum: pick the stronger of two equity markets (relative momentum);
    hold it only if its trailing return is positive (absolute momentum), otherwise hold bonds."""
    mom = closes / closes.shift(lookback) - 1
    pair = mom[[risk_on, alt]].dropna()          # idxmax rejects all-NaN warm-up rows
    best = pair.idxmax(axis=1).reindex(mom.index)
    best_mom = pair.max(axis=1).reindex(mom.index)
    weights = pd.DataFrame(0.0, index=closes.index, columns=[risk_on, alt, safe])
    for t in (risk_on, alt):
        weights[t] = ((best == t) & (best_mom > 0)).astype(float)
    weights[safe] = ((best_mom <= 0) & best_mom.notna()).astype(float)
    weights.attrs["diagnostics"] = {f"Time in {t}": f"{weights[t].mean():.0%}" for t in weights.columns}
    return weights


# ---------------------------------------------------------------------------
# Registry (what the UI shows)
# ---------------------------------------------------------------------------

SINGLE_ASSETS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "IEF", "GLD", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
PAIR_CHOICES = ["SPY", "IVV", "QQQ", "XLK", "EFA", "EEM", "DIA", "IWM", "TLT", "IEF", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "GLD"]
MOMENTUM_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]

BENCHMARKS = ["SPY", "CASH", "^GSPC", "QQQ", "IWM", "EFA", "TLT", "GLD"]

STRATEGIES = {
    "pairs_stat_arb": {
        "name": "Pairs stat-arb (Engle-Granger)",
        "description": "Rolling-OLS hedge ratio, trade the spread z-score at ±entry, exit at ±exit. Reports cointegration test and OU half-life.",
        "fn": pairs_stat_arb,
        "asset_params": ["y", "x"],
        "benchmark": "SPY",
        "params": [
            {"name": "y", "label": "Asset Y", "type": "choice", "choices": PAIR_CHOICES, "default": "XLY"},
            {"name": "x", "label": "Asset X", "type": "choice", "choices": PAIR_CHOICES, "default": "XLV"},
            {"name": "lookback", "label": "Lookback (days)", "type": "int", "default": 252, "min": 20, "max": 1000},
            {"name": "entry_z", "label": "Entry z", "type": "float", "default": 2.0, "min": 0.5, "max": 4},
            {"name": "exit_z", "label": "Exit z", "type": "float", "default": 0.0, "min": 0, "max": 2},
        ],
    },
    "kalman_pairs": {
        "name": "Kalman dynamic-hedge pairs",
        "description": "Same trade, but the hedge ratio is a Kalman-filtered state that drifts over time; spread = filter innovation.",
        "fn": kalman_pairs,
        "asset_params": ["y", "x"],
        "benchmark": "SPY",
        "params": [
            {"name": "y", "label": "Asset Y", "type": "choice", "choices": PAIR_CHOICES, "default": "XLY"},
            {"name": "x", "label": "Asset X", "type": "choice", "choices": PAIR_CHOICES, "default": "XLV"},
            {"name": "delta", "label": "Hedge-ratio drift (delta)", "type": "float", "default": 1e-5, "min": 1e-8, "max": 1e-2},
            {"name": "z_window", "label": "Z-score window (days)", "type": "int", "default": 63, "min": 10, "max": 252},
            {"name": "entry_z", "label": "Entry z", "type": "float", "default": 2.0, "min": 0.5, "max": 4},
            {"name": "exit_z", "label": "Exit z", "type": "float", "default": 0.0, "min": 0, "max": 2},
        ],
    },
    "vol_managed": {
        "name": "Volatility-managed exposure",
        "description": "Moreira & Muir: scale exposure to one asset by target variance / realized variance, capped at max leverage.",
        "fn": vol_managed,
        "asset_params": ["asset"],
        "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": MOMENTUM_UNIVERSE, "default": "SPY"},
            {"name": "target_vol", "label": "Target vol (annual)", "type": "float", "default": 0.15, "min": 0.02, "max": 0.5},
            {"name": "vol_window", "label": "Vol window (days)", "type": "int", "default": 21, "min": 5, "max": 252},
            {"name": "max_leverage", "label": "Max leverage", "type": "float", "default": 2.0, "min": 1, "max": 4},
        ],
    },
    "cross_sectional_momentum": {
        "name": "Cross-sectional momentum",
        "description": f"Rank {', '.join(MOMENTUM_UNIVERSE)} by trailing return, hold the top N sized by inverse vol, rebalance daily.",
        "fn": cross_sectional_momentum,
        "universe": MOMENTUM_UNIVERSE,
        "benchmark": "SPY",
        "params": [
            {"name": "lookback", "label": "Lookback (days)", "type": "int", "default": 126, "min": 20, "max": 504},
            {"name": "top_n", "label": "Holdings", "type": "int", "default": 2, "min": 1, "max": 6},
            {"name": "vol_window", "label": "Vol window (days)", "type": "int", "default": 20, "min": 5, "max": 126},
        ],
    },
    "sma_trend": {
        "name": "Simple moving average (price vs. SMA)",
        "description": "Faber's trend filter: hold the asset while price is above its N-day SMA, cash otherwise. Monthly checking (Faber's original) cuts whipsaws.",
        "fn": sma_trend, "asset_params": ["asset"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "window", "label": "SMA (days)", "type": "int", "default": 200, "min": 5, "max": 500},
            {"name": "monthly", "label": "Check month-end only (0/1)", "type": "int", "default": 1, "min": 0, "max": 1},
        ],
    },
    "sma_crossover": {
        "name": "Moving-average crossover",
        "description": "Trend following: long while the fast MA is above the slow MA (optionally short otherwise). The 50/200 'golden cross'.",
        "fn": sma_crossover, "asset_params": ["asset"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "fast", "label": "Fast MA (days)", "type": "int", "default": 50, "min": 2, "max": 200},
            {"name": "slow", "label": "Slow MA (days)", "type": "int", "default": 200, "min": 10, "max": 500},
            {"name": "allow_short", "label": "Short when below (0/1)", "type": "int", "default": 0, "min": 0, "max": 1},
        ],
    },
    "donchian_breakout": {
        "name": "52-week breakout (Donchian)",
        "description": "Turtle-style: buy a new N-day high, exit on an M-day low. Needs persistent trends; whipsaws in ranges.",
        "fn": donchian_breakout, "asset_params": ["asset"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "entry_window", "label": "Entry high (days)", "type": "int", "default": 252, "min": 10, "max": 504},
            {"name": "exit_window", "label": "Exit low (days)", "type": "int", "default": 126, "min": 5, "max": 504},
        ],
    },
    "time_series_momentum": {
        "name": "Time-series momentum",
        "description": f"Moskowitz-Ooi-Pedersen: long/short each of {', '.join(MOMENTUM_UNIVERSE)} by the sign of its trailing return, each position vol-targeted.",
        "fn": time_series_momentum, "universe": MOMENTUM_UNIVERSE, "benchmark": "SPY",
        "params": [
            {"name": "lookback", "label": "Lookback (days)", "type": "int", "default": 252, "min": 20, "max": 504},
            {"name": "target_vol", "label": "Vol per position", "type": "float", "default": 0.10, "min": 0.02, "max": 0.4},
            {"name": "vol_window", "label": "Vol window (days)", "type": "int", "default": 63, "min": 10, "max": 252},
        ],
    },
    "dual_momentum": {
        "name": "Dual momentum (GEM)",
        "description": "Antonacci: hold the stronger of SPY/EFA if its 12-month return is positive, otherwise hold IEF.",
        "fn": dual_momentum, "universe": ["SPY", "EFA", "IEF"], "benchmark": "SPY",
        "params": [
            {"name": "lookback", "label": "Lookback (days)", "type": "int", "default": 252, "min": 60, "max": 504},
        ],
    },
    "sector_rotation": {
        "name": "Sector rotation",
        "description": "Hold the top-N of the nine S&P sector ETFs by trailing return, equal weight, rebalanced daily.",
        "fn": sector_rotation, "universe": SECTORS, "benchmark": "SPY",
        "params": [
            {"name": "lookback", "label": "Lookback (days)", "type": "int", "default": 126, "min": 20, "max": 504},
            {"name": "top_n", "label": "Sectors held", "type": "int", "default": 3, "min": 1, "max": 8},
        ],
    },
    "rsi_mean_reversion": {
        "name": "RSI(2) mean reversion",
        "description": "Connors: buy when 2-day RSI is oversold and price is above its 200-day MA; sell when price closes above the 5-day MA.",
        "fn": rsi_mean_reversion, "asset_params": ["asset"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "period", "label": "RSI period", "type": "int", "default": 2, "min": 2, "max": 30},
            {"name": "entry", "label": "Entry RSI below", "type": "float", "default": 10, "min": 1, "max": 50},
            {"name": "exit_ma", "label": "Exit MA (days)", "type": "int", "default": 5, "min": 2, "max": 50},
            {"name": "trend_ma", "label": "Trend filter MA (days)", "type": "int", "default": 200, "min": 20, "max": 400},
        ],
    },
    "bollinger_reversion": {
        "name": "Bollinger band reversion",
        "description": "Fade closes outside the ±k-sigma bands, exit when price crosses back through the middle band.",
        "fn": bollinger_reversion, "asset_params": ["asset"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "window", "label": "Window (days)", "type": "int", "default": 20, "min": 5, "max": 200},
            {"name": "k", "label": "Band width (sigma)", "type": "float", "default": 2.0, "min": 0.5, "max": 4},
            {"name": "allow_short", "label": "Short upper band (0/1)", "type": "int", "default": 1, "min": 0, "max": 1},
        ],
    },
    "buy_the_dip": {
        "name": "Buy the dip",
        "description": "Buy when the asset is >X% below its 52-week high, sell once it recovers to within Y% of the high.",
        "fn": buy_the_dip, "asset_params": ["asset"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "entry_drawdown", "label": "Enter below high by", "type": "float", "default": 0.10, "min": 0.02, "max": 0.5},
            {"name": "exit_drawdown", "label": "Exit within", "type": "float", "default": 0.02, "min": 0, "max": 0.2},
        ],
    },
    "vix_contango_filter": {
        "name": "VIX term-structure filter",
        "description": "Hold the asset only while VIX / VIX3M is below the threshold (curve in contango); step aside when the curve inverts. From 2006.",
        "fn": vix_contango_filter, "asset_params": ["asset"], "extra_tickers": ["^VIX", "^VIX3M"], "benchmark": "SPY",
        "params": [
            {"name": "asset", "label": "Asset", "type": "choice", "choices": SINGLE_ASSETS, "default": "SPY"},
            {"name": "max_ratio", "label": "Max VIX/VIX3M", "type": "float", "default": 1.0, "min": 0.7, "max": 1.3},
        ],
    },
    "risk_parity": {
        "name": "Risk parity (SPY / TLT / GLD)",
        "description": "Inverse-volatility weights across stocks, long bonds and gold, rebalanced daily.",
        "fn": risk_parity, "universe": ["SPY", "TLT", "GLD"], "benchmark": "SPY",
        "params": [
            {"name": "vol_window", "label": "Vol window (days)", "type": "int", "default": 63, "min": 10, "max": 252},
        ],
    },
}


# When each strategy became public, for the before/after-publication alpha test.
# Year = the journal, book or SSRN publication named in the source. Checked
# against the publishers' records on 2026-10-11 for the strategies that carry
# findings: Faber (JWM Spring 2007; SSRN working paper 2006), Connors & Alvarez
# (2008), Moskowitz-Ooi-Pedersen (JFE 2012), Antonacci (SSRN, first posted
# April 2012), Moreira & Muir (JF 2017; NBER working paper April 2016). Where a
# working paper preceded the journal by a year, the journal year is used, so
# "before publication" may include up to a year of post-circulation data.
# None = folk strategy with no single source.
PUBLICATIONS = {
    "pairs_stat_arb":           (2006, "Gatev, Goetzmann & Rouwenhorst, Review of Financial Studies (2006)"),
    "kalman_pairs":             (2013, "Chan, Algorithmic Trading (2013)"),
    "vol_managed":              (2017, "Moreira & Muir, Journal of Finance (2017)"),
    "cross_sectional_momentum": (2013, "Asness, Moskowitz & Pedersen, Value and Momentum Everywhere, Journal of Finance (2013)"),
    "sma_trend":                (2007, "Faber, A Quantitative Approach to Tactical Asset Allocation, Journal of Wealth Management (2007)"),
    "sma_crossover":            (1992, "Brock, Lakonishok & LeBaron, Journal of Finance (1992)"),
    "donchian_breakout":        (1960, "Donchian, Financial Analysts Journal (1960)"),
    "time_series_momentum":     (2012, "Moskowitz, Ooi & Pedersen, Journal of Financial Economics (2012)"),
    "dual_momentum":            (2012, "Antonacci, Risk Premia Harvesting Through Dual Momentum, SSRN (2012)"),
    "sector_rotation":          (1999, "Moskowitz & Grinblatt, Do Industries Explain Momentum?, Journal of Finance (1999)"),
    "rsi_mean_reversion":       (2008, "Connors & Alvarez, Short Term Trading Strategies That Work (2008)"),
    "bollinger_reversion":      (2001, "Bollinger, Bollinger on Bollinger Bands (2001)"),
    "buy_the_dip":              (None, "Folk strategy, no single source"),
    "vix_contango_filter":      (2017, "Johnson, Risk Premia and the VIX Term Structure, JFQA (2017)"),
    "risk_parity":              (2005, "Qian, Risk Parity Portfolios, PanAgora (2005)"),
}
for _key, (_year, _source) in PUBLICATIONS.items():
    STRATEGIES[_key]["published"] = _year
    STRATEGIES[_key]["source"] = _source


def universe_for(key: str, params: dict) -> list[str]:
    """Tickers the strategy needs, given its parameters."""
    spec = STRATEGIES[key]
    tickers = list(spec.get("universe", [])) + [params[p] for p in spec.get("asset_params", [])]
    return tickers + list(spec.get("extra_tickers", []))
