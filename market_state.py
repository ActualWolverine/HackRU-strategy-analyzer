"""Daily market-state indicators used to classify regimes.

Every value on date t uses only data available at the close of t, so the
table can be joined to backtest results without look-ahead.
"""

import numpy as np
import pandas as pd

from data_loader import get_closes, get_macro_daily

TRADING_DAYS = 252


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized std of daily log returns."""
    returns = np.log(close).diff()
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def efficiency_ratio(close: pd.Series, window: int = 60) -> pd.Series:
    """Kaufman efficiency ratio: net move / total path length. 1 = straight line, 0 = pure chop."""
    net_move = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net_move / path


def return_autocorr(close: pd.Series, window: int = 60) -> pd.Series:
    """Lag-1 autocorrelation of daily returns. >0 trending, <0 mean-reverting."""
    returns = close.pct_change()
    return returns.rolling(window).corr(returns.shift(1))


def trend_200d(close: pd.Series) -> pd.Series:
    """Percent distance of price above (+) or below (-) its 200-day moving average."""
    return close / close.rolling(200).mean() - 1


def drawdown_52w(close: pd.Series) -> pd.Series:
    """Percent below the trailing 52-week high (0 = at the high)."""
    return close / close.rolling(TRADING_DAYS, min_periods=1).max() - 1


def compute_market_state(ticker: str = "^GSPC") -> pd.DataFrame:
    """One row per trading day with all market-state indicators.

    Price-based indicators are computed on `ticker` (^GSPC has the longest history).
    """
    closes = get_closes([ticker, "^VIX", "^VIX3M"])
    close = closes[ticker].dropna()
    macro = get_macro_daily().reindex(close.index)

    state = pd.DataFrame(index=close.index)
    state["realized_vol_20d"] = realized_vol(close)
    state["efficiency_ratio_60d"] = efficiency_ratio(close)
    state["autocorr_60d"] = return_autocorr(close)
    state["trend_200d"] = trend_200d(close)
    state["drawdown_52w"] = drawdown_52w(close)
    # ^VIX3M starts in 2006, so earlier rows are NaN.
    state["vix_term_structure"] = (closes["^VIX"] / closes["^VIX3M"]).reindex(close.index)
    # FRED's high-yield spread is limited to the last 3 years; Baa - 10Y is the
    # long-history credit-stress proxy.
    state["credit_spread"] = macro["BAA10Y"]
    # About 3 months of trading days.
    state["fed_funds_3m_change"] = macro["DFF"] - macro["DFF"].shift(63)
    return state
