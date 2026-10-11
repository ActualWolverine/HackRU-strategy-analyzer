"""Daily market-state indicators used to classify regimes.

Every value on date t uses only data available at the close of t, so the
table can be joined to backtest results without look-ahead.
"""

import numpy as np
import pandas as pd

from data_loader import get_closes, get_macro_daily, get_ticker

TRADING_DAYS = 252
SECTOR_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]


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


def momentum_12_1(close: pd.Series) -> pd.Series:
    """12-month price return excluding the most recent month."""
    return close.shift(21) / close.shift(252) - 1


def sector_breadth(closes: pd.DataFrame, window: int = 200) -> pd.Series:
    """Share of sectors with valid data whose close is above its rolling average."""
    average = closes.rolling(window).mean()
    valid = closes.notna() & average.notna()
    above_average = (closes > average) & valid
    return above_average.sum(axis=1).div(valid.sum(axis=1).replace(0, np.nan))


def average_directional_index(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder-smoothed ADX from daily high, low, and close."""
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    def wilder_smooth(series: pd.Series) -> pd.Series:
        return series.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    smoothed_range = wilder_smooth(true_range)
    plus_di = 100 * wilder_smooth(plus_dm) / smoothed_range
    minus_di = 100 * wilder_smooth(minus_dm) / smoothed_range
    directional_index = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder_smooth(directional_index)


def sector_correlation(closes: pd.DataFrame, window: int = 60) -> pd.Series:
    """Mean rolling pairwise correlation of daily sector returns."""
    returns = closes.pct_change()
    correlations = [
        returns[left].rolling(window).corr(returns[right])
        for i, left in enumerate(returns.columns)
        for right in returns.columns[i + 1 :]
    ]
    if not correlations:
        return pd.Series(np.nan, index=closes.index)
    return pd.concat(correlations, axis=1).mean(axis=1)


def compute_market_state(ticker: str = "^GSPC") -> pd.DataFrame:
    """One row per trading day with all market-state indicators.

    Price-based indicators are computed on `ticker` (^GSPC has the longest history).
    """
    closes = get_closes([ticker, "^VIX", "^VIX3M"])
    close = closes[ticker].dropna()
    macro = get_macro_daily().reindex(close.index)
    sector_closes = get_closes(SECTOR_TICKERS).reindex(close.index)
    ohlc = get_ticker(ticker).reindex(close.index)

    state = pd.DataFrame(index=close.index)
    state["realized_vol_20d"] = realized_vol(close)
    state["efficiency_ratio_60d"] = efficiency_ratio(close)
    state["autocorr_60d"] = return_autocorr(close)
    state["trend_200d"] = trend_200d(close)
    state["drawdown_52w"] = drawdown_52w(close)
    state["momentum_12_1"] = momentum_12_1(close)
    state["sector_breadth_200d"] = sector_breadth(sector_closes)
    state["adx_14d"] = average_directional_index(ohlc)
    state["sector_correlation_60d"] = sector_correlation(sector_closes)
    # ^VIX3M starts in 2006, so earlier rows are NaN.
    state["vix_term_structure"] = (closes["^VIX"] / closes["^VIX3M"]).reindex(close.index)
    # FRED's high-yield spread is limited to the last 3 years; Baa - 10Y is the
    # long-history credit-stress proxy.
    state["credit_spread"] = macro["BAA10Y"]
    state["yield_curve_10y2y"] = macro["T10Y2Y"]
    # About 3 months of trading days.
    state["fed_funds_3m_change"] = macro["DFF"] - macro["DFF"].shift(63)
    return state
