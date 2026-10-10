"""Read the cached Parquet data. Shared by the backtester and the charts."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_prices() -> pd.DataFrame:
    """Long format: date, open, high, low, close, volume, ticker."""
    return pd.read_parquet(DATA_DIR / "prices.parquet")


@lru_cache(maxsize=1)
def load_macro() -> pd.DataFrame:
    """Long format: date, value, series."""
    return pd.read_parquet(DATA_DIR / "macro.parquet")


def get_ticker(ticker: str, start=None, end=None) -> pd.DataFrame:
    """OHLCV for one ticker, indexed by date."""
    df = load_prices()
    df = df[df["ticker"] == ticker].set_index("date").drop(columns="ticker")
    return df.loc[start:end]


def get_closes(tickers=None, start=None, end=None) -> pd.DataFrame:
    """Wide table of adjusted closes: one column per ticker."""
    df = load_prices()
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    return df.pivot(index="date", columns="ticker", values="close").loc[start:end]


def get_macro_daily(start=None, end=None) -> pd.DataFrame:
    """Macro series aligned to trading days (monthly series forward-filled)."""
    trading_days = pd.DatetimeIndex(sorted(load_prices()["date"].unique()))
    wide = load_macro().pivot(index="date", columns="series", values="value")
    wide = wide.reindex(wide.index.union(trading_days)).ffill().reindex(trading_days)
    return wide.loc[start:end]


def load_market_state(start=None, end=None) -> pd.DataFrame:
    """Daily market-state indicators (see market_state.py), indexed by date."""
    return pd.read_parquet(DATA_DIR / "market_state.parquet").loc[start:end]
