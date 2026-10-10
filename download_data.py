"""Download historical market + macro data into data/ as Parquet.

Run once (or whenever you want fresher data):
    .venv/Scripts/python download_data.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

START = "1995-01-01"
DATA_DIR = Path(__file__).parent / "data"

TICKERS = [
    # Broad market
    "^GSPC", "SPY", "IVV", "QQQ", "IWM", "DIA",
    # International (EFA 2001, EEM 2003)
    "EFA", "EEM",
    # Rates / safety
    "TLT", "IEF", "GLD",
    # Volatility (VIX3M starts 2006; used for VIX term structure)
    "^VIX", "^VIX3M",
    # Sectors (most start Dec 1998)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
]

# FRED series, fetched from the public CSV endpoint (no API key needed).
FRED_SERIES = {
    "DGS10": "10-Year Treasury yield",
    "DGS2": "2-Year Treasury yield",
    "T10Y2Y": "10Y minus 2Y spread",
    "FEDFUNDS": "Effective Fed funds rate (monthly)",
    "DFF": "Effective Fed funds rate (daily)",
    "CPIAUCSL": "CPI (all urban consumers)",
    "UNRATE": "Unemployment rate",
    "BAA10Y": "Baa corporate minus 10Y Treasury (credit spread)",
    "VIXCLS": "VIX close",
}


def download_prices() -> pd.DataFrame:
    frames = []
    for ticker in TICKERS:
        df = yf.Ticker(ticker).history(start=START, auto_adjust=True)
        if df.empty:
            print(f"  ! {ticker}: no data", file=sys.stderr)
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = df.index.tz_localize(None).normalize()
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        frames.append(df.reset_index())
        print(f"  {ticker:6s} {df.index[0].date()} -> {df.index[-1].date()}  ({len(df):,} rows)")
    return pd.concat(frames, ignore_index=True)


def download_fred() -> pd.DataFrame:
    frames = []
    for series in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={START}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), na_values=".")
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna()
        df["series"] = series
        frames.append(df)
        print(f"  {series:13s} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}  ({len(df):,} rows)")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading prices from Yahoo Finance...")
    prices = download_prices()
    prices.to_parquet(DATA_DIR / "prices.parquet", index=False)

    print("Downloading macro series from FRED...")
    macro = download_fred()
    macro.to_parquet(DATA_DIR / "macro.parquet", index=False)

    print(f"Saved {len(prices):,} price rows and {len(macro):,} macro rows to {DATA_DIR}")

    # Imported here so it reads the files just written.
    from market_state import compute_market_state

    state = compute_market_state()
    state.to_parquet(DATA_DIR / "market_state.parquet")
    print(f"Saved market state ({len(state):,} days) to {DATA_DIR / 'market_state.parquet'}")


if __name__ == "__main__":
    main()
