"""Web app: serves the chart page and JSON endpoints backed by the Parquet cache.

Run:
    .venv/Scripts/python -m uvicorn app:app --reload
then open http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data_loader import get_ticker, load_prices

STATIC_DIR = Path(__file__).parent / "static"

# Display names for the selector. Order here is the order shown in the UI.
TICKER_NAMES = {
    "^GSPC": "S&P 500 (index)",
    "SPY": "SPY - S&P 500",
    "QQQ": "QQQ - Nasdaq 100",
    "DIA": "DIA - Dow Jones",
    "IWM": "IWM - Russell 2000",
    "TLT": "TLT - 20Y+ Treasuries",
    "IEF": "IEF - 7-10Y Treasuries",
    "GLD": "GLD - Gold",
    "^VIX": "VIX (index)",
    "^VIX3M": "VIX 3-month (index)",
    "XLK": "XLK - Technology",
    "XLF": "XLF - Financials",
    "XLE": "XLE - Energy",
    "XLV": "XLV - Health Care",
    "XLY": "XLY - Consumer Disc.",
    "XLP": "XLP - Consumer Staples",
    "XLI": "XLI - Industrials",
    "XLU": "XLU - Utilities",
    "XLB": "XLB - Materials",
}

app = FastAPI(title="HackRU Strategy Analyzer")


@app.get("/api/tickers")
def tickers():
    available = set(load_prices()["ticker"].unique())
    return [
        {"symbol": sym, "name": name}
        for sym, name in TICKER_NAMES.items()
        if sym in available
    ]


@app.get("/api/prices/{symbol}")
def prices(symbol: str, start: str | None = None, end: str | None = None):
    if symbol not in TICKER_NAMES:
        raise HTTPException(404, f"Unknown ticker {symbol!r}")
    df = get_ticker(symbol, start, end)
    return {
        "symbol": symbol,
        "name": TICKER_NAMES[symbol],
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "close": df["close"].round(4).tolist(),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
