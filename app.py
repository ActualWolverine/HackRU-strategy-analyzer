"""Web app: serves the chart page and JSON endpoints backed by the Parquet cache.

Run:
    .venv/Scripts/python -m uvicorn app:app --reload
then open http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backtest import backtest
from data_loader import get_ticker, load_prices
from regime import describe_period
from strategies import BENCHMARKS, STRATEGIES, universe_for

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


@app.get("/api/conditions")
def conditions(start: str, end: str):
    """What was different about the market between two dates."""
    if start > end:
        start, end = end, start
    return describe_period(start, end)


@app.get("/api/strategies")
def strategies():
    return {
        "benchmarks": BENCHMARKS,
        "strategies": [
            {"key": key, "name": s["name"], "description": s["description"],
             "benchmark": s["benchmark"], "params": s["params"]}
            for key, s in STRATEGIES.items()
        ],
    }


class BacktestRequest(BaseModel):
    strategy: str
    params: dict = {}
    benchmark: str | None = None
    start: str | None = None
    end: str | None = None


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Compute a strategy from scratch and describe the regimes it did well/badly in."""
    spec = STRATEGIES.get(req.strategy)
    if spec is None:
        raise HTTPException(404, f"Unknown strategy {req.strategy!r}")
    params = {p["name"]: p["default"] for p in spec["params"]}
    for p in spec["params"]:
        if p["name"] in req.params:
            value = req.params[p["name"]]
            params[p["name"]] = {"int": int, "float": float}.get(p["type"], str)(value)
    benchmark = req.benchmark or spec["benchmark"]
    if benchmark not in BENCHMARKS:
        raise HTTPException(400, f"Unknown benchmark {benchmark!r}")
    try:
        result = backtest(spec["fn"], universe_for(req.strategy, params), benchmark, params, req.start, req.end)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Backtest failed: {e}")
    return {"strategy": req.strategy, "name": spec["name"], "params": params, **result}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
