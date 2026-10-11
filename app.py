"""Web app: serves the chart page and JSON endpoints backed by the Parquet cache.

Run:
    .venv/Scripts/python -m uvicorn app:app --reload
then open http://127.0.0.1:8000
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backtest import SIGNIFICANT_T, backtest, period_stats
from data_loader import get_ticker, load_prices
from explain import PERIOD_TASK, STRATEGY_TASK, ExplanationService, period_facts, strategy_facts
from indicators import describe_indicators
from regime import describe_period
from strategies import BENCHMARKS, STRATEGIES, universe_for

STATIC_DIR = Path(__file__).parent / "static"

# Secrets live in .env (git-ignored); see .env.example.
load_dotenv(Path(__file__).parent / ".env")
explanations = ExplanationService.from_env()

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
             "benchmark": s["benchmark"], "params": s["params"],
             "published": s.get("published"), "source": s.get("source")}
            for key, s in STRATEGIES.items()
        ],
    }


def default_params(spec: dict) -> dict:
    return {p["name"]: p["default"] for p in spec["params"]}


class BacktestRequest(BaseModel):
    strategy: str
    params: dict = {}
    benchmark: str | None = None
    start: str | None = None
    end: str | None = None


def resolve(req: "BacktestRequest") -> tuple[dict, dict, str]:
    """Validate a request into (strategy spec, typed params, benchmark)."""
    spec = STRATEGIES.get(req.strategy)
    if spec is None:
        raise HTTPException(404, f"Unknown strategy {req.strategy!r}")
    params = default_params(spec)
    for p in spec["params"]:
        if p["name"] in req.params:
            value = req.params[p["name"]]
            params[p["name"]] = {"int": int, "float": float}.get(p["type"], str)(value)
    benchmark = req.benchmark or spec["benchmark"]
    if benchmark not in BENCHMARKS:
        raise HTTPException(400, f"Unknown benchmark {benchmark!r}")
    return spec, params, benchmark


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Compute a strategy from scratch and describe the regimes it did well/badly in."""
    spec, params, benchmark = resolve(req)
    try:
        result = backtest(spec["fn"], universe_for(req.strategy, params), benchmark, params, req.start, req.end,
                          published=spec.get("published"))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Backtest failed: {e}")
    return {"strategy": req.strategy, "name": spec["name"], "params": params,
            "source": spec.get("source"), **result}


# ---------------------------------------------------------------------------
# Explanations (AI-assisted, bounded to engine facts; see explain.py)
# ---------------------------------------------------------------------------

@app.get("/api/indicators")
def indicators():
    """Reference descriptions of every market-state indicator (deterministic, hand-written)."""
    return describe_indicators()


@app.get("/api/explain/status")
def explain_status():
    return {"provider": explanations.provider,
            "model": getattr(explanations.explainer, "model", None)}


@app.post("/api/explain/strategy")
def explain_strategy(req: BacktestRequest):
    """Why this strategy works in some conditions and not others. Engine facts + narrated explanation."""
    spec, params, benchmark = resolve(req)
    t = time.perf_counter()
    try:
        result = backtest(spec["fn"], universe_for(req.strategy, params), benchmark, params,
                          published=spec.get("published"))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Backtest failed: {e}")
    facts = strategy_facts(result, spec, params)
    engine_ms = (time.perf_counter() - t) * 1000
    return {"kind": "strategy", "engine_latency_ms": round(engine_ms, 1), **explanations.explain(STRATEGY_TASK, facts)}


class PeriodRequest(BacktestRequest):
    start: str
    end: str


@app.post("/api/explain/period")
def explain_period(req: PeriodRequest):
    """How the strategy behaved in a selected window, and why, for a learner."""
    spec, params, benchmark = resolve(req)
    start, end = sorted([req.start, req.end])
    t = time.perf_counter()
    try:
        stats = period_stats(spec["fn"], universe_for(req.strategy, params), benchmark, params, start, end)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Period analysis failed: {e}")
    conditions = describe_period(stats["start"], stats["end"])["indicators"]
    facts = period_facts(stats, spec, params, conditions)
    engine_ms = (time.perf_counter() - t) * 1000
    return {"kind": "period", "stats": stats, "conditions": conditions,
            "engine_latency_ms": round(engine_ms, 1), **explanations.explain(PERIOD_TASK, facts)}


@app.get("/api/leaderboard")
def leaderboard():
    """Alpha (full sample, before and after publication) for every strategy at default settings."""
    rows = []
    for key, spec in STRATEGIES.items():
        params = default_params(spec)
        r = backtest(spec["fn"], universe_for(key, params), spec["benchmark"], params,
                     published=spec.get("published"), describe=False)
        rows.append({"strategy": key, "name": spec["name"], "source": spec.get("source"),
                     "market": r["market"], "alpha": r["alpha"]})
    rows.sort(key=lambda row: -(row["alpha"]["full"] or {}).get("t_stat", float("-inf")))
    return {"significant_t": SIGNIFICANT_T, "rows": rows}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
