# Strategy Analyzer

Strategy Analyzer backtests trading strategies on 30 years of daily ETF data, measures their alpha (including before vs. after each strategy was published), and automatically splits history into periods where a strategy beat, matched or lagged its benchmark. For each of those regimes it reports what the market looked like (volatility, trend, drawdown, VIX term structure, credit spreads, Fed policy) compared with the rest of history, so you can see what conditions a strategy needs to work.

## How to run it

Developed and tested on Python 3.14.6. Python 3.11+ should work (pandas 3 requires it) but is untested. From a clean clone:

```bash
git clone <repo-url> HackRU-strategy-analyzer
cd HackRU-strategy-analyzer
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app
```

Then open http://127.0.0.1:8000.

- The repo includes a data snapshot in `data/` (prices through 2026-10-09), so no download is needed. To refresh it: `python download_data.py` (needs internet; results will then differ slightly from the numbers below).
- The dashboard loads its chart library from the jsDelivr CDN, so the browser needs internet access.
- Tests: `python -m pytest`. Benchmark: `python bench.py`.

**Two components.** The core workflow is a Python backend (FastAPI, `app.py`) running the engine, and a browser dashboard (`static/index.html`) that calls it over HTTP/JSON. Every backtest is computed on request by the backend; nothing is precomputed or exported. Endpoints: `/api/prices`, `/api/conditions`, `/api/strategies`, `/api/backtest`, `/api/leaderboard`.

## Architecture

**Engine → segmenter → features → report → dashboard.** The **engine** (`strategies.py`, `backtest.py`) runs a strategy, a function that turns a price table into daily target weights per ticker. It applies each day's weights to the *next* day's returns, credits idle cash with the daily Fed funds rate and charges it on leverage, and computes performance, CAPM alpha with Newey-West t-stats, and a before/after-publication split. The **segmenter** (`label_relative_performance` in `backtest.py`) labels every day outperform / match / underperform from the strategy's rolling 3-month return vs. the benchmark (terciles, merging runs shorter than 10 days), producing contiguous regimes. The **features** (`market_state.py`, from `download_data.py` / `data_loader.py`) are thirteen daily market-state indicators computed only from data available at each close: 20-day realized volatility, 60-day efficiency ratio, 60-day return autocorrelation, distance from the 200-day average, drawdown from the 52-week high, 12-1 month momentum, sector breadth, 14-day ADX, 60-day sector correlation, VIX/VIX3M, Baa−10Y credit spread, the 10y−2y yield curve, and the 3-month change in Fed funds (the five newer ones are defined below). The **report** (`regime.py`) compares each regime's days against all other days per indicator using a rank-biserial effect size, ranks the indicators by how unusual the regime was, and returns it as JSON with the equity curves and alpha tables. The **dashboard** (`static/index.html`, TradingView lightweight-charts) shows the price chart with drag-to-select region analysis, the strategy form, the equity curve with regime shading and an optional "relative to benchmark" view, a synced rolling-excess-return pane, alpha tables, and an alpha leaderboard across all 15 built-in strategies.

### Additional market-condition indicators

The condition summaries compare indicator values on a selected period (or on
outperforming/underperforming backtest days) with the rest of the available
history. These are descriptive indicators, not a forecast. Calculations use
daily data available by that day's close.

The additional price indicators use data already stored in `data/prices.parquet`:

| Indicator | Definition |
| --- | --- |
| 12-1 month momentum | `close[t-21] / close[t-252] - 1`; trailing 12-month price return excluding the latest 21 trading days. |
| Sector breadth | Fraction of the nine tracked sector ETFs whose close is above its own 200-day simple moving average; only ETFs with valid values count. |
| ADX (14 days) | Directional movement index based on 14-day Wilder-smoothed true range and positive/negative directional movement; ADX is a 14-day Wilder smoothing of `DX = 100 * abs(+DI - -DI) / (+DI + -DI)`. It measures trend strength, not direction. |
| Sector correlation (60 days) | Mean of the rolling 60-day Pearson correlations between daily close-to-close returns for every pair of the nine sector ETFs. |
| Treasury yield curve (10y-2y) | `DGS10 - DGS2`, in percentage points; also available as the FRED `T10Y2Y` series in `data/macro.parquet`. Negative values indicate an inverted curve. |

The price-based momentum and ADX use the S&P 500 index (`^GSPC`); breadth
and correlation use the nine sector ETFs. The Treasury yield curve uses the
existing FRED macro cache and is descriptive context, not a same-day trading
signal. Indicators are undefined until
their required history is available (252 days for momentum, 200 days for
breadth, and their rolling windows for ADX and correlation). Run
`download_data.py` to refresh the cached market-state Parquet file after
changing or downloading the source data.

## Result: 200-day SMA on SPY

Faber's trend filter (Faber, 2007): hold SPY while its price is above its 200-day simple moving average, otherwise hold cash. The rule is checked on the last trading day of each month. Oct 1995 to Oct 2026, 39 trades, invested 76% of the time.

| | SMA strategy | SPY buy & hold |
|---|---|---|
| CAGR | 10.6% | 10.6% |
| Volatility | 12.9% | 19.1% |
| Sharpe (vs. Fed funds) | 0.66 | 0.50 |
| Max drawdown | −24% | −55% |
| CAPM alpha vs. SPY | +4.2%/yr, t = 2.6, beta 0.46 | — |

**Honest reading:**

- **It didn't make more money than SPY; it made the same money with much less risk.** CAGR is identical. Volatility is a third lower and the worst drawdown is less than half SPY's.
- **The +4.2% alpha mostly measures that risk reduction, not extra return.** CAPM assumes a single beta, but a timing strategy's beta is 1.0 when invested and 0 when in cash. With a fixed full-sample beta of 0.46, only 20% of the alpha comes from the 2000–03 and 2007–09 bear markets it sat out. The rest is ordinary bull-market returns that CAPM counts as alpha because it expected only 0.46× the market.
- **It's not statistically significant at the bar that matters.** With 15 strategies tested, a t-stat of about 3 is the bar for a result picked from many (Harvey, Liu & Zhu, 2016). t = 2.6 is suggestive, not conclusive.
- **It's robust where overfitting would show.** Changing the average length barely moves alpha (150 days: +4.1%, 250 days: +4.2%, both t ≈ 2.6), and 20 bp transaction costs only take it to +3.9%. It held up after publication: +4.7%/yr before 2007, +4.0%/yr after. Neither half is significant on its own.
- **What the regime report shows:** it outperforms in sustained, high-volatility markets far below trend (slow bear markets) and underperforms in sharp drawdowns that recover quickly, where it sells near the bottom and rebuys higher: +15% higher in 1998, +11.5% in 2018–19, +5.4% in 2025 and +10.5% in 2026. Both regimes are drawdowns; what separates them is how fast the market recovers, which is the regime the strategy needs.
- **The sample is small.** 39 trades and only a handful of independent bear markets in 30 years.

## Invariants and tests

```bash
python -m pytest        # 235 tests, about 1 minute
```

| Invariant | How it's tested (`tests/`) |
|---|---|
| **No look-ahead.** A decision on day *t* depends only on data up to *t*. | `test_no_lookahead.py`: for every strategy and 3 cutoff dates, corrupts all prices after the cutoff, using either random noise or a mirrored path that reverses every future move, and requires identical weights through the cutoff. Also run end to end through the full backtest. Controls: the corruption must change later trades, so a pass isn't vacuous, and a deliberately cheating strategy that peeks one day ahead must be caught. |
| **Cash reconciliation.** The vectorized engine equals explicit bookkeeping. | `test_reconciliation.py`: an independent ledger buys and sells actual share counts, credits short-sale proceeds to cash, accrues and pays interest, and deducts costs. Its equity must match the engine's to 1e-9 for every strategy, with and without costs, and must match the equity curve the dashboard shows. Controls: an engine that forgets interest on cash must fail, and so must a cost formula that ignores price drift. |
| **Determinism.** Same inputs give bit-identical outputs. | `test_determinism.py`: exact SHA-256 of every full result across repeat runs, forward vs. reverse run order, and two separate processes with different `PYTHONHASHSEED`. The cached data must not be mutated. Control: a one-ulp change must change the hash. |

The engine is vectorized: each strategy sees the whole price table, so no-look-ahead is enforced by these tests rather than by construction. New strategies added to `STRATEGIES` are picked up by all three suites automatically.

## Performance

**27,447 bars/sec**: a full SPY backtest (7,786 daily bars) in 284 ms, median of 20 runs. Repeated runs on the same machine ranged from about 26,000 to 31,000 bars/sec.

That covers everything the dashboard request does: strategy signals, the engine, metrics, three alpha regressions, regime segmentation and the market-conditions report over all 13 indicators. Data is loaded from Parquet once before timing.

- CPU: AMD Ryzen AI 9 HX 370 (12 cores / 24 threads)
- RAM: 31.1 GB
- OS: Windows 11 Home, build 26200
- Python 3.14.6 (64-bit); pandas 3.0.6, numpy 2.5.3, statsmodels 0.15.0

To reproduce (prints the same fields for your machine):

```bash
python bench.py
```

## Limitations and next steps

- **Execution at the signal's close.** Trades happen at the same closing price the signal is computed from (in practice, a market-on-close order). Next: an optional one-day execution delay.
- **Costs are off in the dashboard.** The engine supports transaction costs (and the reconciliation tests cover them), but the UI doesn't expose them yet.
- **Data:**
  - Daily ETF data only, back-adjusted by Yahoo for dividends and splits.
  - The universe is ETFs that exist today (survivorship).
  - FRED only serves 3 years of high-yield spreads, so the Baa−10Y spread stands in.
  - VIX/VIX3M starts in 2006.
- **The regime labels and conditions report use the future on purpose.** They're descriptive: centred windows and full-sample comparisons. Nothing feeds back into a strategy. A regime-switching strategy built on them would need walk-forward fitting with only past data.
- **Parameters chosen in-sample.** The XLY/XLV pairs default was picked from a full-sample cointegration scan.
- **Known bug: pairs stat-arb is not market-neutral.** When its rolling hedge ratio turns negative, the "short" leg becomes long, so it can end up nearly 100% long one ETF. Fix: stay flat when the hedge ratio is ≤ 0.
- **Publication years are unverified.** The years used for the before/after split (`PUBLICATIONS` in `strategies.py`) are best-known first sources and should be checked.
- **CAPM alpha misreads timing strategies** (see the SMA result). Next: Treynor–Mazuy or Henriksson–Merton timing regressions, and multi-factor alpha (stocks + bonds).
- **Next: LLM-generated strategies.** Describe a strategy in English and have it generated as code, run in a sandboxed subprocess that receives one bar at a time, so look-ahead is impossible by construction.

## External dependencies and what was written this weekend

**Prebuilt, not written by us:**
- **Python libraries:** pandas, numpy, pyarrow (Parquet), statsmodels (OLS with Newey-West errors; Engle-Granger cointegration test), FastAPI + uvicorn (web server), requests, yfinance, pytest.
- **Data:** Yahoo Finance daily prices via yfinance; FRED public CSV endpoint (Treasury yields, Fed funds, CPI, unemployment, Baa−10Y spread, VIX).
- **Frontend:** TradingView lightweight-charts 4.2.0, loaded from jsDelivr.
- **Strategy ideas:** the 15 built-in strategies are reimplementations of published or well-known strategies, with sources listed in `PUBLICATIONS` in `strategies.py`.

**Written this weekend:** all code in this repository, developed with an AI coding assistant (Claude Code):
- data download and the market-state features;
- the backtest engine, cash and leverage accounting, and alpha / publication-split statistics;
- the regime segmenter and conditions report;
- all 15 strategy implementations;
- the FastAPI backend and the dashboard;
- the three test suites and the benchmark.
