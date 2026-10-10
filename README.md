# HackRU-strategy-analyzer
This app uses backtesting to find which periods of time a trading strategy works best in, and calculates what the market conditions need to be for the strategy to be successful. Then it explains why the market needs to be in that state for the strategy to work.

## Daily market-condition indicators

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
