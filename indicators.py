"""Hand-written descriptions of the market-state indicators.

These are deterministic reference text, shown when a user clicks an indicator
and given to the AI explainer as the only allowed definitions. Keys match the
columns of data/market_state.parquet and `regime.INDICATORS`.
"""

from regime import INDICATORS

DESCRIPTIONS = {
    "realized_vol_20d": {
        "what": "How much the index has actually been moving day to day over the last 20 trading days, annualized. "
                "It is the standard deviation of daily log returns scaled by the square root of 252.",
        "read": "Around 10-15% is a calm market; 25%+ is stressed; 2008 and March 2020 exceeded 60%. "
                "Volatility clusters: high readings tend to be followed by high readings.",
        "matters": "Volatility-managed strategies size positions by it directly. Trend followers and option-like "
                   "strategies behave very differently in calm versus turbulent markets, so it is usually the first "
                   "indicator to check when a strategy's performance changes.",
        "caveat": "Backward-looking: it reacts to a shock only after the shock has happened.",
    },
    "efficiency_ratio_60d": {
        "what": "Kaufman's efficiency ratio over 60 days: the net price change divided by the sum of all daily "
                "absolute changes along the way. 1 means price moved in a straight line; 0 means it went nowhere "
                "despite moving a lot.",
        "read": "Typical readings are 0.1-0.2. Above 0.3 the market is trending cleanly; below 0.1 it is chopping.",
        "matters": "Trend-following strategies (moving averages, breakouts) need a high ratio: they lose to whipsaws "
                   "when it is low. Mean-reversion strategies prefer the opposite.",
        "caveat": "Says nothing about direction, only about how directly price got where it went.",
    },
    "autocorr_60d": {
        "what": "The correlation between today's return and yesterday's return, measured over the last 60 days.",
        "read": "Positive means moves tend to continue the next day (momentum); negative means they tend to reverse "
                "(mean reversion). For the S&P 500 it is usually slightly negative, around -0.05.",
        "matters": "Short-horizon mean-reversion strategies like RSI(2) and Bollinger reversion are bets on negative "
                   "autocorrelation. When it turns positive, as in a one-way crash or melt-up, they tend to fail.",
        "caveat": "Noisy at 60 days; treat large swings as suggestive rather than precise.",
    },
    "trend_200d": {
        "what": "How far the index is above or below its 200-day simple moving average, as a percentage.",
        "read": "Positive means price is above its long-run average (uptrend); negative means below. Readings "
                "beyond +-10% are stretched.",
        "matters": "This is the signal behind the SMA and crossover strategies. Many timing rules are, in effect, "
                   "'be invested when this is positive'.",
        "caveat": "A moving average lags price by roughly half its window, so it confirms a trend change about 100 "
                  "trading days late.",
    },
    "drawdown_52w": {
        "what": "How far the index is below its highest close of the past 52 weeks. Zero means it is at a new high.",
        "read": "Within -5% is normal; -10% to -20% is a correction; beyond -20% is a bear market.",
        "matters": "Defensive and trend-following strategies earn their keep in deep drawdowns. Buy-the-dip "
                   "strategies trigger on it directly, and are hurt when drawdowns keep deepening.",
        "caveat": "Does not distinguish a fast crash from a slow grind lower, which matters a lot for timing rules.",
    },
    "momentum_12_1": {
        "what": "The index's return over the past 12 months, skipping the most recent month "
                "(close 21 days ago divided by close 252 days ago, minus 1).",
        "read": "Positive means the past year was good. The latest month is skipped because very short-term "
                "returns tend to reverse, which would blur the signal.",
        "matters": "This is the classic academic momentum signal (Jegadeesh & Titman, 1993). Time-series and "
                   "cross-sectional momentum strategies are built on it.",
        "caveat": "Momentum crashes: after a sharp bottom, the signal stays negative while prices rebound, so "
                  "momentum strategies miss the recovery.",
    },
    "sector_breadth_200d": {
        "what": "The share of the nine S&P sector ETFs trading above their own 200-day moving averages.",
        "read": "Near 100% means a broad advance with every sector participating; near 0% means a broad decline. "
                "A high index with low breadth means a few sectors are carrying the market.",
        "matters": "Breadth measures how healthy a trend is. Sector-rotation and momentum strategies thrive when "
                   "leadership is broad and consistent, and struggle when it narrows or rotates abruptly.",
        "caveat": "Available from late 1998, when the sector ETFs launched.",
    },
    "adx_14d": {
        "what": "Wilder's Average Directional Index over 14 days: a 0-100 measure of how strongly the index is "
                "trending, regardless of direction. It compares upward and downward directional movement, "
                "normalized by the true range.",
        "read": "Below 20 means no meaningful trend (ranging); above 25 means a trend is in force; above 40 is a "
                "strong trend.",
        "matters": "Trend strategies want ADX rising above 25. Range-bound (mean-reversion) strategies prefer it low.",
        "caveat": "ADX rises during strong downtrends too; it never tells you which way.",
    },
    "sector_correlation_60d": {
        "what": "The average pairwise correlation of daily returns among the nine sector ETFs over the last 60 days.",
        "read": "Around 0.5-0.6 is normal. Above 0.8 everything is moving together, which happens in panics when "
                "investors sell indiscriminately.",
        "matters": "Pairs trading, sector rotation and cross-sectional momentum all depend on sectors behaving "
                   "differently from each other. When correlation spikes, those differences vanish and the "
                   "strategies have nothing to work with.",
        "caveat": "Available from late 1998. Correlation measured over 60 days responds slowly to a regime change.",
    },
    "vix_term_structure": {
        "what": "The VIX (30-day expected volatility) divided by the VIX3M (3-month expected volatility).",
        "read": "Normally below 1 (contango): investors expect near-term calm. Above 1 (backwardation) means the "
                "market is pricing more fear now than later, which happens in sell-offs.",
        "matters": "It is one of the few forward-looking indicators here. The VIX term-structure filter strategy "
                   "steps out of the market when it inverts. Short-volatility and carry-style strategies are the "
                   "ones that get hurt when it flips.",
        "caveat": "The VIX3M series starts in 2006, so this indicator is empty before then.",
    },
    "credit_spread": {
        "what": "The yield on Baa-rated (lowest investment-grade) corporate bonds minus the 10-year Treasury yield, "
                "in percentage points.",
        "read": "Around 2 points is normal. Above 3 means lenders are demanding extra compensation for default "
                "risk; it reached 5.5 in late 2008.",
        "matters": "A wide spread signals financial stress that often leads or accompanies equity bear markets. "
                   "Risk-parity and defensive strategies tend to do relatively well when it is wide; levered or "
                   "carry strategies do badly.",
        "caveat": "Stands in for the high-yield spread, which FRED only publishes for the last three years.",
    },
    "yield_curve_10y2y": {
        "what": "The 10-year Treasury yield minus the 2-year Treasury yield, in percentage points.",
        "read": "Positive (steep) is normal. Negative (inverted) means short rates exceed long rates; inversions "
                "have preceded every US recession since the 1970s, typically by 6 to 18 months.",
        "matters": "It describes the monetary backdrop. Strategies that hold bonds (risk parity, dual momentum's "
                   "safe asset) are sensitive to it, and a steepening after inversion often marks the start of "
                   "an easing cycle.",
        "caveat": "A slow-moving macro indicator; it is context for a regime, not a timing signal.",
    },
    "fed_funds_3m_change": {
        "what": "How much the effective Fed funds rate has changed over the last three months (about 63 trading "
                "days), in percentage points.",
        "read": "Positive means the Fed is tightening; negative means easing. Changes beyond +-0.75 in a quarter "
                "are aggressive, as in 2022 (tightening) or 2008 and 2020 (easing).",
        "matters": "Rapid easing usually happens during crises, so it coincides with the conditions where "
                   "defensive strategies win. Rapid tightening pressures long-duration assets and strategies "
                   "holding long bonds.",
        "caveat": "The Fed reacts to the economy with a lag, so this indicator usually confirms a regime rather "
                  "than predicting it.",
    },
}

assert set(DESCRIPTIONS) == set(INDICATORS), "every indicator needs a description"


def describe_indicators() -> list[dict]:
    """All indicators with their display metadata and reference text, in table order."""
    return [
        {
            "key": key,
            "label": meta["label"],
            "fmt": meta["fmt"],
            "high": meta["high"],
            "low": meta["low"],
            "formula": meta.get("formula", ""),
            **DESCRIPTIONS[key],
        }
        for key, meta in INDICATORS.items()
    ]
