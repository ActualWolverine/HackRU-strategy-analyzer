"""Look-ahead tests: corrupting the future must not change any past decision.

For each strategy and cutoff date, every price after the cutoff is corrupted.
A strategy that only uses information available at each day's close must
produce exactly the same weights on every day up to and including the cutoff.

Two corruption modes:
  noise  - multiply future prices by heavy random lognormal noise.
  mirror - reflect the future price path, so every future daily move reverses
           direction. Random noise can leave the sign of the next day's move
           unchanged by chance, which would hide a strategy that peeks only one
           day ahead; mirroring guarantees that move flips.

Two control tests make sure the check has teeth: the corruption has to change
the strategy's later trades (so a pass isn't vacuous), and a strategy that
deliberately peeks one day ahead has to be caught.
"""

import numpy as np
import pandas as pd
import pytest

import backtest as bt
from data_loader import get_closes
from strategies import STRATEGIES, universe_for

# Chosen so every ticker in every universe has data on both sides, and to
# include stressed markets where signals flip a lot.
CUTOFFS = ["2008-10-15", "2016-06-30", "2020-03-20"]
# (mode, seed) pairs; the seed only matters for "noise".
CORRUPTIONS = [("noise", 0), ("noise", 1), ("mirror", 0)]
NOISE_SIGMA = 0.3  # daily log-noise; turns the future into a wild random path


def default_params(key: str) -> dict:
    return {p["name"]: p["default"] for p in STRATEGIES[key]["params"]}


def clean_closes(key: str) -> tuple[pd.DataFrame, list[str]]:
    tickers = universe_for(key, default_params(key))
    return get_closes(sorted(set(tickers))).dropna(), tickers


def corrupt_after(closes: pd.DataFrame, cutoff: str, mode: str = "noise", seed: int = 0) -> pd.DataFrame:
    """Corrupt every price strictly after `cutoff` (see module docstring for modes).

    Dates are kept, so the trading calendar is unchanged; only values are corrupted.
    """
    future = closes.index > pd.Timestamp(cutoff)
    out = closes.copy()
    if mode == "noise":
        rng = np.random.default_rng(seed)
        noise = np.exp(rng.normal(0.0, NOISE_SIGMA, size=(future.sum(), closes.shape[1])))
        out.loc[future] = closes.loc[future].to_numpy() * noise
    elif mode == "mirror":
        # p'_t = p_c^2 / p_t: the path reflected about the cutoff price, in log space.
        at_cutoff = closes.loc[~future].iloc[-1]
        out.loc[future] = (at_cutoff ** 2 / closes.loc[future]).to_numpy()
    else:
        raise ValueError(mode)
    return out


def weights_for(fn, closes: pd.DataFrame, tickers: list[str], params: dict) -> pd.DataFrame:
    return fn(closes[tickers], **params).reindex(closes.index).fillna(0.0)


def assert_same_through(a: pd.DataFrame, b: pd.DataFrame, cutoff: str):
    upto = a.index <= pd.Timestamp(cutoff)
    pd.testing.assert_frame_equal(a.loc[upto], b.loc[upto], check_exact=False, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode,seed", CORRUPTIONS)
@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("key", list(STRATEGIES))
def test_strategy_ignores_corrupted_future(key, cutoff, mode, seed):
    closes, tickers = clean_closes(key)
    if closes.index[0] >= pd.Timestamp(cutoff):
        pytest.skip(f"{key} has no data before {cutoff}")
    params = default_params(key)
    fn = STRATEGIES[key]["fn"]

    clean = weights_for(fn, closes, tickers, params)
    corrupted = weights_for(fn, corrupt_after(closes, cutoff, mode, seed), tickers, params)

    assert_same_through(clean, corrupted, cutoff)

    # Control: the strategy really does read the corrupted prices, so the
    # equality above is a real constraint rather than a strategy that ignores data.
    after = clean.index > pd.Timestamp(cutoff)
    assert not np.allclose(clean.loc[after], corrupted.loc[after]), (
        f"{key}: corrupting the future changed nothing at all, so this test proves nothing for it"
    )


# Corrupted futures can produce daily strategy losses beyond -100% (e.g. a short
# into a noise spike); the regime labelling takes log1p of those. That happens
# only after the cutoff, which this test doesn't compare.
@pytest.mark.filterwarnings("ignore:invalid value encountered in log1p")
@pytest.mark.parametrize("mode,seed", [CORRUPTIONS[0], CORRUPTIONS[-1]])
@pytest.mark.parametrize("key", list(STRATEGIES))
def test_backtest_equity_ignores_corrupted_future(key, mode, seed, monkeypatch):
    """End to end through the engine: returns, cash interest and trimming included."""
    cutoff = CUTOFFS[1]
    spec = STRATEGIES[key]
    params = default_params(key)
    args = (spec["fn"], universe_for(key, params), spec["benchmark"], params)

    clean = bt.backtest(*args)

    real_get_closes = bt.get_closes
    monkeypatch.setattr(bt, "get_closes", lambda *a, **k: corrupt_after(real_get_closes(*a, **k), cutoff, mode, seed))
    corrupted = bt.backtest(*args)

    n = sum(d <= cutoff for d in clean["dates"])
    assert n > 0
    assert clean["dates"][:n] == corrupted["dates"][:n]
    assert clean["strategy_equity"][:n] == corrupted["strategy_equity"][:n]
    assert clean["benchmark_equity"][:n] == corrupted["benchmark_equity"][:n]


def peeking_strategy(closes, asset: str = "SPY"):
    """Cheats by one day: holds SPY only on days before it goes up."""
    tomorrow_up = closes[asset].shift(-1) > closes[asset]
    return pd.DataFrame({asset: tomorrow_up.astype(float)}, index=closes.index)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_detector_catches_one_day_peek(cutoff):
    """Mirroring flips the day-after-cutoff move, so a one-day peek must always be caught."""
    closes = get_closes(["SPY"]).dropna()
    clean = weights_for(peeking_strategy, closes, ["SPY"], {})
    corrupted = weights_for(peeking_strategy, corrupt_after(closes, cutoff, "mirror"), ["SPY"], {})
    with pytest.raises(AssertionError):
        assert_same_through(clean, corrupted, cutoff)
