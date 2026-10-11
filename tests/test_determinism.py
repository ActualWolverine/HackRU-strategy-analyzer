"""Determinism: the same inputs must always give bit-identical outputs.

Covers repeat runs in one process, run order (hidden state leaking between
strategies), mutation of the cached price data, and separate processes with
different string-hash seeds (which change set/dict iteration order).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import backtest as bt
import data_loader
from strategies import STRATEGIES, universe_for

ROOT = Path(__file__).resolve().parent.parent


def default_params(key: str) -> dict:
    return {p["name"]: p["default"] for p in STRATEGIES[key]["params"]}


def run_strategy(key: str) -> dict:
    spec = STRATEGIES[key]
    params = default_params(key)
    return bt.backtest(spec["fn"], universe_for(key, params), spec["benchmark"], params,
                       published=spec.get("published"))


def fingerprint(result: dict) -> str:
    """Exact digest of a full backtest result (equity, alpha, regimes, conditions)."""
    # repr() of floats is exact (shortest round-trip), so any bit difference shows up.
    blob = json.dumps(result, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()


def fingerprint_all(order=None) -> dict[str, str]:
    return {key: fingerprint(run_strategy(key)) for key in (order or list(STRATEGIES))}


def frame_digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_repeat_runs_identical(key):
    assert fingerprint(run_strategy(key)) == fingerprint(run_strategy(key))


def test_run_order_does_not_matter():
    keys = list(STRATEGIES)
    forward = fingerprint_all(keys)
    backward = fingerprint_all(list(reversed(keys)))
    assert forward == backward


def test_cached_data_not_mutated():
    """Strategies get frames derived from lru_cached data; none may modify the cache in place."""
    before = (frame_digest(data_loader.load_prices()), frame_digest(data_loader.load_macro()))
    fingerprint_all()
    after = (frame_digest(data_loader.load_prices()), frame_digest(data_loader.load_macro()))
    assert before == after


def test_identical_across_processes_and_hash_seeds():
    code = (
        "import json, sys; sys.path[:0] = [r'%s', r'%s'];"
        "from test_determinism import fingerprint_all; print(json.dumps(fingerprint_all()))"
    ) % (ROOT, ROOT / "tests")
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]
    # And they agree with this process too.
    assert outputs[0] == fingerprint_all()


def test_fingerprint_detects_changes():
    """Control: a one-ulp change anywhere in the result must change the fingerprint."""
    result = run_strategy("sma_trend")
    tweaked = json.loads(json.dumps(result))
    tweaked["strategy_equity"][-1] = float.__add__(tweaked["strategy_equity"][-1], 1e-12)
    assert fingerprint(result) != fingerprint(tweaked)
