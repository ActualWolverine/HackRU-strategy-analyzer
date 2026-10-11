"""Benchmark: full SPY backtest throughput in bars/sec.

Times the same call the dashboard makes for the SMA-on-SPY strategy: strategy
signals, the engine, metrics, alpha regressions (full / before / after
publication), regime segmentation and the market-conditions report. Data is
loaded once before timing (a warm-up run), so this measures compute, not disk.

    python bench.py
"""

import ctypes
import os
import platform
import statistics
import sys
import time

from backtest import backtest
from strategies import STRATEGIES, universe_for

KEY = "sma_trend"
RUNS = 20


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    if sys.platform == "darwin":
        return os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or "unknown"


def ram_gb() -> float:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MemoryStatus()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def main() -> None:
    spec = STRATEGIES[KEY]
    params = {p["name"]: p["default"] for p in spec["params"]}
    args = (spec["fn"], universe_for(KEY, params), spec["benchmark"], params)
    kwargs = {"published": spec.get("published")}

    result = backtest(*args, **kwargs)  # warm-up: loads and caches the Parquet data
    bars = len(result["dates"])

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        backtest(*args, **kwargs)
        times.append(time.perf_counter() - start)
    median = statistics.median(times)

    print(f"Strategy:   {spec['name']} on SPY, {result['dates'][0]} to {result['dates'][-1]}")
    print(f"Bars:       {bars:,} daily bars")
    print(f"Time:       {median * 1000:.1f} ms per full backtest (median of {RUNS}, min {min(times) * 1000:.1f} ms)")
    print(f"Throughput: {bars / median:,.0f} bars/sec")
    print(f"CPU:        {cpu_name()} ({os.cpu_count()} logical cores)")
    print(f"RAM:        {ram_gb():.1f} GB")
    print(f"OS:         {platform.platform()}")
    print(f"Python:     {platform.python_version()} ({platform.architecture()[0]})")


if __name__ == "__main__":
    main()
