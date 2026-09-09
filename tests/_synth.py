"""Shared synthetic-candle generators for the offline tests.

Carved out of the original test scripts so the GBM-path setup lives in one
place. Each function is deterministic given its seed.
"""
import math
import random
import time


def gbm_flat(seed: int, n_bars: int, sigma: float = 0.0006,
             start: float = 65000.0) -> list[dict]:
    """OHLC all equal to the close — the shape test_edge/test_prep used."""
    random.seed(seed)
    t0 = int(time.time() // 60 * 60) - n_bars * 60
    px = start
    out = []
    for i in range(n_bars):
        px *= math.exp(random.gauss(-0.5 * sigma ** 2, sigma))
        out.append({"ts": t0 + i * 60, "open": px, "high": px, "low": px,
                    "close": px, "volume": 1.0})
    return out


def gbm_ohlc(seed: int, n_bars: int, sigma: float = 0.0006,
             start: float = 65000.0) -> list[dict]:
    """Real-ish OHLC bars — the shape test_live used for avg_settle."""
    random.seed(seed)
    t0 = int(time.time() // 60 * 60) - n_bars * 60
    px = start
    out = []
    for i in range(n_bars):
        o = px
        px *= math.exp(random.gauss(-0.5 * sigma ** 2, sigma))
        out.append({"ts": t0 + i * 60, "open": o, "high": max(o, px),
                    "low": min(o, px), "close": px, "volume": 1.0})
    return out
