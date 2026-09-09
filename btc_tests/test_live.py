"""Checks for the live-Kalshi work: TWAP tau, dollar parsing, settlement, capture."""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/hasaninan/Downloads")
import btc_prediction_edge as E

TMP = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp")

# ---- 1. effective_tau: closed form + Monte-Carlo agreement ----
# Var((1/d) * INT_{T-d}^{T} W_s ds - W_t) should equal tau - 2d/3 for tau >= d.
assert abs(E.effective_tau(15.0) - (15.0 - 2.0 / 3.0)) < 1e-12
assert abs(E.effective_tau(1.0) - (1.0 - 2.0 / 3.0)) < 1e-12
assert E.effective_tau(0.0) == 0.0
assert E.effective_tau(-3.0) == 0.0
assert abs(E.effective_tau(5.0, avg_minutes=0.0) - 5.0) < 1e-12   # escape hatch
# inside the averaging window: tau^3/(3 d^2), continuous with the other branch
assert abs(E.effective_tau(0.5) - 0.125 / 3.0) < 1e-12
assert E.effective_tau(0.999) < E.effective_tau(1.001)
print("[ok] effective_tau closed form, continuity, and edge cases")

# Monte-Carlo the actual variance of a last-minute average vs the formula.
random.seed(5)
STEPS_PER_MIN = 60
TAU, D = 5.0, 1.0
n_paths = 40000
vals = []
for _ in range(n_paths):
    w = 0.0
    acc, cnt = 0.0, 0
    total_steps = int(TAU * STEPS_PER_MIN)
    start_avg = int((TAU - D) * STEPS_PER_MIN)
    dt = 1.0 / STEPS_PER_MIN
    for i in range(total_steps):
        w += random.gauss(0.0, math.sqrt(dt))
        if i >= start_avg:
            acc += w
            cnt += 1
    vals.append(acc / cnt)
mc_var = sum(v * v for v in vals) / len(vals)
want = E.effective_tau(TAU)
print(f"    MC var={mc_var:.4f}  formula={want:.4f}")
assert abs(mc_var - want) / want < 0.04, (mc_var, want)
print("[ok] Monte-Carlo confirms the averaged-settlement variance haircut")

# The correction must actually pull probabilities away from 0.5 late in a window.
p_naive = E.prob_finish_above(65100, 65000, 2.0, 0.0006, avg_minutes=0.0)
p_twap = E.prob_finish_above(65100, 65000, 2.0, 0.0006)
print(f"    tau=2m  naive {p_naive:.4f}  twap {p_twap:.4f}")
assert p_twap > p_naive, "averaging cuts variance -> more confident when ITM"
print("[ok] TWAP correction sharpens late-window probabilities")

# ---- 2. dollar-string parsing, with legacy-cent fallback ----
m_new = {"yes_ask_dollars": "0.2800", "no_ask_dollars": "0.7300",
         "yes_ask": None, "no_ask": None}
assert abs(E._market_cents(m_new, "yes_ask") - 28.0) < 1e-9
assert abs(E._market_cents(m_new, "no_ask") - 73.0) < 1e-9
assert E._market_cents(m_new, "yes_bid") is None          # absent -> None
m_old = {"yes_ask": 31}                                    # legacy shape
assert abs(E._market_cents(m_old, "yes_ask") - 31.0) < 1e-9
assert E._market_cents({"yes_ask_dollars": "junk"}, "yes_ask") is None
print("[ok] _market_cents: dollars, legacy cents, junk, missing")

t = E._iso_to_ts("2026-08-12T13:45:00Z")
from datetime import datetime, timezone
assert t == datetime(2026, 8, 12, 13, 45, tzinfo=timezone.utc).timestamp(), t
assert E._iso_to_ts(None) is None and E._iso_to_ts("nope") is None
print("[ok] _iso_to_ts parses Z-suffixed ISO and rejects junk")

# ---- 3. KalshiMarket.vig_cents ----
km = E.KalshiMarket(ticker="T", strike=1.0, close_ts=0.0,
                    quote=E.Quote(up_cost_cents=28.0, down_cost_cents=73.0),
                    yes_bid_cents=27.0, no_bid_cents=72.0)
assert abs(km.vig_cents - 1.0) < 1e-9
assert E.KalshiMarket("T", 1.0, 0.0, None, None, None).vig_cents is None
print("[ok] vig_cents computes overround and tolerates an empty book")

# ---- 4. fill_outcomes prefers the exchange result over the candle proxy ----
# Build a row whose candle proxy would say DOWN but whose venue result says UP.
tmp = TMP / "live_fill.csv"
tmp.unlink(missing_ok=True)
past = time.time() - 3600
d = E.decide(65000, 64900, 5.0, 0.0006,
             E.Quote(up_cost_cents=40, down_cost_cents=62),
             expiry_ts=past, kalshi_ticker="KXBTC15M-TEST-00")
E.log_decision(d, path=tmp)
E.settlement_price = lambda ts: 1000.0                  # proxy would say DOWN
E.fetch_kalshi_settlement = lambda tk: True             # venue says UP
assert E.fill_outcomes(path=tmp) == 1
import csv
row = next(csv.DictReader(tmp.open()))
assert row["outcome_up"] == "1", row          # venue wins
assert row["kalshi_ticker"] == "KXBTC15M-TEST-00"
assert abs(float(row["pnl_cents"]) - 60.0) < 1e-6, row["pnl_cents"]
print("[ok] fill_outcomes settles from the exchange, not the candle proxy")

# ...and falls back to the candle when the venue has no result yet.
tmp2 = TMP / "live_fill2.csv"
tmp2.unlink(missing_ok=True)
E.log_decision(d, path=tmp2)
E.fetch_kalshi_settlement = lambda tk: None             # not settled
E.settlement_price = lambda ts: 1000.0                  # proxy says DOWN
assert E.fill_outcomes(path=tmp2) == 1
row2 = next(csv.DictReader(tmp2.open()))
assert row2["outcome_up"] == "0", row2
print("[ok] fill_outcomes falls back to the candle proxy when unsettled")

# ...and leaves the row alone when neither source can settle it.
tmp3 = TMP / "live_fill3.csv"
tmp3.unlink(missing_ok=True)
E.log_decision(d, path=tmp3)
E.fetch_kalshi_settlement = lambda tk: None
E.settlement_price = lambda ts: None
assert E.fill_outcomes(path=tmp3) == 0
print("[ok] fill_outcomes defers when neither source can settle")

# ---- 5. collect_samples honours the averaged-settlement contract ----
random.seed(9)
SIGMA = 0.0006
n = 8 * 1440
t0 = int(time.time() // 60 * 60) - n * 60
px, candles = 65000.0, []
for i in range(n):
    o = px
    px *= math.exp(random.gauss(-0.5 * SIGMA ** 2, SIGMA))
    candles.append({"ts": t0 + i * 60, "open": o, "high": max(o, px),
                    "low": min(o, px), "close": px, "volume": 1.0})
r_avg = E.backtest(candles, avg_settle=True)
r_pt = E.backtest(candles, avg_settle=False)
print(f"    avg_settle brier={r_avg.brier:.4f}  point brier={r_pt.brier:.4f}")
assert r_avg.samples > 5000 and r_avg.brier < 0.25
assert r_pt.samples > 5000 and r_pt.brier < 0.25
print("[ok] backtest runs and stays calibrated under both settlement rules")

print("\nALL LIVE CHECKS PASSED")
