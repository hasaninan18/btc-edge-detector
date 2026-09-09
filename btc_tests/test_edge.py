"""Offline checks: synthetic GBM candles for the backtest, monkeypatched settlement for fill."""
import math, random, sys, time, csv
from pathlib import Path

sys.path.insert(0, "/Users/hasaninan/Downloads")
import btc_prediction_edge as E

# ---- 1. backtest on synthetic GBM: the model should be well calibrated ----
random.seed(7)
SIGMA = 0.0006          # per-minute vol
n = 6 * 1440            # 6 days of 1-min bars
t0 = int(time.time() // 60 * 60) - n * 60
px, candles = 65000.0, []
for i in range(n):
    px *= math.exp(random.gauss(-0.5 * SIGMA**2, SIGMA))
    candles.append({"ts": t0 + i * 60, "open": px, "high": px, "low": px,
                    "close": px, "volume": 1.0})

r = E.backtest(candles, vol_lookback=90, sample_every=1, avg_settle=False)
print(f"windows={r.windows} samples={r.samples} brier={r.brier:.4f} logloss={r.log_loss:.4f}")
print(r.calibration)
assert r.samples > 5000, r.samples
assert r.brier < 0.25, f"model worse than a coin flip on its own generating process: {r.brier}"

# recover implied sigma sanity: model told the truth about the DGP
print("\n[ok] backtest ran and beat coin-flip Brier on GBM data")

# with simulated market
r2 = E.backtest(candles, market_fn=E.vig_market(0.04), avg_settle=False)
print(f"sim bets={r2.bets} pnl={r2.pnl_cents:+.0f}c hit={r2.hit_rate}")
assert r2.bets > 0

# ---- 2. window scheduler boundary math ----
from datetime import datetime, timezone
for minute, want in [(0, 15), (7, 15), (15, 30), (44, 45), (46, 0), (59, 0)]:
    now = datetime(2026, 7, 22, 10, minute, 30, tzinfo=timezone.utc)
    got = E.next_window_expiry(now)
    assert got.minute == want and got > now, (minute, got)
print("[ok] next_window_expiry boundaries")

# ---- 3. fill_outcomes end-to-end with a stubbed settlement price ----
tmp = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp/paper_test.csv")
tmp.unlink(missing_ok=True)
past = time.time() - 3600
# one with a quote (UP recommended), one model-only
d1 = E.decide(65000, 64900, 5.0, 0.0006, E.Quote(up_cost_cents=40, down_cost_cents=62), expiry_ts=past)
d2 = E.decide(65000, 64900, 5.0, 0.0006, None, expiry_ts=past)
E.log_decision(d1, path=tmp); E.log_decision(d2, path=tmp)

E.settlement_price = lambda ts: 65500.0     # settles above strike -> UP wins
assert E.fill_outcomes(path=tmp) == 2
rows = list(csv.DictReader(tmp.open()))
assert [r["outcome_up"] for r in rows] == ["1", "1"], rows
assert rows[0]["recommended_side"] == "UP", rows[0]
assert abs(float(rows[0]["pnl_cents"]) - 60.0) < 1e-6, rows[0]["pnl_cents"]  # paid 40c, got 100c
assert float(rows[1]["pnl_cents"]) == 0.0                                    # no bet -> flat
assert E.fill_outcomes(path=tmp) == 0, "fill should be idempotent"
print("[ok] fill_outcomes: settles, computes PnL, idempotent")

# losing side check
E.settlement_price = lambda ts: 64000.0
tmp2 = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp/paper_test2.csv"); tmp2.unlink(missing_ok=True)
E.log_decision(d1, path=tmp2)
E.fill_outcomes(path=tmp2)
row = next(csv.DictReader(tmp2.open()))
assert row["outcome_up"] == "0" and abs(float(row["pnl_cents"]) + 40.0) < 1e-6, row
print("[ok] losing side PnL = -cost")

E.log_summary(path=tmp)
print("\nALL OFFLINE CHECKS PASSED")
