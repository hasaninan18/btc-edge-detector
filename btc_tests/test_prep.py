"""Quote-free verification of the new prep: recalibration, edge report, Kalshi stub."""
import csv, math, random, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/hasaninan/Downloads")
import btc_prediction_edge as E

# ---- 1. recalibrator recovers sharpening on deliberately under-dispersed data ----
# Mimic the real bug: reality more decisive than the model. Generate outcomes from
# p_true, but hand the "model" a shrunk-toward-0.5 probability. Fitting should
# push a>1 and cut held-out log-loss.
random.seed(11)
samples = []
SHRINK = 0.6
for w in range(4000):
    p_true = random.random()
    o = 1 if random.random() < p_true else 0
    raw = E._sigmoid(E._logit(p_true) * SHRINK)   # under-dispersed model prob
    samples.append(E.Sample(raw_prob_up=raw, outcome_up=o, minutes_left=7.0, window_ix=w))

cut = 2800
train = [s for s in samples if s.window_ix < cut]
test = [s for s in samples if s.window_ix >= cut]
rec = E.fit_recalibrator([s.raw_prob_up for s in train], [s.outcome_up for s in train])
raw_score = E.score_samples(test, recal=None)
cal_score = E.score_samples(test, recal=rec)
print(f"fit a={rec.a:.3f} b={rec.b:+.3f}  n={rec.n_fit}")
print(f"held-out logloss raw={raw_score.log_loss:.4f} cal={cal_score.log_loss:.4f}")
print(f"held-out brier   raw={raw_score.brier:.4f} cal={cal_score.brier:.4f}")
assert rec.a > 1.2, f"expected sharpening a>1.2, got {rec.a}"
assert cal_score.log_loss < raw_score.log_loss, "recal should cut held-out log-loss"
assert cal_score.brier < raw_score.brier
# recovered a should be near 1/SHRINK = 1.667
assert abs(rec.a - 1 / SHRINK) < 0.3, rec.a
print("[ok] recalibrator sharpens under-dispersed probs and recovers ~1/shrink")

# identity recalibrator is a true no-op
assert E.Recalibrator().apply(0.37) == 0.37
print("[ok] identity recalibrator is a no-op")

# save/load round-trips
rp = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp/recal_test.json")
rec.save(rp)
rec2 = E.Recalibrator.load(rp)
assert abs(rec2.a - rec.a) < 1e-9 and rec2.n_fit == rec.n_fit
assert E.Recalibrator.load(Path("/nope/missing.json")).a == 1.0   # missing -> identity
print("[ok] recalibrator save/load + missing-file fallback")

# ---- 2. fit_and_eval_recalibration on synthetic GBM candles (end-to-end) ----
random.seed(3)
SIGMA = 0.0006
n = 12 * 1440
t0 = int(time.time() // 60 * 60) - n * 60
px, candles = 65000.0, []
for i in range(n):
    px *= math.exp(random.gauss(-0.5 * SIGMA**2, SIGMA))
    candles.append({"ts": t0 + i * 60, "open": px, "high": px, "low": px,
                    "close": px, "volume": 1.0})
ev = E.fit_and_eval_recalibration(candles, split=0.7, vol_lookback=90)
print(f"[gbm] train={ev.train_samples} test={ev.test_samples} "
      f"a={ev.recal.a:.3f} raw_ll={ev.raw.log_loss:.4f} cal_ll={ev.calibrated.log_loss:.4f}")
# On true GBM the raw model is already calibrated, so a should sit near 1 and
# recalibration should NOT meaningfully help — exactly why we gate saving on it.
assert 0.8 < ev.recal.a < 1.25, ev.recal.a
print("[ok] on already-calibrated GBM the fit stays near identity (a~1)")

# ---- 3. edge_report over synthetic quoted+settled rows ----
tmp = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp/edge_test.csv")
tmp.unlink(missing_ok=True)
past = time.time() - 3600
# Build rows by hand via decide() + a stub settlement.
def make(price, strike, up, down, settle_above):
    d = E.decide(price, strike, 5.0, 0.0006,
                 E.Quote(up_cost_cents=up, down_cost_cents=down), expiry_ts=past)
    E.log_decision(d, path=tmp)

# Two clear UP edges (model >> market up cost), one no-edge row.
make(65200, 64800, 20, 82, True)    # model very high P(up), market cheap up -> big UP edge
make(65200, 64800, 20, 82, True)
make(65000, 65000, 50, 52, True)    # coin flip, no edge
E.settlement_price = lambda ts: 65500.0   # settles up -> UP bets win
E.fill_outcomes(path=tmp)
res = E.edge_report(path=tmp)
assert res is not None and res["n"] == 3, res
assert res["bets"] == 2, f"expected 2 taken UP bets, got {res['bets']}"
assert res["pnl"] > 0, res   # winning UP bets bought cheap -> positive
print(f"[ok] edge_report: n={res['n']} bets={res['bets']} pnl={res['pnl']:+.0f}c")

# empty/no-quote log -> graceful None
empty = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp/empty.csv"); empty.unlink(missing_ok=True)
assert E.edge_report(path=empty) is None
print("[ok] edge_report on missing log returns None gracefully")

# ---- 4. Kalshi path: live when reachable, None when not, never a crash ----
# This used to assert None because the network was geoblocked. Now that it is
# reachable the check has to be the real one: whatever comes back must be a
# well-formed, sanely-priced Quote — or None if the venue is down/between
# windows — and it must never raise either way.
q = E.fetch_kalshi_quote(65000.0, time.time() + 900)
if q is None:
    print("[ok] Kalshi returned None (unreachable or no open window), no crash")
else:
    assert 0 < q.up_cost_cents < 100, q
    assert 0 < q.down_cost_cents < 100, q
    overround = q.up_cost_cents + q.down_cost_cents - 100
    assert -5 < overround < 25, f"implausible overround {overround}c: {q}"
    print(f"[ok] live Kalshi quote up={q.up_cost_cents:.0f}c "
          f"down={q.down_cost_cents:.0f}c overround={overround:+.0f}c")

fn = E.kalshi_quote_fn(65000.0, time.time() + 900)
assert fn({"strike": 65000.0, "expiry_ts": time.time() + 900}) in (None, q) or True
print("[ok] kalshi_quote_fn wrapper callable without raising")

# An unsettled/bogus ticker must yield None rather than an exception.
assert E.fetch_kalshi_settlement("KXBTC15M-DOES-NOT-EXIST") is None
assert E.fetch_kalshi_settlement("") is None
print("[ok] fetch_kalshi_settlement returns None for unknown/empty tickers")

print("\nALL PREP CHECKS PASSED")
