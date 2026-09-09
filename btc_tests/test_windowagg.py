"""Per-window aggregation in edge_report: the fix for correlated samples."""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/hasaninan/Downloads")
import btc_prediction_edge as E

TMP = Path("/Users/hasaninan/.claude/jobs/bc9d2a84/tmp")


def build(path: Path, windows: list[tuple[float, int, int]]) -> None:
    """windows: (expiry_ts, n_samples_in_window, settles_up). One UP bet each."""
    path.unlink(missing_ok=True)
    for expiry_ts, n, up in windows:
        for i in range(n):
            d = E.decide(65200, 64800, 5.0, 0.0006,
                         E.Quote(up_cost_cents=20, down_cost_cents=82),
                         expiry_ts=expiry_ts, kalshi_ticker=f"T-{int(expiry_ts)}")
            # stagger ts so "earliest sample in window" is well defined
            d.ts = f"2026-08-12T00:00:{i:02d}+00:00"
            E.log_decision(d, path=path)
    E.fetch_kalshi_settlement = lambda tk: None
    ups = {int(e): u for e, n, u in windows}
    E.settlement_price = lambda ts: 65500.0 if ups[int(ts)] else 64000.0
    E.fill_outcomes(path=path)


past = time.time() - 7200

# ---- 1. one window, many correlated samples -> exactly ONE window-bet ----
p1 = TMP / "agg1.csv"
build(p1, [(past, 45, 1)])
res = E.edge_report(path=p1)
assert res["bets"] == 45, res["bets"]          # sample level still reports all
assert res["window_bets"] == 1, res            # but only one independent bet
assert res["ci95"] is None                     # can't do stats on n=1
assert res["significant"] is False, "n=1 must never read as significant"
print(f"[ok] 45 correlated samples collapse to {res['window_bets']} window-bet, "
      f"no significance claimed")

# ---- 2. the surviving row is the EARLIEST sample in the window ----
with p1.open() as f:
    rows = [r for r in csv.DictReader(f) if r.get("recommended_side")]
assert min(r["ts"] for r in rows) == "2026-08-12T00:00:00+00:00"
print("[ok] window-level bet is the earliest actionable sample")

# ---- 3. the exact bug: all-winning correlated samples must NOT look significant ----
# 45 samples of one lucky window used to be 45 "independent" +80c wins.
# Sample-level PnL is hugely positive; window level must refuse to call it.
assert res["pnl"] > 3000, res["pnl"]           # +80c * 45 at sample level
assert res["mean_c"] > 0                       # the single window did win
assert not res["significant"]
print("[ok] one lucky window no longer masquerades as a significant edge")

# ---- 4. many windows at the break-even hit rate -> CI straddles zero ----
# The bet costs 20c and pays 100c, so break-even is winning 1 in 5: the mean is
# 0.2*(+80) + 0.8*(-20) = 0. A 50/50 hit rate here would be a genuine edge, not
# a null — the null has to be built from the contract's own break-even.
mixed = [(past - i * 900, 10, 1 if i % 5 == 0 else 0) for i in range(40)]
p2 = TMP / "agg2.csv"
build(p2, mixed)
res2 = E.edge_report(path=p2)
assert res2["window_bets"] == 40, res2["window_bets"]
lo, hi = res2["ci95"]
assert lo < 0 < hi, res2["ci95"]
assert res2["significant"] is False
print(f"[ok] 40 mixed windows -> CI [{lo:+.1f}, {hi:+.1f}] straddles zero")

# ---- 5. many windows all winning -> genuinely significant ----
allwin = [(past - i * 900, 10, 1) for i in range(40)]
p3 = TMP / "agg3.csv"
build(p3, allwin)
res3 = E.edge_report(path=p3)
assert res3["window_bets"] == 40
assert res3["significant"] is True, res3
assert res3["ci95"][0] > 0
print(f"[ok] 40 winning windows -> significant, CI lower bound "
      f"{res3['ci95'][0]:+.1f}c")

print("\nALL WINDOW-AGGREGATION CHECKS PASSED")
