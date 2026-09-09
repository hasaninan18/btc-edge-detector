"""The model-vs-market edge report.

Only meaningful once rows carry BOTH a market quote and a settled outcome — i.e.
once quote capture (Kalshi) has been running. Until then it prints "no settled
rows with quotes yet".

The load-bearing idea: every sample inside a 15-min window resolves on the SAME
settlement, so the ~45 rows a window produces are one observation wearing 45
hats. The headline number aggregates to one bet per window before it puts a
confidence interval on anything.
"""
import csv
import math
from pathlib import Path
from statistics import stdev
from typing import Optional

from btc_edge.metrics import _brier
from btc_edge.live.paperlog import LOG_PATH


def edge_report(path: Path = LOG_PATH) -> Optional[dict]:
    """
    Over every settled sample that also had a market quote:
      - how the model's probability compared to the market's,
      - whether acting on our edge threshold actually made money,
      - a breakdown by how big the edge looked at decision time.
    """
    if not path.exists():
        print(f"no log at {path}")
        return None
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("outcome_up") and r.get("market_prob_up")]
    if not rows:
        print("no settled rows with quotes yet — capture quotes first "
              "(watch --prompt-quotes / --kalshi), then re-run")
        return None

    # Calibration of the model against realized outcomes, on quoted samples only.
    m_probs = [float(r["model_prob_up"]) for r in rows]
    outcomes = [int(r["outcome_up"]) for r in rows]

    # Did the market itself predict well? (its implied prob vs realized)
    mkt_probs = [float(r["market_prob_up"]) for r in rows]

    taken = [r for r in rows if r.get("recommended_side")]
    settled_bets = [r for r in taken if r.get("pnl_cents")]
    pnl = sum(float(r["pnl_cents"]) for r in settled_bets)
    wins = sum(1 for r in settled_bets
               if (r["recommended_side"] == "UP") == (r["outcome_up"] == "1"))

    # ---- the part that decides anything: one independent bet per window ----
    #
    # Every sample inside a 15-min window resolves on the SAME settlement, so
    # the ~45 rows a window produces are one observation wearing 45 hats. Summing
    # their PnL and treating the total as 45 independent bets understates the
    # standard error by ~sqrt(45), which is how a single lucky window turns into
    # a "significant" edge. So the headline below aggregates to one bet per
    # window — the earliest sample that cleared the threshold, which is also the
    # only one you could realistically have acted on in real time.
    per_window: dict[str, dict] = {}
    for r in settled_bets:
        wid = r.get("window_id") or r.get("expiry_ts", "")
        prev = per_window.get(wid)
        if prev is None or r["ts"] < prev["ts"]:
            per_window[wid] = r
    win_bets = list(per_window.values())
    win_pnls = [float(r["pnl_cents"]) for r in win_bets]

    # Bucket taken bets by the edge we thought we had (window-level).
    def edge_of(r) -> float:
        return (float(r["edge_up"]) if r["recommended_side"] == "UP"
                else float(r["edge_down"]))
    buckets = {"5-10%": [], "10-20%": [], "20%+": []}
    for r in win_bets:
        e = edge_of(r)
        key = "5-10%" if e < 0.10 else "10-20%" if e < 0.20 else "20%+"
        buckets[key].append(float(r["pnl_cents"]))

    n_windows = len({r.get("window_id") for r in rows})
    print(f"\nquoted & settled: {len(rows)} samples across {n_windows} windows")
    print(f"model  Brier vs outcomes: {_brier(m_probs, outcomes):.4f}")
    print(f"market Brier vs outcomes: {_brier(mkt_probs, outcomes):.4f}  "
          f"(if the market beats the model, there's no edge to take)")

    if not settled_bets:
        print("\nno bets cleared the edge threshold yet")
        return {"n": len(rows), "pnl": 0.0, "bets": 0, "window_bets": 0,
                "mean_c": None, "ci95": None, "significant": False}

    print(f"\nsample-level (correlated — do NOT read as significance):")
    print(f"  bets {len(settled_bets)}   hit {wins / len(settled_bets):.1%}   "
          f"PnL {pnl:+,.0f}c   avg {pnl / len(settled_bets):+.2f}c")

    mean_c = sum(win_pnls) / len(win_pnls)
    win_hits = sum(1 for r in win_bets
                   if (r["recommended_side"] == "UP") == (r["outcome_up"] == "1"))
    print(f"\nWINDOW-LEVEL (independent — this is the number that counts):")
    print(f"  bets {len(win_pnls)}   hit {win_hits / len(win_pnls):.1%}   "
          f"PnL {sum(win_pnls):+,.0f}c   avg {mean_c:+.2f}c/bet")

    ci = None
    significant = False
    if len(win_pnls) >= 2:
        sd = stdev(win_pnls)
        se = sd / math.sqrt(len(win_pnls))
        lo, hi = mean_c - 1.96 * se, mean_c + 1.96 * se
        ci = (lo, hi)
        significant = lo > 0 or hi < 0
        print(f"  95% CI: [{lo:+.2f}c, {hi:+.2f}c]   (sd {sd:.1f}c, se {se:.2f}c)")
        if not significant:
            print("  -> consistent with ZERO edge; keep collecting")
            # Sizing off the observed mean is the classic trap: that mean is
            # itself noisy, so near the boundary it reports "almost done" no
            # matter how little you know. Size off the pessimistic end of the
            # CI as well — that is the number that survives the estimate moving.
            if abs(mean_c) > 1e-9:
                need = (1.96 * sd / abs(mean_c)) ** 2
                print(f"  -> if the true edge really is {mean_c:+.1f}c: "
                      f"~{need:,.0f} window-bets total "
                      f"({max(0.0, need - len(win_pnls)):,.0f} more)")
            worst = min(abs(lo), abs(hi))
            if worst > 1e-9:
                need_w = (1.96 * sd / worst) ** 2
                print(f"  -> if it is only {worst:+.1f}c (pessimistic end of the CI): "
                      f"~{need_w:,.0f} window-bets total "
                      f"({max(0.0, need_w - len(win_pnls)):,.0f} more)")
            print("  -> plan against the second number; the first assumes the "
                  "point estimate is exact, which is what got you here")
        else:
            direction = "POSITIVE" if lo > 0 else "NEGATIVE"
            print(f"  -> {direction} edge, significant at 95%")
    else:
        print("  (need >=2 window-bets before a confidence interval means anything)")

    print("\nby perceived edge at entry (window-level):")
    print("  edge band     n     PnL      avg/bet")
    for k, v in buckets.items():
        if v:
            print(f"  {k:<10} {len(v):4d} {sum(v):+8.0f}c {sum(v)/len(v):+8.2f}c")

    # Staleness diagnostic. If spot moved materially between reading the price
    # and getting the quote back, some of the "edge" above is just us holding a
    # newer clock than the market we are scoring ourselves against.
    drifts = [abs(float(r["price_post"]) - float(r["price"]))
              for r in win_bets if r.get("price_post") and r.get("price")]
    if drifts:
        lags = [float(r["quote_lag_ms"]) for r in win_bets if r.get("quote_lag_ms")]
        avg_drift = sum(drifts) / len(drifts)
        px = sum(float(r["price"]) for r in win_bets if r.get("price")) / len(win_bets)
        print(f"\nquote staleness: spot moved {avg_drift:.2f} on average "
              f"({avg_drift / px:.4%}) during a "
              f"{sum(lags) / len(lags):.0f}ms round trip"
              if lags else f"\nquote staleness: spot moved {avg_drift:.2f} on average")
        print("  (spot is read BEFORE the quote, so this biases against us)")
    else:
        print("\nquote staleness: not measured on these rows (pre-fix capture) — "
              "treat the edge above as an UPPER bound")

    return {"n": len(rows), "pnl": pnl, "bets": len(settled_bets),
            "window_bets": len(win_pnls), "mean_c": mean_c, "ci95": ci,
            "significant": significant}
