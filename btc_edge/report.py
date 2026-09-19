"""The model-vs-market edge report.

Only meaningful once rows carry BOTH a market quote and a settled outcome — i.e.
once quote capture (Kalshi) has been running. Until then it prints "no settled
rows with quotes yet".

The load-bearing idea: every sample inside a 15-min window resolves on the SAME
settlement, so the ~45 rows a window produces are one observation wearing 45
hats. The headline number aggregates to one bet per window before it puts a
confidence interval on anything.

That applies to the Brier comparison as much as to PnL, so the scoring section
reports the paired delta (model - market) at window level with a block
bootstrap over whole windows, and keeps the per-sample number only as a
labelled, non-inferential diagnostic.
"""
import csv
import math
from pathlib import Path
from statistics import stdev
from typing import Optional

from btc_edge.metrics import _brier, block_bootstrap_brier_delta
from btc_edge.live.paperlog import LOG_PATH

# Bootstrap settings live here so the printed report can state them and a test
# can pin the numbers. Seeded: two runs over the same log must agree exactly.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260909


def _window_id(row: dict) -> str:
    return row.get("window_id") or row.get("expiry_ts", "")


def _first_per_window(rows: list[dict]) -> list[dict]:
    """The earliest row of each window — the only one actable on in real time."""
    first: dict[str, dict] = {}
    for r in rows:
        wid = _window_id(r)
        prev = first.get(wid)
        if prev is None or r["ts"] < prev["ts"]:
            first[wid] = r
    return list(first.values())


def _triple(row: dict) -> tuple[float, float, int]:
    return (float(row["model_prob_up"]), float(row["market_prob_up"]),
            int(row["outcome_up"]))


def _brier_level(blocks: list[list[tuple[float, float, int]]],
                 n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> Optional[dict]:
    """Model/market Brier plus the paired delta and its block-bootstrap CI.

    One `block` per independent unit (a window). Resampling happens at block
    granularity, so this is correct whether a block holds one row (window-level)
    or all ~45 of a window's samples (the sample-level diagnostic). Shared with
    `market_backtest`, which passes its own resample count and seed.
    """
    flat = [t for b in blocks if b for t in b]
    if not flat:
        return None
    outcomes = [t[2] for t in flat]
    bs = block_bootstrap_brier_delta(blocks, n_resamples=n_resamples, seed=seed)
    return {
        "windows": bs["n_blocks"],
        "samples": len(flat),
        "model": _brier([t[0] for t in flat], outcomes),
        "market": _brier([t[1] for t in flat], outcomes),
        "delta": bs["delta"],
        "ci95": (bs["lo"], bs["hi"]),
        "p_model_better": bs["p_model_better"],
    }


def _print_brier_row(label: str, lvl: dict) -> None:
    lo, hi = lvl["ci95"]
    print(f"  {label:<26} {lvl['windows']:7d} {lvl['samples']:7d} "
          f"{lvl['model']:8.4f} {lvl['market']:8.4f} {lvl['delta']:+9.4f}   "
          f"[{lo:+.4f}, {hi:+.4f}]")


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
    win_bets = _first_per_window(settled_bets)
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

    # ---- scoring: the paired delta, one observation per window ----
    #
    # The aggregation choice, said out loud because it decides the number: the
    # headline row scores the FIRST ACTIONABLE sample of each window — the
    # earliest row that cleared the edge threshold and became a bet. Three
    # reasons. (1) It is the row that could actually have been traded; later
    # rows in the same window are hindsight. (2) It is exactly the row the PnL
    # headline below uses, so Brier and PnL describe the same decisions instead
    # of two different populations. (3) One row per window makes the
    # observations independent, which is the precondition for any interval.
    #
    # Its cost, said equally out loud: conditioning on "we bet" keeps only
    # windows where the model disagreed with the quote, which is not a fair
    # test of calibration in general. So the same delta over the first quoted
    # sample of EVERY window is printed underneath as the unconditional check.
    # The per-sample number — the one previously quoted as 0.1035 vs 0.1075 —
    # stays as a diagnostic and nothing more.
    by_window: dict[str, list[tuple[float, float, int]]] = {}
    for r in rows:
        by_window.setdefault(_window_id(r), []).append(_triple(r))
    lvl_sample = _brier_level(list(by_window.values()))
    lvl_all = _brier_level([[_triple(r)] for r in _first_per_window(rows)])
    lvl_traded = _brier_level([[_triple(r)] for r in win_bets])

    print("\nmodel vs market Brier — paired delta = model - market "
          "(negative = model beat the quote):")
    print(f"  {'level':<26} {'windows':>7} {'samples':>7} {'model':>8} "
          f"{'market':>8} {'delta':>9}   95% CI (block bootstrap)")
    if lvl_traded:
        _print_brier_row("window-level (traded)", lvl_traded)
    _print_brier_row("window-level (all quoted)", lvl_all)
    _print_brier_row("per-sample (diagnostic)", lvl_sample)
    print("    ^ correlated rows: one window counted up to 45 times. The CI is "
          "block-bootstrapped,")
    print("      but the point estimate is still weighted by samples-per-window "
          "— do NOT read as significance")
    print(f"  ({BOOTSTRAP_RESAMPLES:,} resamples drawing whole windows, "
          f"seed {BOOTSTRAP_SEED})")

    head = lvl_traded or lvl_all
    head_label = "traded windows" if lvl_traded else "all quoted windows"
    h_lo, h_hi = head["ci95"]
    if h_hi < 0:
        verdict = f"model beats the market on {head_label}, significant at 95%"
    elif h_lo > 0:
        verdict = f"market beats the model on {head_label}, significant at 95%"
    else:
        verdict = (f"CI straddles zero on {head_label} — no measurable scoring "
                   f"edge either way")
    print(f"  -> {verdict}; model scored better in "
          f"{head['p_model_better']:.0%} of resamples")

    brier = {"sample": lvl_sample, "window_traded": lvl_traded,
             "window_all": lvl_all,
             "bootstrap": {"n_resamples": BOOTSTRAP_RESAMPLES,
                           "seed": BOOTSTRAP_SEED}}

    if not settled_bets:
        print("\nno bets cleared the edge threshold yet")
        return {"n": len(rows), "pnl": 0.0, "bets": 0, "window_bets": 0,
                "mean_c": None, "ci95": None, "significant": False,
                "brier": brier}

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
            "significant": significant, "brier": brier}
