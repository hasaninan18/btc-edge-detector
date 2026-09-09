"""Phase 5: backfill outcomes and PnL once windows expire, and a quick summary.

`fill_outcomes` prefers the exchange's own settlement (when the row carries a
Kalshi ticker) and falls back to reconstructing the outcome from the Coinbase
candle the contract settles on. Idempotent.
"""
import csv
import os
import time
from pathlib import Path
from typing import Optional

from btc_edge.data import fetch_candle_range, fetch_kalshi_settlement
from btc_edge.metrics import _calibration_report
from btc_edge.live.paperlog import CSV_FIELDS, LOG_PATH


def settlement_price(expiry_ts: float) -> Optional[float]:
    """
    Close of the 1-min candle that the contract settles on. Returns None if the
    candle isn't published yet (Coinbase lags by a minute or two).
    """
    minute = int(expiry_ts // 60) * 60
    rows = fetch_candle_range(minute - 300, minute + 300)
    exact = [r for r in rows if r["ts"] == minute]
    if exact:
        return float(exact[0]["close"])
    before = [r for r in rows if r["ts"] < minute]
    return float(before[-1]["close"]) if before else None


def _pnl_cents(side: Optional[str], up_cost: Optional[float],
               down_cost: Optional[float], outcome_up: bool) -> Optional[float]:
    """PnL in cents for one paper contract at the logged side."""
    if not side:
        return 0.0            # we passed — flat, and that counts as a result
    if side == "UP":
        if up_cost is None:
            return None
        return (100.0 - up_cost * 100.0) if outcome_up else -up_cost * 100.0
    if down_cost is None:
        return None
    return (100.0 - down_cost * 100.0) if not outcome_up else -down_cost * 100.0


def fill_outcomes(path: Path = LOG_PATH, grace_seconds: int = 120) -> int:
    """
    Backfill settle_price / outcome_up / pnl_cents for every logged row whose
    window has closed. Idempotent — rows already filled are left alone.
    Returns the number of rows updated.
    """
    if not path.exists():
        print(f"no log at {path}")
        return 0

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0

    now = time.time()
    settle_cache: dict[int, Optional[float]] = {}
    result_cache: dict[str, Optional[bool]] = {}
    updated = 0

    for row in rows:
        if row.get("outcome_up"):
            continue
        try:
            expiry_ts = float(row["expiry_ts"])
        except (KeyError, ValueError):
            continue
        if now < expiry_ts + grace_seconds:
            continue   # not settled yet (or candle not published)

        # Ground truth first: if the row carries a Kalshi ticker, ask the venue
        # how it actually resolved. Only fall back to reconstructing the outcome
        # from a Coinbase candle when the exchange can't tell us — the two
        # disagree on near-the-money windows, because the contract settles on a
        # 60-second BRTI average rather than one exchange's minute close.
        ticker = (row.get("kalshi_ticker") or "").strip()
        outcome_up: Optional[bool] = None
        if ticker:
            if ticker not in result_cache:
                result_cache[ticker] = fetch_kalshi_settlement(ticker)
            outcome_up = result_cache[ticker]

        key = int(expiry_ts // 60)
        if key not in settle_cache:
            settle_cache[key] = settlement_price(expiry_ts)
        settle = settle_cache[key]

        if outcome_up is None:
            if settle is None:
                continue    # neither source can settle this row yet
            outcome_up = settle > float(row["strike"])
        up_cost = float(row["market_prob_up"]) if row.get("market_prob_up") else None
        down_cost = float(row["market_prob_down"]) if row.get("market_prob_down") else None
        pnl = _pnl_cents(row.get("recommended_side") or None, up_cost, down_cost, outcome_up)

        row["settle_price"] = "" if settle is None else f"{settle:.2f}"
        row["outcome_up"] = "1" if outcome_up else "0"
        row["pnl_cents"] = "" if pnl is None else f"{pnl:.2f}"
        updated += 1

    if updated:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)

    print(f"filled {updated} row(s) in {path}")
    return updated


def log_summary(path: Path = LOG_PATH) -> None:
    """Calibration + PnL over whatever has settled so far."""
    if not path.exists():
        print(f"no log at {path}")
        return
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("outcome_up")]
    if not rows:
        print("nothing settled yet")
        return
    probs = [float(r["model_prob_up"]) for r in rows]
    outcomes = [int(r["outcome_up"]) for r in rows]
    pnl = sum(float(r["pnl_cents"]) for r in rows if r.get("pnl_cents"))
    taken = sum(1 for r in rows if r.get("recommended_side"))
    print(f"\nsettled samples: {len(rows)}   bets taken: {taken}   "
          f"paper PnL: {pnl:+.1f}c")
    print(_calibration_report(probs, outcomes))
