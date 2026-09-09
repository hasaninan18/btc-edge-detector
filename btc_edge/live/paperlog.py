"""The paper-trade CSV: schema and append.

One row per Decision, with three outcome columns left blank until the window
settles (filled by btc_edge.live.fill).
"""
import csv
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path

from btc_edge.decision import Decision

LOG_PATH = Path("paper_trades.csv")

OUTCOME_FIELDS = ["settle_price", "outcome_up", "pnl_cents"]
CSV_FIELDS = [f.name for f in dataclass_fields(Decision)] + OUTCOME_FIELDS


def log_decision(d: Decision, path: Path = LOG_PATH) -> None:
    row = asdict(d)
    row.update({k: "" for k in OUTCOME_FIELDS})   # filled in after expiry
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
