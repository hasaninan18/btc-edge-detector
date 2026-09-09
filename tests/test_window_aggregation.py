"""Per-window aggregation in edge_report: the fix for correlated samples.

Converted from btc_tests/test_windowagg.py. Every sample inside a 15-minute
window resolves on the same settlement, so N samples must collapse to ONE
independent bet before any significance is claimed.
"""
import csv
import time

import btc_edge as E

from btc_edge.live import fill as FILL  # owns settlement_price / fetch_kalshi_settlement

PAST = time.time() - 7200


def _build(path, windows, monkeypatch):
    """windows: list of (expiry_ts, n_samples, settles_up). One UP bet each."""
    for expiry_ts, n, up in windows:
        for i in range(n):
            d = E.decide(65200, 64800, 5.0, 0.0006,
                         E.Quote(up_cost_cents=20, down_cost_cents=82),
                         expiry_ts=expiry_ts, kalshi_ticker=f"T-{int(expiry_ts)}")
            d.ts = f"2026-08-12T00:00:{i:02d}+00:00"   # stagger so "earliest" is defined
            E.log_decision(d, path=path)
    ups = {int(e): u for e, _, u in windows}
    monkeypatch.setattr(FILL, "fetch_kalshi_settlement", lambda tk: None)
    monkeypatch.setattr(FILL, "settlement_price",
                        lambda ts: 65500.0 if ups[int(ts)] else 64000.0)
    E.fill_outcomes(path=path)


def test_correlated_samples_collapse_to_one_window_bet(tmp_path, monkeypatch):
    log = tmp_path / "agg1.csv"
    _build(log, [(PAST, 45, 1)], monkeypatch)
    res = E.edge_report(path=log)

    assert res["bets"] == 45           # sample level still reports every row
    assert res["window_bets"] == 1     # but only one independent bet
    assert res["ci95"] is None         # no stats on n=1
    assert res["significant"] is False


def test_surviving_row_is_the_earliest_sample(tmp_path, monkeypatch):
    log = tmp_path / "agg1.csv"
    _build(log, [(PAST, 45, 1)], monkeypatch)
    E.edge_report(path=log)
    with log.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("recommended_side")]
    assert min(r["ts"] for r in rows) == "2026-08-12T00:00:00+00:00"


def test_one_lucky_window_is_not_significant(tmp_path, monkeypatch):
    log = tmp_path / "agg1.csv"
    _build(log, [(PAST, 45, 1)], monkeypatch)
    res = E.edge_report(path=log)
    assert res["pnl"] > 3000           # +80c * 45 at sample level
    assert res["mean_c"] > 0
    assert not res["significant"]


def test_many_break_even_windows_straddle_zero(tmp_path, monkeypatch):
    # Bet costs 20c, pays 100c -> break-even is winning 1 in 5. Mean PnL = 0.
    mixed = [(PAST - i * 900, 10, 1 if i % 5 == 0 else 0) for i in range(40)]
    log = tmp_path / "agg2.csv"
    _build(log, mixed, monkeypatch)
    res = E.edge_report(path=log)

    assert res["window_bets"] == 40
    lo, hi = res["ci95"]
    assert lo < 0 < hi
    assert res["significant"] is False


def test_many_winning_windows_are_significant(tmp_path, monkeypatch):
    allwin = [(PAST - i * 900, 10, 1) for i in range(40)]
    log = tmp_path / "agg3.csv"
    _build(log, allwin, monkeypatch)
    res = E.edge_report(path=log)

    assert res["window_bets"] == 40
    assert res["significant"] is True
    assert res["ci95"][0] > 0
