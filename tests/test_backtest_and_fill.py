"""Offline checks for the backtest harness and the outcome-fill pass.

Converted from btc_tests/test_edge.py: the sys.path hack and the hard-coded job
temp directory are gone; settlement is stubbed with monkeypatch and scratch
files go to pytest's tmp_path.
"""
import csv
from datetime import datetime, timezone

import pytest

import btc_edge as E
from _synth import gbm_flat

from btc_edge.live import fill as FILL  # module owning the settlement hooks patched below


def test_backtest_beats_coinflip_on_gbm():
    candles = gbm_flat(seed=7, n_bars=6 * 1440)
    r = E.backtest(candles, vol_lookback=90, sample_every=1, avg_settle=False)
    assert r.samples > 5000
    assert r.brier < 0.25, f"worse than a coin flip on its own DGP: {r.brier}"


def test_simulated_market_takes_bets():
    candles = gbm_flat(seed=7, n_bars=6 * 1440)
    r = E.backtest(candles, market_fn=E.vig_market(0.04), avg_settle=False)
    assert r.bets > 0


@pytest.mark.parametrize("minute,want", [(0, 15), (7, 15), (15, 30),
                                         (44, 45), (46, 0), (59, 0)])
def test_next_window_expiry_boundaries(minute, want):
    now = datetime(2026, 7, 22, 10, minute, 30, tzinfo=timezone.utc)
    got = E.next_window_expiry(now)
    assert got.minute == want and got > now


def test_fill_outcomes_settles_computes_pnl_and_is_idempotent(tmp_path, monkeypatch):
    log = tmp_path / "paper.csv"
    past = _hours_ago(1)
    d1 = E.decide(65000, 64900, 5.0, 0.0006,
                  E.Quote(up_cost_cents=40, down_cost_cents=62), expiry_ts=past)
    d2 = E.decide(65000, 64900, 5.0, 0.0006, None, expiry_ts=past)
    E.log_decision(d1, path=log)
    E.log_decision(d2, path=log)

    monkeypatch.setattr(FILL, "settlement_price", lambda ts: 65500.0)  # above strike
    assert E.fill_outcomes(path=log) == 2

    rows = list(csv.DictReader(log.open()))
    assert [r["outcome_up"] for r in rows] == ["1", "1"]
    assert rows[0]["recommended_side"] == "UP"
    assert float(rows[0]["pnl_cents"]) == pytest.approx(60.0)   # paid 40c, got 100c
    assert float(rows[1]["pnl_cents"]) == 0.0                   # no bet -> flat
    assert E.fill_outcomes(path=log) == 0, "fill must be idempotent"


def test_fill_outcomes_losing_side(tmp_path, monkeypatch):
    log = tmp_path / "paper.csv"
    d = E.decide(65000, 64900, 5.0, 0.0006,
                 E.Quote(up_cost_cents=40, down_cost_cents=62), expiry_ts=_hours_ago(1))
    E.log_decision(d, path=log)

    monkeypatch.setattr(FILL, "settlement_price", lambda ts: 64000.0)  # below strike
    E.fill_outcomes(path=log)

    row = next(csv.DictReader(log.open()))
    assert row["outcome_up"] == "0"
    assert float(row["pnl_cents"]) == pytest.approx(-40.0)     # lost the 40c stake


def test_log_summary_runs_on_settled_log(tmp_path, monkeypatch, capsys):
    log = tmp_path / "paper.csv"
    d = E.decide(65000, 64900, 5.0, 0.0006,
                 E.Quote(up_cost_cents=40, down_cost_cents=62), expiry_ts=_hours_ago(1))
    E.log_decision(d, path=log)
    monkeypatch.setattr(FILL, "settlement_price", lambda ts: 65500.0)
    E.fill_outcomes(path=log)

    E.log_summary(path=log)
    assert "settled samples: 1" in capsys.readouterr().out


def _hours_ago(h: float) -> float:
    import time
    return time.time() - h * 3600
