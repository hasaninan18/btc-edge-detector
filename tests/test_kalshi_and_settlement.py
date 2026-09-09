"""Live-Kalshi work: TWAP-adjusted tau, dollar parsing, exchange settlement.

Converted from btc_tests/test_live.py. Network-touching assertions are marked
@pytest.mark.network (skipped unless --run-network); everything else is offline
and deterministic.
"""
import math
import random
from datetime import datetime, timezone

import csv
import pytest

import btc_prediction_edge as E
from _synth import gbm_ohlc

FILL = E  # module owning settlement_price / fetch_kalshi_settlement (patch target)


# ------------------------------------------------------------- effective_tau --

def test_effective_tau_closed_form_and_edge_cases():
    assert E.effective_tau(15.0) == pytest.approx(15.0 - 2.0 / 3.0, abs=1e-12)
    assert E.effective_tau(1.0) == pytest.approx(1.0 - 2.0 / 3.0, abs=1e-12)
    assert E.effective_tau(0.0) == 0.0
    assert E.effective_tau(-3.0) == 0.0
    assert E.effective_tau(5.0, avg_minutes=0.0) == pytest.approx(5.0, abs=1e-12)
    # inside the averaging window: tau^3 / (3 d^2), continuous with the branch above
    assert E.effective_tau(0.5) == pytest.approx(0.125 / 3.0, abs=1e-12)
    assert E.effective_tau(0.999) < E.effective_tau(1.001)


def test_effective_tau_matches_monte_carlo():
    random.seed(5)
    steps_per_min, tau, d, n_paths = 60, 5.0, 1.0, 40000
    dt = 1.0 / steps_per_min
    total_steps = int(tau * steps_per_min)
    start_avg = int((tau - d) * steps_per_min)
    vals = []
    for _ in range(n_paths):
        w = acc = 0.0
        cnt = 0
        for i in range(total_steps):
            w += random.gauss(0.0, math.sqrt(dt))
            if i >= start_avg:
                acc += w
                cnt += 1
        vals.append(acc / cnt)
    mc_var = sum(v * v for v in vals) / len(vals)
    want = E.effective_tau(tau)
    assert abs(mc_var - want) / want < 0.04


def test_twap_correction_sharpens_late_window():
    p_naive = E.prob_finish_above(65100, 65000, 2.0, 0.0006, avg_minutes=0.0)
    p_twap = E.prob_finish_above(65100, 65000, 2.0, 0.0006)
    assert p_twap > p_naive, "averaging cuts variance -> more confident when ITM"


# ------------------------------------------------------- dollar/ISO parsing ---

def test_market_cents_dollars_legacy_and_junk():
    m_new = {"yes_ask_dollars": "0.2800", "no_ask_dollars": "0.7300",
             "yes_ask": None, "no_ask": None}
    assert E._market_cents(m_new, "yes_ask") == pytest.approx(28.0)
    assert E._market_cents(m_new, "no_ask") == pytest.approx(73.0)
    assert E._market_cents(m_new, "yes_bid") is None
    assert E._market_cents({"yes_ask": 31}, "yes_ask") == pytest.approx(31.0)
    assert E._market_cents({"yes_ask_dollars": "junk"}, "yes_ask") is None


def test_iso_to_ts():
    t = E._iso_to_ts("2026-08-12T13:45:00Z")
    assert t == datetime(2026, 8, 12, 13, 45, tzinfo=timezone.utc).timestamp()
    assert E._iso_to_ts(None) is None
    assert E._iso_to_ts("nope") is None


def test_vig_cents_and_empty_book():
    km = E.KalshiMarket(ticker="T", strike=1.0, close_ts=0.0,
                        quote=E.Quote(up_cost_cents=28.0, down_cost_cents=73.0),
                        yes_bid_cents=27.0, no_bid_cents=72.0)
    assert km.vig_cents == pytest.approx(1.0)
    assert E.KalshiMarket("T", 1.0, 0.0, None, None, None).vig_cents is None


# --------------------------------------------- fill_outcomes source priority --

def _one_row_log(tmp_path, name):
    log = tmp_path / name
    import time
    d = E.decide(65000, 64900, 5.0, 0.0006,
                 E.Quote(up_cost_cents=40, down_cost_cents=62),
                 expiry_ts=time.time() - 3600, kalshi_ticker="KXBTC15M-TEST-00")
    E.log_decision(d, path=log)
    return log


def test_fill_prefers_exchange_result_over_candle(tmp_path, monkeypatch):
    log = _one_row_log(tmp_path, "live_fill.csv")
    monkeypatch.setattr(FILL, "settlement_price", lambda ts: 1000.0)   # proxy -> DOWN
    monkeypatch.setattr(FILL, "fetch_kalshi_settlement", lambda tk: True)  # venue -> UP
    assert E.fill_outcomes(path=log) == 1
    row = next(csv.DictReader(log.open()))
    assert row["outcome_up"] == "1"            # venue wins
    assert row["kalshi_ticker"] == "KXBTC15M-TEST-00"
    assert float(row["pnl_cents"]) == pytest.approx(60.0)


def test_fill_falls_back_to_candle_when_venue_unsettled(tmp_path, monkeypatch):
    log = _one_row_log(tmp_path, "live_fill2.csv")
    monkeypatch.setattr(FILL, "fetch_kalshi_settlement", lambda tk: None)
    monkeypatch.setattr(FILL, "settlement_price", lambda ts: 1000.0)   # proxy -> DOWN
    assert E.fill_outcomes(path=log) == 1
    assert next(csv.DictReader(log.open()))["outcome_up"] == "0"


def test_fill_defers_when_neither_source_can_settle(tmp_path, monkeypatch):
    log = _one_row_log(tmp_path, "live_fill3.csv")
    monkeypatch.setattr(FILL, "fetch_kalshi_settlement", lambda tk: None)
    monkeypatch.setattr(FILL, "settlement_price", lambda ts: None)
    assert E.fill_outcomes(path=log) == 0


# ----------------------------------------------- averaged-settlement backtest --

def test_backtest_runs_under_both_settlement_rules():
    candles = gbm_ohlc(seed=9, n_bars=8 * 1440)
    r_avg = E.backtest(candles, avg_settle=True)
    r_pt = E.backtest(candles, avg_settle=False)
    assert r_avg.samples > 5000 and r_avg.brier < 0.25
    assert r_pt.samples > 5000 and r_pt.brier < 0.25
