"""Recalibration, the edge report, and the Kalshi quote stub.

Converted from btc_tests/test_prep.py. The live-Kalshi assertions are marked
@pytest.mark.network; the rest is offline.
"""
import time

import pytest

import btc_prediction_edge as E
from _synth import gbm_flat

FILL = E  # module owning settlement_price (patch target)


# ---------------------------------------------------------- recalibrator fit --

def test_recalibrator_sharpens_underdispersed_and_recovers_shrink():
    import random
    random.seed(11)
    shrink = 0.6
    samples = []
    for w in range(4000):
        p_true = random.random()
        o = 1 if random.random() < p_true else 0
        raw = E._sigmoid(E._logit(p_true) * shrink)   # under-dispersed model prob
        samples.append(E.Sample(raw_prob_up=raw, outcome_up=o,
                                minutes_left=7.0, window_ix=w))
    train = [s for s in samples if s.window_ix < 2800]
    test = [s for s in samples if s.window_ix >= 2800]
    rec = E.fit_recalibrator([s.raw_prob_up for s in train],
                             [s.outcome_up for s in train])
    raw_score = E.score_samples(test, recal=None)
    cal_score = E.score_samples(test, recal=rec)

    assert rec.a > 1.2, f"expected sharpening a>1.2, got {rec.a}"
    assert cal_score.log_loss < raw_score.log_loss
    assert cal_score.brier < raw_score.brier
    assert abs(rec.a - 1 / shrink) < 0.3   # recovers ~1/shrink = 1.667


def test_identity_recalibrator_is_a_noop():
    assert E.Recalibrator().apply(0.37) == 0.37


def test_recalibrator_save_load_and_missing_file(tmp_path):
    rec = E.Recalibrator(a=1.23, b=-0.05, n_fit=999)
    rp = tmp_path / "recal.json"
    rec.save(rp)
    rec2 = E.Recalibrator.load(rp)
    assert rec2.a == pytest.approx(rec.a) and rec2.n_fit == rec.n_fit
    assert E.Recalibrator.load(tmp_path / "missing.json").a == 1.0   # -> identity


def test_fit_and_eval_stays_near_identity_on_calibrated_gbm():
    candles = gbm_flat(seed=3, n_bars=12 * 1440)
    ev = E.fit_and_eval_recalibration(candles, split=0.7, vol_lookback=90)
    # True GBM is already calibrated, so the fit should not stray far from a=1 —
    # which is exactly why saving is gated on held-out improvement.
    assert 0.8 < ev.recal.a < 1.25


# --------------------------------------------------------------- edge_report --

def test_edge_report_over_synthetic_quoted_and_settled_rows(tmp_path, monkeypatch):
    log = tmp_path / "edge.csv"
    past = time.time() - 3600

    def make(price, strike, up, down):
        d = E.decide(price, strike, 5.0, 0.0006,
                     E.Quote(up_cost_cents=up, down_cost_cents=down), expiry_ts=past)
        E.log_decision(d, path=log)

    make(65200, 64800, 20, 82)   # big UP edge
    make(65200, 64800, 20, 82)
    make(65000, 65000, 50, 52)   # coin flip, no edge

    monkeypatch.setattr(FILL, "settlement_price", lambda ts: 65500.0)  # settles up
    E.fill_outcomes(path=log)

    res = E.edge_report(path=log)
    assert res is not None and res["n"] == 3
    assert res["bets"] == 2
    assert res["pnl"] > 0


def test_edge_report_on_missing_log_returns_none(tmp_path):
    assert E.edge_report(path=tmp_path / "nope.csv") is None


# ------------------------------------------------------------- Kalshi (live) --

@pytest.mark.network
def test_kalshi_quote_is_wellformed_or_none():
    q = E.fetch_kalshi_quote(65000.0, time.time() + 900)
    if q is None:
        return   # unreachable or between windows — acceptable, must not raise
    assert 0 < q.up_cost_cents < 100
    assert 0 < q.down_cost_cents < 100
    overround = q.up_cost_cents + q.down_cost_cents - 100
    assert -5 < overround < 25, f"implausible overround {overround}c"


@pytest.mark.network
def test_fetch_kalshi_settlement_unknown_ticker_is_none():
    assert E.fetch_kalshi_settlement("KXBTC15M-DOES-NOT-EXIST") is None
    assert E.fetch_kalshi_settlement("") is None


@pytest.mark.network
def test_kalshi_quote_fn_wrapper_is_callable_without_raising():
    fn = E.kalshi_quote_fn(65000.0, time.time() + 900)
    fn({"strike": 65000.0, "expiry_ts": time.time() + 900})   # returns Quote or None
