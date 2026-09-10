"""Golden master: pins the current numeric output of edge_report() and backtest().

This test exists to make refactoring safe. It captures the exact numbers the
code produces *today* — before the single module was split into packages — so
any later change that moves a number, however small, fails here loudly.

If you changed modelling logic on purpose and this fails, update the constants
below in the same commit and say so in the message. If you were only moving
code around, a failure here means you broke something.

Inputs are frozen:
  * edge_report() runs against tests/fixtures/paper_trades.csv (a copy of the
    real 217-row paper log as of 2026-08-13).
  * backtest() runs against synthetic GBM candles from a fixed seed and a fixed
    epoch, so the window boundaries and every sample are deterministic.
"""
import io
import math
import random
from contextlib import redirect_stdout
from dataclasses import asdict

import pytest

import btc_edge as E

REL = 1e-9


def _synthetic_candles(seed: int = 1234, days: int = 8) -> list[dict]:
    """Deterministic GBM 1-minute candles. Must not change — golden inputs."""
    random.seed(seed)
    sigma = 0.0006
    n = days * 1440
    t0 = 1_700_000_000 // 60 * 60 - n * 60
    px = 65000.0
    candles = []
    for i in range(n):
        o = px
        px *= math.exp(random.gauss(-0.5 * sigma ** 2, sigma))
        candles.append({"ts": t0 + i * 60, "open": o, "high": max(o, px),
                        "low": min(o, px), "close": px, "volume": 1.0})
    return candles


# ---------------------------------------------------------------- edge_report --

def test_edge_report_pnl_dict_is_pinned(fixtures_dir):
    """The PnL half of the return value. Unchanged since the original module.

    `brier` was added later (window-level scoring) and is pinned separately
    below; it is popped here so this constant stays the one the pre-package
    code produced.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = E.edge_report(path=fixtures_dir / "paper_trades.csv")
    res.pop("brier")

    assert res == {
        "n": 204,
        "pnl": pytest.approx(1201.2, rel=REL),
        "bets": 82,
        "window_bets": 32,
        "mean_c": pytest.approx(14.5375, rel=REL),
        "ci95": pytest.approx((-0.14644702777547103, 29.22144702777547), rel=REL),
        "significant": False,
    }


def test_edge_report_brier_dict_is_pinned(fixtures_dir):
    """Window-level Brier, the paired delta, and its block-bootstrap interval.

    The bootstrap is seeded, so these interval bounds are exact numbers, not
    approximately-right ones. Changing BOOTSTRAP_SEED or BOOTSTRAP_RESAMPLES in
    btc_edge/report.py moves them and must be a deliberate edit here too.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = E.edge_report(path=fixtures_dir / "paper_trades.csv")

    assert res["brier"] == {
        "bootstrap": {"n_resamples": 10_000, "seed": 20260909},
        # The number this project used to quote: 204 correlated rows.
        "sample": {
            "windows": 54,
            "samples": 204,
            "model": pytest.approx(0.10347969181072715, rel=REL),
            "market": pytest.approx(0.10751372058823529, rel=REL),
            "delta": pytest.approx(-0.004034028777508143, rel=REL),
            "ci95": pytest.approx(
                (-0.015555009613474596, 0.01110820872183993), rel=REL),
            "p_model_better": pytest.approx(0.7504, rel=REL),
        },
        # The headline: the 32 rows the PnL number is also computed from.
        "window_traded": {
            "windows": 32,
            "samples": 32,
            "model": pytest.approx(0.19732156049114027, rel=REL),
            "market": pytest.approx(0.197788, rel=REL),
            "delta": pytest.approx(-0.0004664395088597173, rel=REL),
            "ci95": pytest.approx(
                (-0.04814834074285129, 0.060609936629548694), rel=REL),
            "p_model_better": pytest.approx(0.5518, rel=REL),
        },
        # Unconditional check: first quoted sample of every window. Note the
        # delta is POSITIVE here — the market scores better once the sample is
        # not selected on the model having disagreed with it.
        "window_all": {
            "windows": 54,
            "samples": 54,
            "model": pytest.approx(0.19772247955846492, rel=REL),
            "market": pytest.approx(0.19461931481481484, rel=REL),
            "delta": pytest.approx(0.0031031647436501, rel=REL),
            "ci95": pytest.approx(
                (-0.024349440976947686, 0.04134042974296799), rel=REL),
            "p_model_better": pytest.approx(0.4664, rel=REL),
        },
    }


def test_edge_report_brier_lines_are_pinned(fixtures_dir):
    buf = io.StringIO()
    with redirect_stdout(buf):
        E.edge_report(path=fixtures_dir / "paper_trades.csv")
    out = buf.getvalue()

    assert "quoted & settled: 204 samples across 54 windows" in out
    assert ("  window-level (traded)           32      32   0.1973   0.1978   "
            "-0.0005   [-0.0481, +0.0606]") in out
    assert ("  window-level (all quoted)       54      54   0.1977   0.1946   "
            "+0.0031   [-0.0243, +0.0413]") in out
    assert ("  per-sample (diagnostic)         54     204   0.1035   0.1075   "
            "-0.0040   [-0.0156, +0.0111]") in out
    assert "do NOT read as significance" in out
    assert "CI straddles zero on traded windows" in out
    assert "WINDOW-LEVEL (independent — this is the number that counts):" in out
    assert "bets 32   hit 65.6%   PnL +465c   avg +14.54c/bet" in out
    assert "95% CI: [-0.15c, +29.22c]" in out
    assert "consistent with ZERO edge" in out


# ------------------------------------------------------------------- backtest --

def test_backtest_avg_settle_is_pinned():
    r = E.backtest(_synthetic_candles(), avg_settle=True)
    assert r.windows == 761
    assert r.samples == 11415
    assert r.brier == pytest.approx(0.1383139076909232, rel=REL)
    assert r.log_loss == pytest.approx(0.41698716415151404, rel=REL)
    assert r.bets == 0
    assert r.pnl_cents == 0.0
    assert r.hit_rate is None


def test_backtest_point_settle_is_pinned():
    r = E.backtest(_synthetic_candles(), avg_settle=False)
    assert r.windows == 761
    assert r.samples == 11415
    assert r.brier == pytest.approx(0.1669590317009093, rel=REL)
    assert r.log_loss == pytest.approx(0.49214138619367825, rel=REL)


def test_backtest_with_saved_recalibrator_is_pinned():
    recal = E.Recalibrator.load(E.RECAL_PATH)
    assert recal.n_fit == 30160  # the shipped fit; guards against a swapped file
    r = E.backtest(_synthetic_candles(), avg_settle=True, recal=recal)
    assert r.brier == pytest.approx(0.13810183102966198, rel=REL)
    assert r.log_loss == pytest.approx(0.416135824792117, rel=REL)


def test_backtest_simulated_market_pnl_is_pinned():
    r = E.backtest(_synthetic_candles(), avg_settle=False,
                   market_fn=E.vig_market(0.04))
    assert r.bets == 1775
    assert r.pnl_cents == pytest.approx(8604.902739168743, rel=REL)
    assert r.hit_rate == pytest.approx(0.9864788732394366, rel=REL)


def test_backtestresult_shape_is_stable():
    """The dataclass field set is part of the contract downstream code reads."""
    r = E.backtest(_synthetic_candles(), avg_settle=True)
    assert set(asdict(r)) == {
        "windows", "samples", "brier", "log_loss", "calibration",
        "bets", "pnl_cents", "hit_rate",
    }
