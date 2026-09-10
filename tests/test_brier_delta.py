"""Paired Brier delta and the window-block bootstrap behind its interval.

The property these tests are really defending: an interval on the model-vs-market
Brier delta must be computed over WINDOWS, not rows. Rows inside a 15-minute
window resolve on one settlement, so resampling rows individually manufactures
independent observations that do not exist and reports an interval that is far
too tight. Two tests below pin that difference directly (block invariance under
row duplication, and the naive-vs-block width gap) so a future "simplification"
back to a per-row bootstrap fails loudly instead of quietly.
"""
import csv
import random
from pathlib import Path

import pytest

import btc_edge as E
from btc_edge.metrics import block_bootstrap_brier_delta, brier_delta
from btc_edge.report import _first_per_window, _triple, _window_id


# ------------------------------------------------------------- paired delta --

def test_brier_delta_is_model_minus_market():
    m = [0.9, 0.2, 0.6]
    k = [0.5, 0.5, 0.5]
    o = [1, 0, 1]
    assert brier_delta(m, k, o) == pytest.approx(E._brier(m, o) - E._brier(k, o))


def test_brier_delta_negative_when_model_is_sharper_and_right():
    o = [1, 1, 0, 0]
    confident_and_right = [0.95, 0.95, 0.05, 0.05]
    coinflip = [0.5] * 4
    assert brier_delta(confident_and_right, coinflip, o) < 0
    # ...and positive the other way round: sharp and WRONG is worse than 50/50.
    confident_and_wrong = [0.05, 0.05, 0.95, 0.95]
    assert brier_delta(confident_and_wrong, coinflip, o) > 0


# --------------------------------------------------------- bootstrap basics --

def _blocks(n_windows: int = 12, rows_per_window: int = 1, seed: int = 0):
    """Deterministic toy blocks, deliberately heterogeneous.

    Blocks must differ from one another or every resample lands on the same
    delta and the bootstrap looks reproducible for the wrong reason.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n_windows):
        o = rng.randint(0, 1)
        m = round(rng.uniform(0.05, 0.95), 3)
        k = round(rng.uniform(0.05, 0.95), 3)
        out.append([(m, k, o)] * rows_per_window)
    return out


def test_bootstrap_point_estimate_matches_the_plain_delta():
    blocks = _blocks()
    flat = [t for b in blocks for t in b]
    bs = block_bootstrap_brier_delta(blocks, n_resamples=200, seed=1)
    assert bs["delta"] == pytest.approx(
        brier_delta([t[0] for t in flat], [t[1] for t in flat],
                    [t[2] for t in flat]))


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    a = block_bootstrap_brier_delta(_blocks(), n_resamples=500, seed=7)
    b = block_bootstrap_brier_delta(_blocks(), n_resamples=500, seed=7)
    c = block_bootstrap_brier_delta(_blocks(), n_resamples=500, seed=8)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])  # a real resample, not a stub


def test_bootstrap_interval_contains_the_point_estimate():
    bs = block_bootstrap_brier_delta(_blocks(), n_resamples=2000, seed=3)
    assert bs["lo"] <= bs["delta"] <= bs["hi"]


def test_bootstrap_reports_its_own_provenance():
    bs = block_bootstrap_brier_delta(_blocks(n_windows=9), n_resamples=64, seed=5)
    assert bs["n_blocks"] == 9
    assert bs["n_resamples"] == 64
    assert bs["seed"] == 5
    assert 0.0 <= bs["p_model_better"] <= 1.0


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError):
        block_bootstrap_brier_delta([], n_resamples=10, seed=0)
    with pytest.raises(ValueError):
        block_bootstrap_brier_delta([[], []], n_resamples=10, seed=0)
    with pytest.raises(ValueError):
        block_bootstrap_brier_delta(_blocks(), n_resamples=0, seed=0)


def test_empty_blocks_are_skipped_not_counted():
    blocks = _blocks(n_windows=6)
    padded = blocks + [[], []]
    a = block_bootstrap_brier_delta(blocks, n_resamples=300, seed=11)
    b = block_bootstrap_brier_delta(padded, n_resamples=300, seed=11)
    assert a["n_blocks"] == b["n_blocks"] == 6
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])


# ------------------------------------------------- the block structure bites --

def test_duplicating_rows_inside_a_window_changes_nothing():
    """45 copies of one observation are still one observation.

    Scaling every block's row count by the same factor scales its SSE and its n
    together, so every resampled delta is identical. If this ever fails, the
    bootstrap has started counting rows instead of windows.
    """
    one = block_bootstrap_brier_delta(_blocks(rows_per_window=1),
                                      n_resamples=1000, seed=42)
    many = block_bootstrap_brier_delta(_blocks(rows_per_window=45),
                                       n_resamples=1000, seed=42)
    assert many["delta"] == pytest.approx(one["delta"])
    assert many["lo"] == pytest.approx(one["lo"])
    assert many["hi"] == pytest.approx(one["hi"])


def test_naive_per_row_bootstrap_invents_precision_on_correlated_rows():
    """The mistake this module exists to prevent, at its clearest.

    45 exact copies of each window's single observation. Resampling rows
    individually sees 45x the sample size and shrinks the interval by roughly
    sqrt(45) ~ 6.7, while the block bootstrap correctly reports the same
    interval it would report on the 12 underlying observations. Both agree on
    the point estimate — the fabricated part is only the precision.
    """
    windows = _blocks(n_windows=12, rows_per_window=45)
    rows = [t for b in windows for t in b]

    block = block_bootstrap_brier_delta(windows, n_resamples=2000, seed=1)
    naive = block_bootstrap_brier_delta([[t] for t in rows],
                                        n_resamples=2000, seed=1)

    assert block["delta"] == pytest.approx(naive["delta"])
    assert (block["hi"] - block["lo"]) > 4 * (naive["hi"] - naive["lo"])


def test_block_bootstrap_is_wider_than_the_per_row_one_on_the_real_log(
        fixtures_dir):
    """Same comparison on the actual 204-row paper log.

    The gap is smaller here than in the synthetic case (observed ratio ~1.23x,
    seeded and deterministic) because samples inside a real window drift apart
    as expiry approaches rather than being identical copies. It is still the
    wrong direction to be wrong in, so it is pinned.
    """
    with (fixtures_dir / "paper_trades.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("outcome_up") and r.get("market_prob_up")]

    by_window: dict[str, list] = {}
    for r in rows:
        by_window.setdefault(_window_id(r), []).append(_triple(r))

    block = block_bootstrap_brier_delta(list(by_window.values()),
                                        n_resamples=2000, seed=1)
    naive = block_bootstrap_brier_delta([[_triple(r)] for r in rows],
                                        n_resamples=2000, seed=1)

    assert block["delta"] == pytest.approx(naive["delta"])  # same point estimate
    assert (block["hi"] - block["lo"]) > 1.15 * (naive["hi"] - naive["lo"])


# ------------------------------------------------------ wiring in the report --

def test_first_per_window_takes_the_earliest_row():
    rows = [
        {"window_id": "w1", "ts": "2026-08-12T14:05:00", "tag": "late"},
        {"window_id": "w1", "ts": "2026-08-12T14:01:00", "tag": "early"},
        {"window_id": "w2", "ts": "2026-08-12T14:20:00", "tag": "only"},
    ]
    got = {r["window_id"]: r["tag"] for r in _first_per_window(rows)}
    assert got == {"w1": "early", "w2": "only"}


def test_first_per_window_falls_back_to_expiry_ts():
    rows = [{"expiry_ts": "1786544100.0", "ts": "b"},
            {"expiry_ts": "1786544100.0", "ts": "a"}]
    assert [r["ts"] for r in _first_per_window(rows)] == ["a"]


def test_edge_report_exposes_all_three_levels(fixtures_dir, capsys):
    res = E.edge_report(path=fixtures_dir / "paper_trades.csv")
    b = res["brier"]

    assert set(b) == {"sample", "window_traded", "window_all", "bootstrap"}
    assert b["bootstrap"] == {"n_resamples": 10_000, "seed": 20260909}

    # Window levels are one row per window; the sample level is not.
    assert b["window_traded"]["windows"] == b["window_traded"]["samples"] == 32
    assert b["window_all"]["windows"] == b["window_all"]["samples"] == 54
    assert b["sample"]["windows"] == 54
    assert b["sample"]["samples"] == 204

    # The traded rows are the same 32 the PnL headline uses.
    assert b["window_traded"]["windows"] == res["window_bets"]

    for lvl in (b["sample"], b["window_traded"], b["window_all"]):
        assert lvl["delta"] == pytest.approx(lvl["model"] - lvl["market"])
        assert lvl["ci95"][0] <= lvl["delta"] <= lvl["ci95"][1]


def test_edge_report_keeps_the_correlation_warning_on_the_sample_line(
        fixtures_dir, capsys):
    E.edge_report(path=fixtures_dir / "paper_trades.csv")
    out = capsys.readouterr().out

    assert "paired delta = model - market" in out
    assert "window-level (traded)" in out
    assert "window-level (all quoted)" in out
    assert "per-sample (diagnostic)" in out
    assert "do NOT read as significance" in out
    assert "10,000 resamples drawing whole windows, seed 20260909" in out


def test_edge_report_is_deterministic_across_runs(fixtures_dir, capsys):
    """A seeded bootstrap must not move the reported interval between runs."""
    a = E.edge_report(path=fixtures_dir / "paper_trades.csv")["brier"]
    b = E.edge_report(path=fixtures_dir / "paper_trades.csv")["brier"]
    capsys.readouterr()
    for level in ("sample", "window_traded", "window_all"):
        assert a[level] == b[level]


def test_edge_report_still_reports_brier_when_no_bet_cleared(tmp_path,
                                                            fixtures_dir):
    """No bets taken is not a reason to skip scoring the model."""
    src = fixtures_dir / "paper_trades.csv"
    dst = tmp_path / "no_bets.csv"
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for r in rows:
        r["recommended_side"] = ""
        r["pnl_cents"] = ""
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    res = E.edge_report(path=Path(dst))
    assert res["bets"] == 0
    assert res["brier"]["window_traded"] is None
    assert res["brier"]["window_all"]["windows"] == 54
    assert res["brier"]["sample"]["samples"] == 204
