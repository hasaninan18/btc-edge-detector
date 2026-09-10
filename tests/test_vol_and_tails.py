"""Tests for the alternative vol estimators, the Student-t tail, and the
held-out bake-off that compares them.

The theme throughout: these are *experimental* alternatives to a baseline that
is still the default, so the tests care as much about "did this leave the
baseline alone" and "is the comparison actually paired and out-of-sample" as
about whether the maths is right.
"""
import math
import random
from functools import partial

import pytest

import btc_edge as E
from btc_edge.experiments import (
    MemoVol,
    Variant,
    block_bootstrap_brier_delta,
    build_variants,
    run_experiment,
)
from btc_edge.tails import (
    normal_cdf,
    regularized_incomplete_beta,
    standardized_t_cdf,
    student_t_cdf,
)
from btc_edge.vol import (
    Garch11,
    GarchVol,
    ewma_vol_factory,
    ewma_vol_per_minute,
    fit_garch11,
    log_returns,
)


# ------------------------------------------------------------------- fixtures

def _gbm_closes(n: int = 400, sigma: float = 0.0006, seed: int = 7) -> list[float]:
    """Constant-volatility GBM. Every estimator should agree on this one."""
    rng = random.Random(seed)
    px = 65000.0
    out = [px]
    for _ in range(n):
        px *= math.exp(rng.gauss(-0.5 * sigma ** 2, sigma))
        out.append(px)
    return out


def _clustered_closes(n: int = 400, seed: int = 11) -> list[float]:
    """Calm first half, loud second half — the case a flat stdev gets wrong."""
    rng = random.Random(seed)
    px = 65000.0
    out = [px]
    for i in range(n):
        sigma = 0.0002 if i < n // 2 else 0.0020
        px *= math.exp(rng.gauss(0.0, sigma))
        out.append(px)
    return out


def _late_burst_closes(n: int = 400, burst: int = 20, seed: int = 13) -> list[float]:
    """Calm throughout except for the final `burst` bars."""
    rng = random.Random(seed)
    px = 65000.0
    out = [px]
    for i in range(n):
        sigma = 0.0025 if i >= n - burst else 0.0002
        px *= math.exp(rng.gauss(0.0, sigma))
        out.append(px)
    return out


def _synthetic_candles(seed: int = 4242, days: int = 6,
                       sigma: float = 0.0006) -> list[dict]:
    rng = random.Random(seed)
    n = days * 1440
    t0 = 1_700_000_000 // 60 * 60 - n * 60
    px = 65000.0
    candles = []
    for i in range(n):
        o = px
        px *= math.exp(rng.gauss(-0.5 * sigma ** 2, sigma))
        candles.append({"ts": t0 + i * 60, "open": o, "high": max(o, px),
                        "low": min(o, px), "close": px, "volume": 1.0})
    return candles


# ----------------------------------------------------------------- Student-t

def test_incomplete_beta_matches_closed_forms():
    # I_x(1,1) = x, and I_x(a,b) = 1 - I_{1-x}(b,a).
    for x in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert regularized_incomplete_beta(1.0, 1.0, x) == pytest.approx(x, abs=1e-12)
        assert regularized_incomplete_beta(2.5, 3.5, x) == pytest.approx(
            1.0 - regularized_incomplete_beta(3.5, 2.5, 1.0 - x), abs=1e-12)


def test_student_t_cdf_matches_published_quantiles():
    """Two-sided 95% t critical values, i.e. CDF should read 0.975."""
    for nu, t_crit in [(1, 12.706), (2, 4.303), (5, 2.571),
                       (10, 2.228), (30, 2.042), (100, 1.984)]:
        assert student_t_cdf(t_crit, nu) == pytest.approx(0.975, abs=5e-5)


def test_student_t_is_symmetric_and_monotone():
    for nu in (3.0, 5.0, 12.0):
        assert student_t_cdf(0.0, nu) == pytest.approx(0.5)
        prev = 0.0
        for t in [-4, -2, -1, 0, 1, 2, 4]:
            cur = student_t_cdf(t, nu)
            assert cur > prev
            prev = cur
            assert cur + student_t_cdf(-t, nu) == pytest.approx(1.0, abs=1e-12)


def test_student_t_converges_to_normal_as_nu_grows():
    for z in (-2.5, -1.0, 0.5, 1.96):
        assert standardized_t_cdf(1e6)(z) == pytest.approx(normal_cdf(z), abs=1e-4)


def test_standardized_t_has_unit_variance():
    """
    The whole comparison rests on this: switching tails must not change scale.

    For a symmetric zero-mean X,  E[X^2] = INT_0^inf 2x P(|X| > x) dx  and
    P(|X| > x) = 2(1 - F(x)), so the variance is 4 * INT_0^inf x (1-F(x)) dx.
    """
    step, upper = 0.002, 40.0
    for nu in (4.0, 8.0, 30.0):
        cdf = standardized_t_cdf(nu)
        total = 0.0
        x = step / 2
        while x < upper:
            total += x * (1.0 - cdf(x)) * step
            x += step
        assert 4.0 * total == pytest.approx(1.0, rel=3e-3)


def test_standardized_t_rejects_infinite_variance():
    for nu in (0.5, 2.0):
        with pytest.raises(ValueError):
            standardized_t_cdf(nu)


def test_standardized_t_is_more_decisive_than_normal_in_the_body():
    """
    Documents the fact that drives the result: at unit variance the Student-t is
    NOT uniformly "fatter". It is more peaked in the body and only overtakes the
    normal past |z| ~ 1.8-2.0. Fifteen-minute contracts live in the body, so a t
    tail sharpens these probabilities rather than widening them.
    """
    t5 = standardized_t_cdf(5.0)
    for z in (0.25, 0.5, 1.0, 1.5):
        assert t5(z) > normal_cdf(z)
    for z in (2.5, 3.0, 4.0):
        assert t5(z) < normal_cdf(z)


# ---------------------------------------------------------------------- EWMA

def test_ewma_recovers_the_true_sigma_on_constant_vol():
    closes = _gbm_closes(n=3000, sigma=0.0006)
    assert ewma_vol_per_minute(closes) == pytest.approx(0.0006, rel=0.25)


def test_ewma_tracks_a_vol_regime_change_and_flat_stdev_does_not():
    """Recent half is the loud one, so weighting recent bars must read higher."""
    closes = _clustered_closes()
    flat = E.realized_vol_per_minute(closes)
    assert ewma_vol_per_minute(closes, lam=0.94) > flat
    assert ewma_vol_per_minute(closes, lam=0.97) > flat


def test_shorter_ewma_memory_reacts_faster_to_a_late_burst():
    """
    Only meaningful when the regime change is recent enough that the two decay
    rates actually see different data. With a burst confined to the last ~20
    bars, lam=0.94 (effective memory ~17) weights it far more than lam=0.97
    (~33). Note the ordering is NOT a general property: once both windows sit
    entirely inside the loud regime the difference is just noise, which is why
    the test above only compares each against the flat estimate.
    """
    closes = _late_burst_closes()
    flat = E.realized_vol_per_minute(closes)
    fast = ewma_vol_per_minute(closes, lam=0.94)
    slow = ewma_vol_per_minute(closes, lam=0.97)
    assert fast > slow > flat


def test_ewma_rejects_bad_lambda_and_short_input():
    closes = _gbm_closes(n=100)
    for lam in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            ewma_vol_per_minute(closes, lam=lam)
    with pytest.raises(ValueError):
        ewma_vol_per_minute(closes[:5])


def test_ewma_factory_binds_lambda():
    closes = _clustered_closes()
    fn = ewma_vol_factory(0.9)
    assert fn(closes) == ewma_vol_per_minute(closes, lam=0.9)


# --------------------------------------------------------------------- GARCH

def test_garch_recovers_parameters_from_a_simulated_garch_series():
    """Simulate a known GARCH(1,1) and check the fit lands in the neighbourhood."""
    rng = random.Random(3)
    omega, alpha, beta = 1e-8, 0.08, 0.90
    var = omega / (1 - alpha - beta)
    rets = []
    for _ in range(6000):
        r = rng.gauss(0.0, math.sqrt(var))
        rets.append(r)
        var = omega + alpha * r * r + beta * var
    fit = fit_garch11(rets)
    assert fit.converged
    assert fit.alpha == pytest.approx(alpha, abs=0.05)
    assert fit.beta == pytest.approx(beta, abs=0.10)
    assert 0.0 < fit.persistence < 1.0


def test_garch_fit_is_stationary_and_positive_on_real_shapes():
    fit = fit_garch11(log_returns(_clustered_closes()))
    assert fit.omega > 0
    assert fit.alpha >= 0 and fit.beta >= 0
    assert fit.persistence < 1.0


def test_garch_alpha_is_not_pinned_to_a_hand_picked_ceiling():
    """
    Regression test. An earlier parameterisation capped alpha at 0.5 and the fit
    sat on the cap, reporting the constraint instead of the data. Stationarity
    is the only bound that should ever bind.
    """
    fits = [fit_garch11(log_returns(_clustered_closes(seed=s))) for s in range(8)]
    assert not all(f.alpha == pytest.approx(0.5, abs=1e-6) for f in fits)
    assert all(f.persistence < 1.0 for f in fits)


def test_garch_vol_is_a_drop_in_for_the_baseline_estimator():
    closes = _clustered_closes()
    g = GarchVol()
    sigma = g(closes)
    assert sigma > 0 and math.isfinite(sigma)
    # Same order of magnitude as the flat estimate — it is a different weighting
    # of the same returns, not a different quantity.
    assert 0.1 < sigma / E.realized_vol_per_minute(closes) < 10.0


def test_garch_vol_reacts_to_the_newest_bar_between_refits():
    """
    Coefficients may be reused across calls, but the variance recursion is
    re-run every call, so a shock in the last bar must move the estimate even
    when no refit happens.
    """
    closes = _gbm_closes(n=300, sigma=0.0004)
    g = GarchVol(refit_every=10_000)      # effectively: fit once, never again
    calm = g(closes)
    shocked = g(closes + [closes[-1] * 1.02])
    assert shocked > calm


def test_garch_vol_falls_back_visibly_rather_than_silently():
    g = GarchVol()
    closes = _gbm_closes(n=200)
    g(closes)
    assert g.fallbacks == 0
    # Force the degenerate path and confirm it is both handled and counted.
    g._params = Garch11(omega=float("nan"), alpha=0.1, beta=0.8, converged=True)
    g.refit_every = 10_000
    g._calls = 1
    assert g(closes) == pytest.approx(E.realized_vol_per_minute(closes))
    assert g.fallbacks == 1


# ------------------------------------------------- the baseline is untouched

def test_default_prob_finish_above_is_unchanged_by_the_cdf_hook():
    """The `cdf` argument must be a pure addition — defaults identical to before."""
    args = (65100.0, 65000.0, 7.5, 0.0006)
    assert E.prob_finish_above(*args) == E.prob_finish_above(*args, cdf=E._norm_cdf)


def test_collect_samples_defaults_reproduce_the_baseline_walk():
    candles = _synthetic_candles()
    base = E.collect_samples(candles, avg_settle=False)
    explicit = E.collect_samples(
        candles, avg_settle=False,
        vol_fn=E.realized_vol_per_minute,
        prob_fn=partial(E.prob_finish_above, cdf=normal_cdf),
    )
    assert [s.raw_prob_up for s in base] == [s.raw_prob_up for s in explicit]


def test_swapping_the_vol_estimator_actually_changes_the_probabilities():
    candles = _synthetic_candles()
    base = E.collect_samples(candles, avg_settle=False)
    ewma = E.collect_samples(candles, avg_settle=False,
                             vol_fn=ewma_vol_factory(0.94))
    assert len(base) == len(ewma)
    assert any(a.raw_prob_up != b.raw_prob_up for a, b in zip(base, ewma))


def test_vol_and_tail_do_not_change_which_samples_exist():
    """
    The paired comparison in `run_experiment` depends on this: sample selection
    is a property of the candle grid, never of the estimator.
    """
    candles = _synthetic_candles()
    keys = []
    for vol_fn in (E.realized_vol_per_minute, ewma_vol_factory(0.94), GarchVol()):
        s = E.collect_samples(candles, avg_settle=False, vol_fn=vol_fn)
        keys.append([(x.window_ix, x.minutes_left, x.outcome_up) for x in s])
    assert keys[0] == keys[1] == keys[2]


# ----------------------------------------------------------------- bootstrap

def test_block_bootstrap_ci_brackets_the_point_estimate():
    rng = random.Random(0)
    n_win, per_win = 120, 15
    outcomes, pv, pb, windows = [], [], [], []
    i = 0
    for _ in range(n_win):
        o = rng.random() < 0.5
        idx = []
        for _ in range(per_win):
            outcomes.append(int(o))
            pv.append(rng.random())
            pb.append(rng.random())
            idx.append(i)
            i += 1
        windows.append(idx)
    point = E._brier(pv, outcomes) - E._brier(pb, outcomes)
    lo, hi = block_bootstrap_brier_delta(outcomes, pv, pb, windows,
                                         n_boot=400, seed=1)
    assert lo < point < hi


def test_block_bootstrap_is_wider_than_naive_resampling():
    """
    The reason the block exists. Samples inside a window share one outcome, so
    resampling them individually understates the spread. Build data where the
    variant is better in some windows and worse in others, and the window-block
    interval must be the wider of the two.
    """
    rng = random.Random(5)
    n_win, per_win = 80, 15
    outcomes, pv, pb, windows = [], [], [], []
    i = 0
    for w in range(n_win):
        o = int(rng.random() < 0.5)
        # Whole-window swing: the variant is confidently right or confidently
        # wrong for all 15 rows at once.
        good = rng.random() < 0.5
        for _ in range(per_win):
            outcomes.append(o)
            pv.append((0.9 if o else 0.1) if good else (0.1 if o else 0.9))
            pb.append(0.5)
            i += 1
        windows.append(list(range(i - per_win, i)))
    singles = [[j] for j in range(len(outcomes))]
    blo, bhi = block_bootstrap_brier_delta(outcomes, pv, pb, windows,
                                           n_boot=600, seed=2)
    nlo, nhi = block_bootstrap_brier_delta(outcomes, pv, pb, singles,
                                           n_boot=600, seed=2)
    assert (bhi - blo) > (nhi - nlo)


def test_block_bootstrap_is_reproducible_under_a_fixed_seed():
    outcomes = [1, 0] * 30
    pv = [0.6, 0.4] * 30
    pb = [0.5] * 60
    windows = [list(range(i, i + 6)) for i in range(0, 60, 6)]
    a = block_bootstrap_brier_delta(outcomes, pv, pb, windows, n_boot=200, seed=99)
    b = block_bootstrap_brier_delta(outcomes, pv, pb, windows, n_boot=200, seed=99)
    assert a == b


# ------------------------------------------------------------------ MemoVol

def test_memo_vol_caches_per_window_and_returns_identical_values():
    calls = []

    def fn(closes):
        calls.append(len(closes))
        return E.realized_vol_per_minute(closes)

    memo = MemoVol(fn, "probe")
    w1, w2 = _gbm_closes(n=100)[:90], _gbm_closes(n=100, seed=8)[:90]
    assert memo(w1) == memo(w1)
    memo(w2)
    assert len(calls) == 2
    assert memo.hits == 1 and memo.misses == 2


# --------------------------------------------------------------- experiment

def test_build_variants_puts_the_baseline_first_and_covers_the_grid():
    vs = build_variants(lambdas=(0.94, 0.97), nus=(4.0, 6.0, 10.0))
    assert len(vs) == 4 * 4
    assert vs[0].vol_name == "stdev" and vs[0].tail_name == "normal"
    assert {v.vol_name for v in vs} == {"stdev", "ewma0.94", "ewma0.97", "garch11"}
    assert {v.tail_name for v in vs} == {"normal", "t4", "t6", "t10"}


def test_build_variants_shares_one_memo_per_vol_estimator():
    """Otherwise every tail would refit GARCH over the same windows."""
    vs = build_variants(lambdas=(), nus=(5.0,), garch=True)
    by_vol: dict[str, set[int]] = {}
    for v in vs:
        by_vol.setdefault(v.vol_name, set()).add(id(v.vol_fn))
    assert all(len(ids) == 1 for ids in by_vol.values())


def test_run_experiment_scores_only_held_out_windows():
    candles = _synthetic_candles(days=4)
    variants = build_variants(lambdas=(0.94,), nus=(5.0,), garch=False)
    rep = run_experiment(candles, days=4, variants=variants,
                         n_boot=100, verbose=False)
    assert len(rep.results) == 4
    assert rep.test_samples > 0
    assert rep.train_samples > rep.test_samples          # 70/30 split
    # ~30% of windows held out, allowing for boundary rounding.
    total = rep.train_samples + rep.test_samples
    assert 0.25 < rep.test_samples / total < 0.35


def test_run_experiment_reports_the_baseline_with_no_delta():
    candles = _synthetic_candles(days=4)
    rep = run_experiment(candles, days=4,
                         variants=build_variants(lambdas=(), nus=(5.0,), garch=False),
                         n_boot=100, verbose=False)
    base = rep.results[0]
    assert base.variant.name == "stdev + normal"
    assert base.delta_brier is None and base.delta_ci is None
    assert all(r.delta_brier is not None for r in rep.results[1:])


def test_run_experiment_deltas_agree_with_the_reported_briers():
    candles = _synthetic_candles(days=4)
    rep = run_experiment(candles, days=4,
                         variants=build_variants(lambdas=(0.94,), nus=(), garch=False),
                         n_boot=100, verbose=False)
    base = rep.results[0]
    for r in rep.results[1:]:
        assert r.delta_brier == pytest.approx(r.cal_brier - base.cal_brier)


def test_run_experiment_accepts_a_vol_fn_that_only_changes_values():
    """A different sigma is fine — it must not change which samples exist."""
    candles = _synthetic_candles(days=4)
    good = build_variants(lambdas=(), nus=(), garch=False)[0]
    other = Variant(vol_name="short-lookback", tail_name="normal",
                    vol_fn=lambda closes: E.realized_vol_per_minute(closes[-40:]),
                    prob_fn=good.prob_fn)
    rep = run_experiment(candles, days=4, variants=[good, other],
                         n_boot=50, verbose=False)
    assert rep.results[1].delta_brier is not None


def test_run_experiment_rejects_variants_with_mismatched_test_sets(monkeypatch):
    """
    The guard that keeps the pairing honest. If a variant ever scored a
    different set of held-out samples, its Brier would be an average over
    different windows and the delta would be meaningless. The comparison must
    refuse rather than quietly produce a number.
    """
    import btc_edge.experiments as X

    candles = _synthetic_candles(days=4)
    variants = build_variants(lambdas=(), nus=(5.0,), garch=False)
    real = X.fit_and_eval_recalibration
    seen = {"n": 0}

    def flaky(*a, **kw):
        e = real(*a, **kw)
        seen["n"] += 1
        if seen["n"] == 2:               # drop a sample from the second variant
            e.test = e.test[1:]
            e.calibrated_probs = e.calibrated_probs[1:]
        return e

    monkeypatch.setattr(X, "fit_and_eval_recalibration", flaky)
    with pytest.raises(AssertionError, match="different held-out sample set"):
        X.run_experiment(candles, days=4, variants=variants,
                         n_boot=50, verbose=False)


def test_recal_eval_exposes_the_test_set_it_scored():
    candles = _synthetic_candles(days=4)
    e = E.fit_and_eval_recalibration(candles, split=0.7, avg_settle=False)
    assert len(e.test) == e.test_samples
    assert len(e.calibrated_probs) == e.test_samples
    assert e.calibrated.brier == pytest.approx(
        E._brier(e.calibrated_probs, [s.outcome_up for s in e.test]))
    # Held-out means held out: no train window may appear in the test set.
    assert min(s.window_ix for s in e.test) > 0
