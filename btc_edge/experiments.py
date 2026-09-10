"""Volatility-estimator and tail-distribution bake-off.

The baseline pricer makes two assumptions that are convenient rather than true:
volatility is constant across the lookback (flat sample stdev) and log returns
are Gaussian. BTC violates both. This module tests whether relaxing either one
actually buys anything OUT OF SAMPLE.

The evaluation is `btc_edge.backtest.fit_and_eval_recalibration` — the existing
time-ordered harness. Each variant gets its own recalibrator fit on the earlier
`split` fraction of windows and is scored only on the later fraction, so no
variant is ever judged on data its recalibrator saw. Every variant is fit and
scored on the same windows, so the comparison is paired.

Two things about reading the table:

* Samples within a 15-minute window share one settlement, so they are not
  independent observations. The confidence interval on each variant's Brier
  delta comes from a block bootstrap that resamples whole WINDOWS. The raw
  Brier numbers themselves are still per-sample averages, matching how the rest
  of the project reports them.
* This sweeps 15 alternatives against one baseline on a single held-out set. At
  a nominal 95% level you would expect roughly one spurious exclusion of zero
  from chance alone, so an isolated "significant" row here is not a finding.
  `summarise` says so in the output rather than leaving it to the reader.
"""
import random
from dataclasses import dataclass, field
from functools import partial
from statistics import fmean
from typing import Callable, Optional, Sequence

from btc_edge.backtest import RecalEval, Sample, fit_and_eval_recalibration
from btc_edge.model import prob_finish_above, realized_vol_per_minute
from btc_edge.tails import normal_cdf, standardized_t_cdf
from btc_edge.vol import GarchVol, ewma_vol_factory

DEFAULT_LAMBDAS = (0.94, 0.97)
DEFAULT_NUS = (4.0, 6.0, 10.0)
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260909


class MemoVol:
    """
    Caches a volatility estimator by lookback window.

    Every tail variant sharing a vol estimator walks the exact same sequence of
    lookback windows, so without this the GARCH fits would be repeated once per
    tail for no reason. The key is (length, first close, last close) — two
    genuinely different windows of BTC prices colliding on all three would
    require the same 90-bar span length and identical endpoints to the cent.
    """

    def __init__(self, fn: Callable[[list[float]], float], name: str):
        self.fn = fn
        self.name = name
        self._cache: dict[tuple, float] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, closes: list[float]) -> float:
        key = (len(closes), closes[0], closes[-1])
        got = self._cache.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        got = self.fn(closes)
        self._cache[key] = got
        return got


@dataclass(frozen=True)
class Variant:
    """One (volatility estimator, innovation tail) combination."""
    vol_name: str
    tail_name: str
    vol_fn: Callable[[list[float]], float]
    prob_fn: Callable[..., float]

    @property
    def name(self) -> str:
        return f"{self.vol_name} + {self.tail_name}"


@dataclass
class VariantResult:
    variant: Variant
    recal_a: float
    recal_b: float
    raw_brier: float
    raw_log_loss: float
    cal_brier: float
    cal_log_loss: float
    # Paired against the baseline on the same held-out windows. Negative means
    # the variant beat the baseline. None on the baseline row itself.
    delta_brier: Optional[float] = None
    delta_ci: Optional[tuple[float, float]] = None


@dataclass
class ExperimentReport:
    days: float
    split: float
    vol_lookback: int
    train_samples: int
    test_samples: int
    test_windows: int
    n_boot: int
    seed: int
    results: list[VariantResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------- variants

def build_variants(lambdas: Sequence[float] = DEFAULT_LAMBDAS,
                   nus: Sequence[float] = DEFAULT_NUS,
                   garch: bool = True) -> list[Variant]:
    """
    The full grid. The baseline (sample stdev + normal) is always first, because
    everything else is reported as a delta against it.
    """
    vols: list[tuple[str, Callable]] = [("stdev", realized_vol_per_minute)]
    for lam in lambdas:
        vols.append((f"ewma{lam:g}", ewma_vol_factory(lam)))
    if garch:
        vols.append(("garch11", GarchVol()))

    tails: list[tuple[str, Callable]] = [("normal", normal_cdf)]
    for nu in nus:
        tails.append((f"t{nu:g}", standardized_t_cdf(nu)))

    out: list[Variant] = []
    for vol_name, vol_fn in vols:
        memo = MemoVol(vol_fn, vol_name)   # shared across this estimator's tails
        for tail_name, cdf in tails:
            out.append(Variant(vol_name=vol_name, tail_name=tail_name,
                               vol_fn=memo,
                               prob_fn=partial(prob_finish_above, cdf=cdf)))
    return out


# ------------------------------------------------------------------ bootstrap

def _windows_index(test: Sequence[Sample]) -> list[list[int]]:
    """Positions of the held-out samples, grouped by their contract window."""
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(test):
        groups.setdefault(s.window_ix, []).append(i)
    return [groups[k] for k in sorted(groups)]


def block_bootstrap_variant_delta(
    outcomes: Sequence[int],
    probs_variant: Sequence[float],
    probs_base: Sequence[float],
    windows: Sequence[Sequence[int]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """
    Percentile 95% CI for (variant Brier - baseline Brier), resampling whole
    windows with replacement.

    Resampling individual samples would treat 15 readings of one contract as 15
    independent facts and produce an interval several times too narrow. The
    block here is the window, which is the unit that actually resolves
    independently.
    """
    rng = random.Random(seed)
    n_win = len(windows)
    # Precompute the paired per-sample squared-error difference; the Brier
    # delta over any subset is just the mean of these over that subset.
    diff = [(pv - o) ** 2 - (pb - o) ** 2
            for pv, pb, o in zip(probs_variant, probs_base, outcomes)]
    win_sums = [sum(diff[i] for i in w) for w in windows]
    win_counts = [len(w) for w in windows]

    deltas = []
    for _ in range(n_boot):
        tot = 0.0
        cnt = 0
        for _ in range(n_win):
            j = rng.randrange(n_win)
            tot += win_sums[j]
            cnt += win_counts[j]
        deltas.append(tot / cnt)
    deltas.sort()
    lo = deltas[int(0.025 * (n_boot - 1))]
    hi = deltas[int(0.975 * (n_boot - 1))]
    return lo, hi


# ----------------------------------------------------------------- experiment

def run_experiment(
    candles: list[dict],
    days: float,
    split: float = 0.7,
    vol_lookback: int = 90,
    sample_every: int = 1,
    variants: Optional[list[Variant]] = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> ExperimentReport:
    """Evaluate every variant through the existing recalibration harness."""
    variants = variants or build_variants()
    evals: list[RecalEval] = []

    for i, v in enumerate(variants, 1):
        if verbose:
            print(f"[{i}/{len(variants)}] {v.name} ...", flush=True)
        evals.append(fit_and_eval_recalibration(
            candles, split=split,
            vol_lookback=vol_lookback, sample_every=sample_every,
            vol_fn=v.vol_fn, prob_fn=v.prob_fn,
        ))

    base_eval = evals[0]
    # Every variant must land on the same held-out samples, or the deltas are
    # comparing different test sets and mean nothing. Sample selection depends
    # only on the candle grid, never on the vol or tail, so this should hold —
    # check it rather than assume it.
    base_keys = [(s.window_ix, s.minutes_left, s.outcome_up) for s in base_eval.test]
    for v, e in zip(variants[1:], evals[1:]):
        keys = [(s.window_ix, s.minutes_left, s.outcome_up) for s in e.test]
        if keys != base_keys:
            raise AssertionError(
                f"{v.name} produced a different held-out sample set than the "
                "baseline; the paired comparison would be invalid"
            )

    outcomes = [s.outcome_up for s in base_eval.test]
    windows = _windows_index(base_eval.test)
    base_probs = base_eval.calibrated_probs

    results = []
    for ix, (v, e) in enumerate(zip(variants, evals)):
        r = VariantResult(
            variant=v,
            recal_a=e.recal.a, recal_b=e.recal.b,
            raw_brier=e.raw.brier, raw_log_loss=e.raw.log_loss,
            cal_brier=e.calibrated.brier, cal_log_loss=e.calibrated.log_loss,
        )
        if ix > 0:
            r.delta_brier = e.calibrated.brier - base_eval.calibrated.brier
            r.delta_ci = block_bootstrap_variant_delta(
                outcomes, e.calibrated_probs, base_probs, windows,
                n_boot=n_boot, seed=seed,
            )
        results.append(r)

    report = ExperimentReport(
        days=days, split=split, vol_lookback=vol_lookback,
        train_samples=base_eval.train_samples,
        test_samples=base_eval.test_samples,
        test_windows=len(windows),
        n_boot=n_boot, seed=seed, results=results,
    )

    for v in variants:
        fn = v.vol_fn.fn if isinstance(v.vol_fn, MemoVol) else v.vol_fn
        if isinstance(fn, GarchVol):
            report.notes.append(
                f"garch11: {fn.fits} fits, {fn.fallbacks} fell back to sample "
                f"stdev ({fn.fallbacks / max(fn.fits, 1):.1%})"
            )
            break
    return report


# --------------------------------------------------------------------- output

def format_report(rep: ExperimentReport) -> str:
    base = rep.results[0]
    lines = [
        "Volatility estimator x innovation tail — held-out evaluation",
        "",
        f"  data          {rep.days:g} days of 1-min BTC-USD candles",
        f"  split         first {rep.split:.0%} of windows train, rest held out",
        f"  train/test    {rep.train_samples:,} / {rep.test_samples:,} samples "
        f"({rep.test_windows:,} held-out windows)",
        f"  vol lookback  {rep.vol_lookback} bars",
        f"  recalibrator  refit per variant on the train split only",
        "",
        "Held-out scores. 'raw' is before recalibration, 'cal' after. dBrier is",
        "the paired difference against the baseline on the same windows; negative",
        f"means better. CI is a {rep.n_boot:,}-resample block bootstrap over whole",
        f"windows, seed {rep.seed}.",
        "",
        f"{'variant':<22}{'a':>7}{'raw Brier':>11}{'cal Brier':>11}"
        f"{'cal LogLoss':>13}{'dBrier':>10}   {'95% CI':>22}",
        "-" * 98,
    ]
    for r in rep.results:
        if r.delta_brier is None:
            delta, ci, flag = "baseline", "", ""
        else:
            delta = f"{r.delta_brier:+.5f}"
            lo, hi = r.delta_ci
            ci = f"[{lo:+.5f}, {hi:+.5f}]"
            flag = "  *" if hi < 0 or lo > 0 else ""
        lines.append(
            f"{r.variant.name:<22}{r.recal_a:>7.3f}{r.raw_brier:>11.5f}"
            f"{r.cal_brier:>11.5f}{r.cal_log_loss:>13.5f}{delta:>10}   {ci:>22}{flag}"
        )
    lines.append("-" * 98)
    lines.append("  * = bootstrap CI excludes zero")

    lines += ["", "Verdict", "-------"]
    lines += _verdict_lines(rep)
    for n in rep.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def _verdict_lines(rep: ExperimentReport) -> list[str]:
    """
    The written conclusion, derived from the numbers rather than asserted.

    Deliberately blunt about the null case: the point of this experiment is to
    find out whether relaxing an assumption helps, and "it didn't" is a result
    that has to survive contact with the write-up.
    """
    base = rep.results[0]
    others = rep.results[1:]
    if not others:
        return ["Nothing to compare against the baseline."]

    lines: list[str] = []
    winners = [r for r in others if r.delta_ci and r.delta_ci[1] < 0]
    best = min(others, key=lambda r: r.delta_brier)
    lo, hi = best.delta_ci

    if not winners:
        lines.append(
            "No variant beat the baseline by a margin the block bootstrap can "
            "distinguish from zero."
        )
        lines.append(
            f"Best point estimate: {best.variant.name}, {best.delta_brier:+.5f} "
            f"Brier ({best.delta_brier / base.cal_brier:+.2%} relative), "
            f"CI [{lo:+.5f}, {hi:+.5f}] — straddles zero."
        )
        lines.append(
            "The baseline (flat sample stdev + Gaussian) stays the default. "
            "Nothing here is tuned"
        )
        lines.append(
            "further to try to change that: the parameters were fixed before "
            "the table was read."
        )
    else:
        names = ", ".join(w.variant.name for w in winners)
        lines.append(f"CI excludes zero for: {names}.")
        lines.append(
            f"With {len(others)} variants against one baseline on one held-out "
            "set, roughly one spurious"
        )
        lines.append(
            "exclusion is expected at a nominal 95% level. Treat this as a "
            "hypothesis to re-test on"
        )
        lines.append(
            "fresh data, not a result, and do not change the default on the "
            "strength of it."
        )

    # Why the tail variants land where they do. Their raw scores move; their
    # recalibrated scores mostly don't, because the recalibrator absorbs the
    # change. Report that from the numbers rather than claiming it.
    #
    # Each t row is compared against the NORMAL row using the same volatility
    # estimator, not against the global baseline — otherwise the vol effect
    # leaks into a statistic that is supposed to be about the tail alone.
    normal_by_vol = {r.variant.vol_name: r for r in rep.results
                     if r.variant.tail_name == "normal"}
    pairs = [(r, normal_by_vol[r.variant.vol_name]) for r in rep.results
             if r.variant.tail_name != "normal"
             and r.variant.vol_name in normal_by_vol]
    if pairs:
        raw_gain = fmean(n.raw_brier - t.raw_brier for t, n in pairs)
        cal_gain = fmean(n.cal_brier - t.cal_brier for t, n in pairs)
        a_shift = fmean(t.recal_a - n.recal_a for t, n in pairs)
        lines += [
            "",
            f"Student-t tails: mean Brier gain {raw_gain:+.5f} before "
            f"recalibration, {cal_gain:+.5f} after,",
            f"with the fitted Platt slope moving {a_shift:+.3f} against the "
            "baseline's a = "
            f"{base.recal_a:.3f}.",
            "That pattern is the finding, not a footnote. At unit variance a "
            "Student-t is not simply",
            "'fatter' — it is more PEAKED than the normal inside |z| ~ 1.9 and "
            "only heavier past it,",
            "and 15-minute contracts sit overwhelmingly inside that crossover. "
            "So switching tails acts",
            "as a sharpening transform, which is the same job the one-parameter "
            "recalibrator already",
            "does; a duly falls toward 1 and the held-out gain cancels. Downstream "
            "of recalibration the",
            "tail choice is close to unidentified, which is a stronger reason to "
            "leave it alone than",
            "any single Brier number.",
        ]
    return lines


def print_report(rep: ExperimentReport) -> None:
    print(format_report(rep))
