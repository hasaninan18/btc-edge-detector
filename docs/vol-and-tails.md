# Volatility estimators and tail distributions: a null result

**Question.** The pricer assumes constant volatility over the lookback (a flat
sample standard deviation) and Gaussian log returns. BTC obeys neither. Does
relaxing either assumption improve held-out probability forecasts for the
15-minute contract?

**Answer.** No. Sixteen combinations were tested; not one beat the baseline by a
margin distinguishable from zero. The baseline stays the default.

Reproduce with:

```
python -m btc_edge vol-tails --days 30
```

---

## Setup

| | |
|---|---|
| Data | 30 days of Coinbase 1-minute BTC-USD candles (43,199 bars, no gaps) |
| Windows | 2,873 aligned 15-minute contract windows, sampled every minute |
| Split | First 70% of windows train, last 30% held out — time-ordered, never shuffled |
| Train / test | 30,165 / 12,930 samples (862 held-out windows) |
| Vol lookback | 90 bars, identical for every variant |
| Evaluation | The existing `fit_and_eval_recalibration()` harness in `btc_edge/backtest.py` |
| CI | 2,000-resample block bootstrap over whole windows, seed 20260909 |

Each variant gets **its own recalibrator**, fit on the train split only and
applied to the held-out split. No variant is ever scored on data its
recalibrator saw. Every variant is fit and scored on the same windows, and
`run_experiment` asserts that the held-out sample sets match exactly before
computing any delta — otherwise the "paired" comparison would be comparing
different averages.

### What was implemented

**Volatility** (`btc_edge/vol.py`), all behind the baseline's
`list[float] -> float` signature so they drop into `collect_samples(vol_fn=...)`:

- `ewma_vol_per_minute(closes, lam)` — RiskMetrics exponential decay, λ ∈ {0.94, 0.97}.
- `GarchVol()` — GARCH(1,1) refit by quasi-MLE on the trailing lookback at every
  sample (43,012 fits over the walk, 0 fallbacks), variance recursion re-run each
  call. Returns are rescaled to unit variance before fitting; without that, ω sits
  near 1e-9 and the simplex search never gets going.

**Tails** (`btc_edge/tails.py`): standardised Student-t, ν ∈ {4, 6, 10}, supplied
as a CDF to `prob_finish_above(cdf=...)`. Standardised is the important word —
a raw t<sub>ν</sub> has variance ν/(ν−2), so dropping one in unscaled would
inflate total variance *and* change tail shape, and any movement in the results
would be an uninterpretable mix of the two. Every CDF here has unit variance, so
the vol estimator remains the sole owner of scale and the tail changes shape only.

No scipy: the regularised incomplete beta is implemented directly (Lentz
continued fraction) and checked against published t critical values for
ν ∈ {1, 2, 5, 10, 30, 100}.

---

## Results

Held-out scores. `raw` is before recalibration, `cal` after. ΔBrier is the paired
difference against the baseline on the same windows; negative means better.

| variant | a | raw Brier | cal Brier | cal LogLoss | ΔBrier | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **stdev + normal** (baseline) | 1.162 | 0.14806 | 0.14748 | 0.44422 | — | — |
| stdev + t4 | 0.986 | 0.14743 | 0.14748 | 0.44262 | +0.00000 | [−0.00036, +0.00036] |
| stdev + t6 | 1.062 | 0.14743 | 0.14744 | 0.44273 | −0.00004 | [−0.00025, +0.00018] |
| stdev + t10 | 1.109 | 0.14764 | 0.14745 | 0.44311 | −0.00003 | [−0.00015, +0.00009] |
| ewma0.94 + normal | 1.138 | 0.14757 | 0.14726 | 0.44156 | −0.00022 | [−0.00111, +0.00065] |
| ewma0.94 + t4 | 0.966 | 0.14739 | 0.14738 | 0.44141 | −0.00010 | [−0.00115, +0.00095] |
| ewma0.94 + t6 | 1.040 | 0.14719 | 0.14729 | 0.44110 | −0.00019 | [−0.00117, +0.00076] |
| ewma0.94 + t10 | 1.085 | 0.14728 | 0.14726 | 0.44113 | −0.00022 | [−0.00116, +0.00069] |
| ewma0.97 + normal | 1.168 | 0.14774 | 0.14723 | 0.44229 | −0.00025 | [−0.00070, +0.00018] |
| ewma0.97 + t4 | 0.988 | 0.14725 | 0.14731 | 0.44154 | −0.00017 | [−0.00084, +0.00049] |
| ewma0.97 + t6 | 1.065 | 0.14718 | 0.14724 | 0.44142 | −0.00024 | [−0.00079, +0.00030] |
| ewma0.97 + t10 | 1.112 | 0.14735 | 0.14722 | 0.44159 | −0.00026 | [−0.00076, +0.00023] |
| garch11 + normal | 1.138 | 0.14742 | 0.14708 | 0.44084 | **−0.00040** | [−0.00130, +0.00048] |
| garch11 + t4 | 0.966 | 0.14723 | 0.14720 | 0.44057 | −0.00028 | [−0.00132, +0.00081] |
| garch11 + t6 | 1.040 | 0.14703 | 0.14711 | 0.44027 | −0.00037 | [−0.00136, +0.00062] |
| garch11 + t10 | 1.085 | 0.14712 | 0.14708 | 0.44032 | −0.00040 | [−0.00133, +0.00054] |

**Every confidence interval straddles zero.** The best point estimate is GARCH(1,1)
with either a Gaussian or a t(10) tail, at −0.00040 Brier — a 0.27% relative
improvement, with a CI running from −0.00130 to +0.00048.

Nothing was tuned after reading this table. λ, ν and the lookback were fixed in
advance; the grid was run once.

---

## Why the Student-t does nothing (the interesting part)

The prior going in was that fat tails would not help, because the fitted
recalibrator has a > 1 — reality is *more* decisive than the model, and fat tails
sound like the opposite of decisive. The prior was right about the conclusion and
wrong about the reason, which is worth writing down.

**A unit-variance Student-t is not uniformly fatter than a normal. It is more
peaked in the body.** Holding variance fixed, the extra mass in the tails has to
come from somewhere, and it comes from the shoulders:

| \|z\| | normal | t4 | t6 | t10 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.6915 | 0.7407 | 0.7186 | 0.7058 |
| 1.0 | 0.8413 | 0.8849 | 0.8667 | 0.8552 |
| 1.5 | 0.9332 | 0.9494 | 0.9421 | 0.9378 |
| 2.0 | 0.9772 | 0.9763 | 0.9751 | 0.9753 |
| 3.0 | 0.9987 | 0.9934 | 0.9948 | 0.9963 |

The crossover is at |z| ≈ 1.96 (t4), 1.86 (t6), 1.80 (t10). And 15-minute
contracts live below it: across the backtest samples the median |z| is **0.51**,
and **88% of samples fall inside |z| < 1.9**.

So at this horizon, switching to a Student-t is a *sharpening* transform — it
pushes probabilities away from 0.5, not toward it. That is the same job the
one-parameter Platt recalibrator already does, and the two are close to
interchangeable:

- Before recalibration, the t variants **do** improve: baseline raw Brier 0.14806
  vs 0.14743 for stdev+t4, and raw log-loss improves across the board.
- After recalibration the improvement **disappears** — 0.14748 vs 0.14748 — and
  the fitted slope drops from a = 1.162 to a = 0.986. The recalibrator simply
  stops sharpening, because the tail is now doing it.

Averaged over all twelve t variants, each compared against the normal-tail
variant that uses the *same* volatility estimator (so the vol effect is excluded
from the statistic), the tail is worth:

| | mean Brier gain |
|---|---:|
| before recalibration | **+0.00040** |
| after recalibration | **−0.00003** |

with the fitted Platt slope moving −0.110 on average. The gain does not shrink
after recalibration; it is gone.

Downstream of recalibration the tail choice is close to **unidentified**. That is
a stronger reason to leave the Gaussian in place than any single Brier number:
the alternative is not worse, it is redundant, and it costs an incomplete-beta
evaluation per quote.

A caveat on the direction of that argument: this says nothing about tails at
horizons long enough for |z| to routinely exceed 2. It is a statement about
15-minute contracts.

## Why the vol estimators do nothing

GARCH(1,1) has the best point estimate of the sixteen, and the extremes fall
where volatility clustering predicts: with the tail held Gaussian, GARCH is the
best variant and the flat stdev the worst, on both calibrated Brier and
calibrated log-loss.

The middle does not cooperate, though. The two EWMA variants swap places between
the two metrics — λ=0.97 wins on Brier (0.14723 vs 0.14726) while λ=0.94 wins on
log-loss (0.44156 vs 0.44229). A decay rate that is better by one proper scoring
rule and worse by another, on the same held-out windows, is what noise looks
like. That reading is the one the confidence intervals independently support:
the whole spread from worst to best variant is 0.00040 Brier, and the CI on that
best variant is nearly seven times as wide.

Two reasons to be more sceptical of GARCH here than the table alone suggests:

1. **Weak identification.** An 89-return lookback is very little data for
   GARCH(1,1); α and β trade off against each other along a ridge. Over 1,078
   fits sampled every 40 bars across the 30 days, all converged, with median
   α = 0.086 and β = 0.790 — plausible values. But **33.6% of windows landed at
   persistence α + β > 0.99**, effectively IGARCH, which is the signature of a
   likelihood that is nearly flat along the ridge rather than of a genuinely
   near-unit-root month. The one-step-ahead conditional variance is stable
   enough to use; the individual coefficients are not estimates of anything and
   should not be quoted as such.
2. **Cost.** It is ~1 ms per quote against a few microseconds for the baseline,
   for an effect the data cannot confirm exists.

An earlier version of the fit capped α at 0.5 and the optimiser sat on the cap in
window after window — the "estimate" was reporting the constraint, not the data.
Stationarity (α + β < 1) is now the only bound, no fit comes near it from the α
side (0.00% of the 1,078 above α = 0.99), and
`test_garch_alpha_is_not_pinned_to_a_hand_picked_ceiling` guards the regression.

---

## Threats to this conclusion

- **Multiple comparisons.** Fifteen alternatives against one baseline on a single
  held-out set. At a nominal 95% level roughly one spurious exclusion of zero
  would be expected by chance. None occurred, so this cuts the harmless way here
  — but had one row come up "significant", it would have been a hypothesis, not a
  finding.
- **One 30-day regime.** A single contiguous month of BTC. A different
  volatility regime could plausibly move the vol-estimator ordering; it is far
  less likely to move the tail conclusion, which follows from where |z| sits
  rather than from this particular month.
- **GARCH refit cadence.** Coefficients are refit at every sample here
  (`refit_every=1`). A slower cadence would be cheaper and slightly staler; it was
  not tested, because the estimator did not earn the follow-up.
- **The recalibrator absorbs a lot.** Every conclusion above is *conditional on
  recalibration*. Without it, both the t tails and the alternative vol estimators
  look like modest improvements. If the recalibrator were ever dropped from the
  live path, this experiment would need re-reading.

## What changed in the code

Additive only — the default path through `prob_finish_above` and
`collect_samples` is unchanged, and the golden-master test confirms it:

- `btc_edge/vol.py` (new) — EWMA and GARCH(1,1) estimators.
- `btc_edge/tails.py` (new) — standardised Student-t, incomplete beta.
- `btc_edge/experiments.py` (new) — the grid, the paired block bootstrap, the report.
- `btc_edge/model.py` — `prob_finish_above` gains a `cdf=` argument, defaulting to
  the normal.
- `btc_edge/backtest.py` — `collect_samples` gains `vol_fn=` / `prob_fn=` hooks,
  both defaulting to the baseline; `RecalEval` now also carries the held-out
  samples and their calibrated probabilities, which is what makes a bootstrap
  possible at all.
- `btc_edge/cli.py` — `python -m btc_edge vol-tails`.
