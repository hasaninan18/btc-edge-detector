"""Fair-value model: realized volatility and the GBM finish-above probability.

Phase 2 of the original single file. Pure functions, no I/O. The one wrinkle
worth reading is `effective_tau` — these contracts settle on a 60-second BRTI
average, not a point price, which shortens the variance-equivalent time to
expiry. See the derivation in the docstring.
"""
import math
from statistics import stdev
from typing import Callable

# Contract window length. Lives here (rather than in the scheduler) because both
# the live loop and the backtest need it and this module has no dependencies.
WINDOW_MINUTES = 15

MIN_CLOSES_FOR_VOL = 20


def realized_vol_per_minute(closes: list[float]) -> float:
    """
    Sample standard deviation of 1-minute log returns.
    Units: per-minute (not annualized). Use directly with time-in-minutes.
    """
    if len(closes) < MIN_CLOSES_FOR_VOL:
        raise ValueError("need at least ~20 closes for a stable estimate")
    log_rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    return stdev(log_rets)


# This flat stdev assumes volatility is constant across the lookback, which BTC
# violates. `btc_edge.vol` supplies EWMA and GARCH(1,1) alternatives behind the
# same `list[float] -> float` signature; `collect_samples(vol_fn=...)` is the
# hook that swaps them in. As of the 30-day evaluation in `btc_edge.experiments`
# neither beat this one out of sample, so it remains the default.


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# These contracts do NOT settle on the price printed at the closing bell. Per
# Kalshi's own rules text for KXBTC15M:
#
#   "If the simple average of the sixty seconds of CF Benchmarks' BRTI before
#    10:00 AM is at least the simple average of the sixty seconds before
#    9:45 AM, then the market resolves to Yes."
#
# So the settled quantity is a 60-second time-average, not a point sample. That
# matters: averaging over the last minute strictly *reduces* terminal variance.
# For a driftless random walk observed at t with tau minutes left, writing
# A for the average over the final delta minutes,
#
#   Var(A - P_t) = (1/delta^2) * INT INT min(s,u) ds du  =  tau - (2/3)*delta
#
# i.e. the contract behaves like one expiring (2/3) of a minute EARLIER. At
# tau=15 that is a 2% variance haircut (noise), but at tau=2 it is 33% — and
# near expiry is exactly where the model is asked for its most confident
# numbers, so ignoring it biases every late-window probability toward 0.5.
SETTLE_AVG_MINUTES = 1.0


def effective_tau(minutes_to_expiry: float,
                  avg_minutes: float = SETTLE_AVG_MINUTES) -> float:
    """
    Variance-equivalent time to expiry for a contract settling on the average of
    the final `avg_minutes`, rather than on a point price.

    Two regimes, both derived from Var((1/d)*INT_{T-d}^{T} P_s ds - P_t):
      tau >= d : tau - 2d/3      (we are outside the averaging window)
      tau <  d : tau^3 / (3d^2)  (we are inside it; the already-realised part of
                                  the average is unknown to us at 1-min
                                  sampling, so this is an approximation that
                                  correctly collapses to 0 as tau -> 0)
    """
    if minutes_to_expiry <= 0:
        return 0.0
    d = avg_minutes
    if d <= 0:
        return minutes_to_expiry
    if minutes_to_expiry >= d:
        return minutes_to_expiry - 2.0 * d / 3.0
    return minutes_to_expiry ** 3 / (3.0 * d * d)


def prob_finish_above(
    price: float,
    strike: float,
    minutes_to_expiry: float,
    sigma_per_minute: float,
    avg_minutes: float = SETTLE_AVG_MINUTES,
    cdf: Callable[[float], float] = _norm_cdf,
) -> float:
    """
    Under geometric Brownian motion with zero drift over short horizons,
    P(settle > K) = N( (ln(P/K) - 0.5 * sigma^2 * tau) / (sigma * sqrt(tau)) )
    where tau is in the same time units as sigma (minutes here).

    `tau` is the *effective* time from `effective_tau`, which accounts for the
    contract settling on a 60-second average. Pass avg_minutes=0 to recover the
    naive point-settlement model (used by tests that generate point outcomes).

    `cdf` is the distribution of the standardised innovation. It defaults to the
    normal, and anything passed in its place MUST have unit variance — the
    scale of the move belongs to `sigma_per_minute`, and a CDF that also
    carries scale would double-count it. `btc_edge.tails.standardized_t_cdf`
    supplies fat-tailed alternatives on that contract.

    Note the -0.5*sigma^2*tau Ito term is left alone when the tail changes. It
    is the drift that makes E[S_T] = S_t under *lognormal* returns and is not
    the exact martingale correction for a t; at these horizons it is worth
    about 4e-6 in log space against a 2e-3 standard deviation, so correcting it
    would move no probability by a full basis point while making the two
    variants differ in drift as well as shape. Isolating the tail is the point.
    """
    if minutes_to_expiry <= 0:
        return 1.0 if price > strike else 0.0
    tau = effective_tau(minutes_to_expiry, avg_minutes)
    total_var = sigma_per_minute ** 2 * tau
    total_sd = math.sqrt(total_var)
    if total_sd == 0:
        return 1.0 if price > strike else 0.0
    z = (math.log(price / strike) - 0.5 * total_var) / total_sd
    return cdf(z)
