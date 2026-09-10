"""Alternative volatility estimators, behind the baseline's interface.

`btc_edge.model.realized_vol_per_minute` is a flat sample stdev of 1-minute log
returns over the lookback: every bar in the window gets equal weight, which is
the right estimator only if volatility is constant. It isn't — BTC clusters.
This module adds two estimators that let recent bars matter more:

    ewma_vol_per_minute(closes, lam=...)   RiskMetrics exponential decay
    GarchVol()(closes)                     GARCH(1,1), quasi-MLE on the lookback

Both take `list[float]` of closes oldest-first and return a per-minute sigma, so
they are drop-in substitutes anywhere the baseline is used — including the
`vol_fn` hook on `collect_samples`.

Everything is pure standard library, matching the rest of the package. That
rules out a proper optimiser, so the GARCH fit uses a small Nelder-Mead; see
`fit_garch11` for why that is adequate here and where it is not.
"""
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Optional, Sequence

from btc_edge.model import MIN_CLOSES_FOR_VOL, realized_vol_per_minute

# RiskMetrics' daily default. At 1-minute sampling it implies an effective
# memory of 1/(1-lam) ~ 17 bars, i.e. the estimate is dominated by the last
# quarter-hour. That is a real modelling choice, not a neutral default, which
# is why the experiment sweeps it rather than assuming it.
DEFAULT_LAMBDA = 0.94

# Bars used to seed the recursion with an unconditional variance before the
# filter takes over. Seeding from the whole lookback and then filtering through
# it again would use each bar twice; seeding from the oldest bars does not.
_SEED_BARS = 20

# Ceiling on alpha + beta. Not a modelling opinion — just enough headroom below
# 1.0 that the unconditional variance omega/(1-alpha-beta) stays finite.
_MAX_PERSIST = 0.999


def log_returns(closes: Sequence[float]) -> list[float]:
    if len(closes) < MIN_CLOSES_FOR_VOL:
        raise ValueError("need at least ~20 closes for a stable estimate")
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def ewma_vol_per_minute(closes: Sequence[float],
                        lam: float = DEFAULT_LAMBDA) -> float:
    """
    Exponentially weighted volatility: var_t = lam*var_{t-1} + (1-lam)*r_t^2.

    Note this is an uncentred second moment — the mean return is assumed zero,
    which at a 1-minute horizon is a far better assumption than trying to
    estimate a drift from 90 bars of noise. The baseline's `stdev` subtracts a
    sample mean; the difference is negligible at this scale but it is a real
    difference, so it is stated rather than papered over.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    rets = log_returns(closes)
    var = fmean(r * r for r in rets[:_SEED_BARS])
    for r in rets[_SEED_BARS:]:
        var = lam * var + (1.0 - lam) * r * r
    return math.sqrt(var)


def ewma_vol_factory(lam: float = DEFAULT_LAMBDA):
    """`ewma_vol_per_minute` with lam bound, for use as a bare `vol_fn`."""
    def fn(closes: Sequence[float]) -> float:
        return ewma_vol_per_minute(closes, lam=lam)
    fn.__name__ = f"ewma{lam:g}_vol_per_minute"
    return fn


# ------------------------------------------------------------------ GARCH(1,1)

@dataclass(frozen=True)
class Garch11:
    """Fitted GARCH(1,1) in the units of the returns it was fit on."""
    omega: float
    alpha: float
    beta: float
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def filter_next_var(self, rets: Sequence[float]) -> float:
        """
        Run the variance recursion through `rets` and return the one-step-ahead
        conditional variance — the quantity the pricer actually wants.
        """
        var = self.omega / max(1.0 - self.persistence, 1e-12)
        for r in rets:
            var = self.omega + self.alpha * r * r + self.beta * var
        return var


def _nll(params: tuple[float, float, float], rets: Sequence[float]) -> float:
    """Gaussian quasi-log-likelihood of GARCH(1,1), negated. Lower is better."""
    omega, alpha, beta = params
    persistence = alpha + beta
    if omega <= 0 or alpha < 0 or beta < 0 or persistence >= 1.0:
        return float("inf")
    var = omega / (1.0 - persistence)
    total = 0.0
    for r in rets:
        if var <= 0:
            return float("inf")
        total += math.log(var) + r * r / var
        var = omega + alpha * r * r + beta * var
    return 0.5 * total


def _unpack(theta: Sequence[float]) -> tuple[float, float, float]:
    """
    Unconstrained R^3 -> the GARCH simplex.

    alpha and beta are squashed so that stationarity (alpha + beta < 1) holds by
    construction, and the third coordinate parameterises the *unconditional*
    variance rather than omega directly. On returns scaled to unit variance
    that puts a sensible starting point near theta = 0 in every coordinate,
    which is the difference between Nelder-Mead converging and Nelder-Mead
    wandering around in 1e-9-land.

    The only box constraint is the stationarity one. An earlier version capped
    alpha at 0.5, which sounds harmless but is not: on 1-minute BTC returns the
    fit ran straight into that ceiling and sat there, so the "estimate" was
    reporting the constraint rather than the data. `_MAX_PERSIST` below is the
    sole remaining bound and exists only to keep the recursion finite.
    """
    ta, tb, tv = theta
    alpha = _MAX_PERSIST / (1.0 + math.exp(-max(min(ta, 40.0), -40.0)))
    beta = (_MAX_PERSIST - alpha) / (1.0 + math.exp(-max(min(tb, 40.0), -40.0)))
    uncond = math.exp(max(min(tv, 20.0), -20.0))
    omega = uncond * (1.0 - alpha - beta)
    return omega, alpha, beta


def _nelder_mead(f, x0: list[float], step: float = 0.5,
                 max_iter: int = 400, tol: float = 1e-8) -> tuple[list[float], bool]:
    """Textbook Nelder-Mead. Three parameters, smooth objective — it is enough."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += step
        simplex.append(pt)
    fvals = [f(p) for p in simplex]
    converged = False
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) <= tol * (abs(fvals[0]) + tol):
            converged = True
            break
        centroid = [fmean(p[j] for p in simplex[:-1]) for j in range(n)]
        worst = simplex[-1]
        refl = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        f_refl = f(refl)
        if f_refl < fvals[0]:
            exp_pt = [centroid[j] + 2.0 * (centroid[j] - worst[j]) for j in range(n)]
            f_exp = f(exp_pt)
            simplex[-1], fvals[-1] = ((exp_pt, f_exp) if f_exp < f_refl
                                      else (refl, f_refl))
        elif f_refl < fvals[-2]:
            simplex[-1], fvals[-1] = refl, f_refl
        else:
            con = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            f_con = f(con)
            if f_con < fvals[-1]:
                simplex[-1], fvals[-1] = con, f_con
            else:
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j])
                                  for j in range(n)]
                    fvals[i] = f(simplex[i])
    else:
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
    return simplex[0], converged


def fit_garch11(rets: Sequence[float], max_iter: int = 400) -> Garch11:
    """
    Quasi-MLE fit of GARCH(1,1) to `rets`.

    Returns are rescaled to roughly unit variance before fitting and the
    parameters scaled back afterwards; raw 1-minute BTC returns are ~1e-4, so
    omega lands near 1e-9 and an unscaled simplex search is hopeless.

    Caveat that belongs in any results table this feeds: with a 90-bar lookback
    there are ~89 observations, and GARCH(1,1) is weakly identified at that
    sample size — alpha and beta trade off against each other along a ridge.
    The fit is stable enough to produce a sensible conditional variance, but
    the individual coefficients should not be read as estimates of anything.
    """
    scale = math.sqrt(fmean(r * r for r in rets)) or 1.0
    z = [r / scale for r in rets]

    def obj(theta: Sequence[float]) -> float:
        return _nll(_unpack(theta), z)

    # alpha ~ 0.10, beta ~ 0.85, unconditional variance ~ 1 on the scaled series.
    a0, b0 = 0.10, 0.85
    theta0 = [math.log(a0 / (_MAX_PERSIST - a0)),
              math.log(b0 / (_MAX_PERSIST - a0 - b0)), 0.0]
    theta, converged = _nelder_mead(obj, theta0, max_iter=max_iter)
    omega_s, alpha, beta = _unpack(theta)
    return Garch11(omega=omega_s * scale * scale, alpha=alpha, beta=beta,
                   converged=converged)


class GarchVol:
    """
    GARCH(1,1) volatility as a `list[float] -> float` callable.

    Two things are worth knowing about how this behaves inside a backtest walk:

    * The variance recursion is re-run over the supplied lookback on EVERY call,
      so the returned sigma reacts to the newest bar immediately.
    * The coefficients are refit every `refit_every` calls. A fit on an 89-bar
      lookback costs about a millisecond, so the default of 1 — refit at every
      sample, no staleness at all — is affordable even over a 30-day walk.
      Raise it if a much longer walk gets slow; at `refit_every=n` the
      coefficients are at most n samples stale, and since `collect_samples`
      walks forward in time, n=15 means one contract window.

    Neither of those looks ahead: a fit only ever sees the trailing lookback the
    caller passed, which is entirely history at that sample's timestamp.

    If a fit fails to converge or lands on a non-stationary point, the call
    falls back to the plain sample stdev and records it in `fallbacks`, so a
    silently-degenerate estimator shows up in the results instead of
    masquerading as GARCH.
    """

    def __init__(self, refit_every: int = 1, max_iter: int = 400):
        self.refit_every = max(1, refit_every)
        self.max_iter = max_iter
        self._params: Optional[Garch11] = None
        self._calls = 0
        self.fits = 0
        self.fallbacks = 0

    def __call__(self, closes: Sequence[float]) -> float:
        rets = log_returns(closes)
        if self._params is None or self._calls % self.refit_every == 0:
            fitted = fit_garch11(rets, max_iter=self.max_iter)
            self.fits += 1
            if fitted.converged and 0.0 < fitted.persistence < 1.0 and fitted.omega > 0:
                self._params = fitted
        self._calls += 1
        var = self._params.filter_next_var(rets) if self._params else -1.0
        if not var > 0 or math.isinf(var) or math.isnan(var):
            self.fallbacks += 1
            return realized_vol_per_minute(list(closes))
        return math.sqrt(var)
