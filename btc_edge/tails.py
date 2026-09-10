"""Innovation distributions for the finish-above probability.

The baseline model assumes Gaussian log returns. BTC log returns at the
1-minute scale are visibly leptokurtic, so this module supplies a standardised
Student-t as an alternative tail, behind the same `float -> float` CDF
interface `prob_finish_above` consumes.

"Standardised" matters. A raw t_nu has variance nu/(nu-2), so dropping one into
the pricing formula in place of a standard normal would quietly *inflate* total
variance as well as fatten the tails, and any change in the numbers would be a
mix of the two effects. Every CDF here has unit variance, so switching tails
changes the SHAPE only and the vol estimator remains the sole owner of scale.

No scipy — the project runs on the standard library, so the regularised
incomplete beta is implemented here (Lentz continued fraction, the textbook
recipe). It is accurate to ~1e-13, well past anything that matters at 1-cent
contract granularity.
"""
import math
from typing import Callable

_FPMIN = 1e-300
_EPS = 3e-14
_MAXIT = 300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta, evaluated by Lentz's method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b). Uses the reflection I_x(a,b) = 1 - I_{1-x}(b,a) for stability."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, nu: float) -> float:
    """CDF of a raw Student-t with `nu` degrees of freedom (variance nu/(nu-2))."""
    if nu <= 0:
        raise ValueError("degrees of freedom must be positive")
    if t == 0.0:
        return 0.5
    x = nu / (nu + t * t)
    tail = 0.5 * regularized_incomplete_beta(nu / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def normal_cdf(z: float) -> float:
    """Standard normal CDF. Same function the baseline model has always used."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def standardized_t_cdf(nu: float) -> Callable[[float], float]:
    """
    CDF of a Student-t rescaled to unit variance, as a drop-in for `normal_cdf`.

    If T ~ t_nu then T / sqrt(nu/(nu-2)) has variance 1, so
        P(Z <= z) = P(T <= z * sqrt(nu/(nu-2))).
    Requires nu > 2; below that the t has no finite variance and the whole
    "same scale, different shape" comparison stops meaning anything.
    """
    if nu <= 2.0:
        raise ValueError("nu must exceed 2 for a unit-variance Student-t")
    scale = math.sqrt(nu / (nu - 2.0))

    def cdf(z: float) -> float:
        return student_t_cdf(z * scale, nu)

    cdf.__name__ = f"t{nu:g}_cdf"
    return cdf
