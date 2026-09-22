"""Probability recalibration — a one-parameter Platt scaling in logit space.

The 30-day backtest appeared to show the raw GBM model was slightly
*under-dispersed* — reality more decisive than a driftless random walk — and
this scaling was added to sharpen the odds in response.

That premise is now known to be wrong. The replay it was fit on
(`collect_samples`) carries a one-minute look-ahead: it prices a sample with a
bar whose close is only knowable a minute after the `minutes_left` label
claims (issue #2). Refit with the alignment corrected, the slope is a ≈ 0.99 —
essentially the identity. The "short-horizon momentum" this was correcting for
was the model having already seen a minute of that momentum. The transform is
kept because it is frozen, near-identity, and provably immaterial to the
headline result (`market-backtest --no-recal` moves net PnL by 0.01c), but it
should not be described as correcting a real property of BTC.

    p_cal = sigmoid(a * logit(p) + b)         a>1 sharpens toward the extremes

`a` and `b` are fit on historical backtest data by minimising log-loss, then
frozen to a JSON file and reused live. Identity (a=1, b=0) is a no-op, so the
whole path is safe before any fit exists.
"""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class Recalibrator:
    a: float = 1.0
    b: float = 0.0
    n_fit: int = 0            # samples it was fit on (0 = identity/unfitted)

    def apply(self, p: float) -> float:
        if self.a == 1.0 and self.b == 0.0:
            return p
        return _sigmoid(self.a * _logit(p) + self.b)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Recalibrator":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text()))


RECAL_PATH = Path("recalibrator.json")


def fit_recalibrator(probs: list[float], outcomes: list[int],
                     iters: int = 4000, lr: float = 0.1) -> Recalibrator:
    """
    Fit a, b by gradient descent on log-loss with feature x = logit(p).
    Pure Python, no numpy — same dependency-light spirit as the rest.
    """
    xs = [_logit(p) for p in probs]
    n = len(xs)
    if n == 0:
        return Recalibrator()
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for x, o in zip(xs, outcomes):
            s = _sigmoid(a * x + b)
            err = s - o                       # dLogLoss/dz for one sample
            ga += err * x
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    return Recalibrator(a=a, b=b, n_fit=n)
