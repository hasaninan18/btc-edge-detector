"""Scoring primitives shared by the backtest, the paper-log summary and the
model-vs-market edge report: Brier, log-loss, and a decile calibration table.
"""
import math


def _brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _log_loss(probs: list[float], outcomes: list[int]) -> float:
    eps = 1e-6
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(max(p, eps), 1 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(probs)


def _calibration_report(probs: list[float], outcomes: list[int],
                        n_buckets: int = 10) -> str:
    """Predicted vs realized frequency per decile — the thing that matters."""
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_buckets)]
    for p, o in zip(probs, outcomes):
        idx = min(int(p * n_buckets), n_buckets - 1)
        buckets[idx].append((p, o))
    lines = ["  bucket      n    predicted    realized    gap"]
    for i, b in enumerate(buckets):
        if not b:
            continue
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        pred = sum(p for p, _ in b) / len(b)
        real = sum(o for _, o in b) / len(b)
        lines.append(f"  {lo:.1f}-{hi:.1f} {len(b):6d} {pred:11.1%} "
                     f"{real:11.1%} {real - pred:+7.1%}")
    return "\n".join(lines)
