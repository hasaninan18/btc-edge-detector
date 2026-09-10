"""Scoring primitives shared by the backtest, the paper-log summary and the
model-vs-market edge report: Brier, log-loss, a decile calibration table, and
the paired model-vs-market Brier delta with a block-bootstrapped interval.
"""
import math
import random


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


def brier_delta(model_probs: list[float], market_probs: list[float],
                outcomes: list[int]) -> float:
    """Paired delta: model Brier - market Brier, scored against the SAME outcomes.

    Printing two independent Brier scores hides the pairing. Both are scored on
    one shared set of settlements, so their difference carries far less noise
    than either number does alone, and the difference is what the question
    ("does the model beat the quote?") actually asks about. Negative means the
    model won.
    """
    return _brier(model_probs, outcomes) - _brier(market_probs, outcomes)


def block_bootstrap_brier_delta(
    blocks: list[list[tuple[float, float, int]]],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for the paired Brier delta, resampling whole blocks.

    `blocks` holds one list of (model_prob, market_prob, outcome) per
    INDEPENDENT unit — a 15-minute window here. A resample draws windows with
    replacement and takes all of a drawn window's samples or none of them.

    That is the entire point. Rows inside a window resolve on the same
    settlement, so bootstrapping individual rows would treat ~45 copies of one
    observation as 45 draws and shrink the interval by roughly sqrt(45) — the
    same mistake the window-level PnL aggregation already exists to avoid.

    Exact and cheap: Brier is a mean of squared errors, so each block need only
    contribute (model SSE, market SSE, n), and a resample is a sum over blocks
    rather than a rescan of every row.
    """
    if not blocks:
        raise ValueError("block_bootstrap_brier_delta: no blocks")
    if n_resamples < 1:
        raise ValueError("block_bootstrap_brier_delta: n_resamples must be >= 1")

    parts: list[tuple[float, float, int]] = []
    for b in blocks:
        if not b:
            continue
        parts.append((sum((p - o) ** 2 for p, _, o in b),
                      sum((q - o) ** 2 for _, q, o in b),
                      len(b)))
    if not parts:
        raise ValueError("block_bootstrap_brier_delta: every block is empty")

    point = ((sum(p[0] for p in parts) - sum(p[1] for p in parts))
             / sum(p[2] for p in parts))

    rng = random.Random(seed)
    n_blocks = len(parts)
    deltas: list[float] = []
    for _ in range(n_resamples):
        sse_m = sse_k = 0.0
        n = 0
        for _ in range(n_blocks):
            a, b_, c = parts[rng.randrange(n_blocks)]
            sse_m += a
            sse_k += b_
            n += c
        deltas.append((sse_m - sse_k) / n)
    deltas.sort()

    last = n_resamples - 1
    return {
        "delta": point,
        "lo": deltas[int(round((alpha / 2) * last))],
        "hi": deltas[int(round((1 - alpha / 2) * last))],
        # Share of resamples in which the model scored better (delta < 0) — a
        # one-sided bootstrap p-value in all but name; read it as one.
        "p_model_better": sum(1 for d in deltas if d < 0) / n_resamples,
        "n_blocks": n_blocks,
        "n_resamples": n_resamples,
        "seed": seed,
    }
