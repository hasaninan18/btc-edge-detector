"""Phase 6: replay the model over historical candles to measure calibration
without waiting weeks for live samples.

`collect_samples` does the one expensive walk; `score_samples` layers scoring,
recalibration and simulated PnL on top. `fit_and_eval_recalibration` is the
time-ordered train/test harness the recalibrator (and any future vol/tail
experiment) must be judged on.
"""
import json
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from btc_edge.calibration import Recalibrator, fit_recalibrator
from btc_edge.data import fetch_candle_range
from btc_edge.metrics import _brier, _calibration_report, _log_loss
from btc_edge.model import (
    MIN_CLOSES_FOR_VOL,
    SETTLE_AVG_MINUTES,
    WINDOW_MINUTES,
    prob_finish_above,
    realized_vol_per_minute,
)
from btc_edge.decision import MIN_EDGE


@dataclass
class BacktestResult:
    windows: int
    samples: int
    brier: float
    log_loss: float
    calibration: str
    bets: int
    pnl_cents: float
    hit_rate: Optional[float]


def load_candles_cached(days: float, cache_dir: Path = Path(".candle_cache")) -> list[dict]:
    """Fetch (and cache on disk) `days` of 1-min candles ending now."""
    cache_dir.mkdir(exist_ok=True)
    end = int(time.time() // 60 * 60)
    start = end - int(days * 86400)
    cache = cache_dir / f"btc_1m_{start}_{end // 3600 * 3600}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"fetching {days} day(s) of 1-min candles (~{int(days * 1440)} bars)...")
    rows = fetch_candle_range(start, end)
    cache.write_text(json.dumps(rows))
    return rows


@dataclass
class Sample:
    raw_prob_up: float
    outcome_up: int
    minutes_left: float
    window_ix: int          # 0-based window index, for time-ordered train/test splits


def collect_samples(
    candles: list[dict],
    window_minutes: int = WINDOW_MINUTES,
    vol_lookback: int = 90,
    sample_every: int = 1,
    verbose: bool = False,
    avg_settle: bool = True,
) -> list[Sample]:
    """
    Walk historical candles once and emit a raw (unrecalibrated) model sample at
    every minute of every aligned window. Scoring, recalibration and PnL are all
    layered on top of this — the expensive replay happens exactly once.

    avg_settle=True replays the contract Kalshi actually writes: both the strike
    and the settlement are 60-second averages (of the minute *ending* at the
    window's open and close respectively), and a tie resolves Up. We only have
    1-minute bars, so the average within a bar is proxied by OHLC/4 — coarse,
    but far closer to a time-average than the single close price used before,
    which additionally sampled the strike a minute *after* the window opened.

    avg_settle=False restores point settlement on the closing prints, which is
    what synthetic GBM test data generates.
    """
    by_ts = {c["ts"]: c for c in candles}
    ts_sorted = sorted(by_ts)
    closes_sorted = [float(by_ts[t]["close"]) for t in ts_sorted]
    if len(ts_sorted) < vol_lookback + window_minutes + 2:
        raise ValueError("not enough candles to backtest")

    out: list[Sample] = []
    step = window_minutes * 60
    # Start once we have a full vol lookback behind us, on a window boundary.
    first_ok = ts_sorted[vol_lookback]
    start_ts = ((first_ok + step - 1) // step) * step

    def bar_twap(bar: dict) -> float:
        """OHLC/4 — a cheap stand-in for the mean price inside a 1-minute bar."""
        return (float(bar["open"]) + float(bar["high"])
                + float(bar["low"]) + float(bar["close"])) / 4.0

    avg_min = SETTLE_AVG_MINUTES if avg_settle else 0.0
    window_ix = -1
    for open_ts in range(start_ts, ts_sorted[-1] - step, step):
        expiry_ts = open_ts + step
        if open_ts not in by_ts or expiry_ts not in by_ts:
            continue   # gap in the data, skip the window
        if avg_settle:
            # The averaged minute is the one *ending* at each boundary.
            strike_bar = by_ts.get(open_ts - 60)
            settle_bar = by_ts.get(expiry_ts - 60)
            if strike_bar is None or settle_bar is None:
                continue
            strike = bar_twap(strike_bar)
            settle = bar_twap(settle_bar)
            outcome_up = 1 if settle >= strike else 0    # ties resolve Up
        else:
            strike = float(by_ts[open_ts]["close"])
            settle = float(by_ts[expiry_ts]["close"])
            outcome_up = 1 if settle > strike else 0
        window_ix += 1

        for offset in range(0, window_minutes, sample_every):
            t = open_ts + offset * 60
            bar = by_ts.get(t)
            if bar is None:
                continue
            # Binary-search the trailing window instead of rescanning the series.
            hi = bisect_right(ts_sorted, t)
            lo = bisect_left(ts_sorted, t - vol_lookback * 60)
            hist = closes_sorted[lo:hi]
            if len(hist) < MIN_CLOSES_FOR_VOL:
                continue
            sigma = realized_vol_per_minute(hist)
            price = float(bar["close"])
            minutes_left = (expiry_ts - t) / 60.0
            p_up = prob_finish_above(price, strike, minutes_left, sigma,
                                     avg_minutes=avg_min)
            out.append(Sample(p_up, outcome_up, minutes_left, window_ix))

        if verbose and window_ix and window_ix % 200 == 0:
            print(f"  ...{window_ix} windows, {len(out)} samples")

    if not out:
        raise ValueError("backtest produced no samples")
    return out


def score_samples(
    samples: list[Sample],
    recal: Optional[Recalibrator] = None,
    market_fn: Optional[Callable[[float, float], tuple[float, float]]] = None,
    min_edge: float = MIN_EDGE,
) -> BacktestResult:
    """Turn raw samples into a scored result, optionally recalibrated."""
    probs = [(recal.apply(s.raw_prob_up) if recal else s.raw_prob_up) for s in samples]
    outcomes = [s.outcome_up for s in samples]
    bets = wins = 0
    pnl = 0.0
    if market_fn is not None:
        for p_up, s in zip(probs, samples):
            up_cost, down_cost = market_fn(p_up, s.minutes_left)
            if p_up - up_cost >= min_edge and up_cost < 1.0:
                bets += 1
                won = s.outcome_up == 1
                pnl += (100 - up_cost * 100) if won else -up_cost * 100
                wins += int(won)
            elif (1 - p_up) - down_cost >= min_edge and down_cost < 1.0:
                bets += 1
                won = s.outcome_up == 0
                pnl += (100 - down_cost * 100) if won else -down_cost * 100
                wins += int(won)
    windows = (samples[-1].window_ix + 1) if samples else 0
    return BacktestResult(
        windows=windows,
        samples=len(probs),
        brier=_brier(probs, outcomes),
        log_loss=_log_loss(probs, outcomes),
        calibration=_calibration_report(probs, outcomes),
        bets=bets,
        pnl_cents=pnl,
        hit_rate=(wins / bets) if bets else None,
    )


def backtest(
    candles: list[dict],
    window_minutes: int = WINDOW_MINUTES,
    vol_lookback: int = 90,
    sample_every: int = 1,
    market_fn: Optional[Callable[[float, float], tuple[float, float]]] = None,
    min_edge: float = MIN_EDGE,
    recal: Optional[Recalibrator] = None,
    verbose: bool = False,
    avg_settle: bool = True,
) -> BacktestResult:
    """Collect samples and score them in one shot (back-compat convenience)."""
    samples = collect_samples(candles, window_minutes, vol_lookback,
                              sample_every, verbose=verbose,
                              avg_settle=avg_settle)
    return score_samples(samples, recal=recal, market_fn=market_fn, min_edge=min_edge)


@dataclass
class RecalEval:
    recal: Recalibrator
    train_samples: int
    test_samples: int
    raw: BacktestResult      # held-out test, no recalibration
    calibrated: BacktestResult  # held-out test, recalibrated


def fit_and_eval_recalibration(
    candles: list[dict],
    split: float = 0.7,
    **walk_kwargs,
) -> RecalEval:
    """
    Fit the recalibrator on the *earlier* `split` fraction of windows and
    evaluate on the later held-out fraction. Time-ordered so we never train on
    the future — the calibrated Brier/log-loss on the test slice is the honest
    read on whether the sharpening actually helps.
    """
    samples = collect_samples(candles, **walk_kwargs)
    n_windows = samples[-1].window_ix + 1
    cut = int(n_windows * split)
    train = [s for s in samples if s.window_ix < cut]
    test = [s for s in samples if s.window_ix >= cut]
    if not train or not test:
        raise ValueError("not enough windows to split for recalibration")

    recal = fit_recalibrator([s.raw_prob_up for s in train],
                             [s.outcome_up for s in train])
    return RecalEval(
        recal=recal,
        train_samples=len(train),
        test_samples=len(test),
        raw=score_samples(test, recal=None),
        calibrated=score_samples(test, recal=recal),
    )


def print_recal_eval(e: RecalEval) -> None:
    r = e.recal
    print(f"\nrecalibrator: a={r.a:.3f} b={r.b:+.3f}  "
          f"(fit on {e.train_samples} samples; a>1 = sharpen toward 0/1)")
    print(f"held-out test: {e.test_samples} samples\n")
    print(f"{'metric':<10}{'raw':>12}{'calibrated':>14}{'improved?':>12}")
    for name, rv, cv in [("Brier", e.raw.brier, e.calibrated.brier),
                         ("LogLoss", e.raw.log_loss, e.calibrated.log_loss)]:
        better = "yes" if cv < rv else "no"
        print(f"{name:<10}{rv:>12.4f}{cv:>14.4f}{better:>12}")
    print("\ncalibrated test calibration:")
    print(e.calibrated.calibration)


def vig_market(vig: float = 0.04):
    """
    Toy counterparty: prices the *true-ish* probability with a spread, using a
    slightly different vol estimate than ours. Only useful for sensitivity
    checks — it cannot tell you whether real quotes are beatable.
    """
    def fn(model_prob: float, minutes_left: float) -> tuple[float, float]:
        # Nudge the market toward 50/50 (market underreacts to short-dated moves)
        m = 0.5 + (model_prob - 0.5) * 0.85
        return min(m + vig / 2, 0.99), min((1 - m) + vig / 2, 0.99)
    return fn


def print_backtest(r: BacktestResult) -> None:
    print(f"\nwindows: {r.windows}   samples: {r.samples}")
    print(f"Brier:   {r.brier:.4f}   (0.25 = coin flip, lower is better)")
    print(f"LogLoss: {r.log_loss:.4f}   (0.693 = coin flip)")
    print("\ncalibration:")
    print(r.calibration)
    if r.bets:
        print(f"\nsimulated bets: {r.bets}   hit rate: {r.hit_rate:.1%}   "
              f"PnL: {r.pnl_cents:+,.0f}c   "
              f"avg: {r.pnl_cents / r.bets:+.2f}c/bet")
        print("  (simulated counterparty — sensitivity only, not evidence)")
