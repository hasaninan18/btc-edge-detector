"""Phase 4: sample a live 15-min window once a minute (or sub-minute on Kalshi).

Three entry points:
  * watch_window  — one window, minute cadence, quote via a callback
  * watch_forever — roll straight into the next window, settling as it goes
  * watch_kalshi  — the real capture loop: contract discovered from the
                    exchange, sub-minute polling, every row carries its ticker

The ordering note inside watch_kalshi (read spot BEFORE the quote) is
load-bearing — it makes any measured edge an understatement rather than an
artefact. Do not reorder it.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from btc_edge.calibration import RECAL_PATH, Recalibrator
from btc_edge.data import (
    Quote,
    current_price,
    fetch_kalshi_market,
    fetch_kalshi_quote,
    fetch_recent_1min_candles,
)
from btc_edge.decision import Decision, decide
from btc_edge.model import WINDOW_MINUTES, prob_finish_above, realized_vol_per_minute
from btc_edge.live.fill import fill_outcomes
from btc_edge.live.paperlog import LOG_PATH, log_decision


def next_window_expiry(now: Optional[datetime] = None,
                       window_minutes: int = WINDOW_MINUTES) -> datetime:
    """Next :00/:15/:30/:45 boundary strictly in the future (UTC)."""
    now = now or datetime.now(timezone.utc)
    floor_min = (now.minute // window_minutes) * window_minutes
    boundary = now.replace(minute=floor_min, second=0, microsecond=0)
    while boundary <= now:
        boundary += timedelta(minutes=window_minutes)
    return boundary


def _sleep_until(target_ts: float) -> None:
    remaining = target_ts - time.time()
    while remaining > 0:
        time.sleep(min(remaining, 5.0))   # short naps so Ctrl-C stays responsive
        remaining = target_ts - time.time()


class LivePrompter:
    """
    Frictionless manual quote capture. The model's P(up) is printed *before* the
    prompt so you can eyeball the edge as you type. Carry-forward: hit Enter to
    reuse the last line you gave (RH's quote barely moves minute to minute), so a
    whole 15-min window is usually two or three keystrokes, not fifteen prompts.

      '47 54'  set Up=47c Down=54c
      <Enter>  reuse the last quote
      's'      skip this tick (log a model-only sample)
    """
    def __init__(self) -> None:
        self.last: Optional[Quote] = None

    def __call__(self, ctx: dict) -> Optional[Quote]:
        hint = f" [last {self.last.up_cost_cents:.0f}/{self.last.down_cost_cents:.0f}]" \
               if self.last else ""
        try:
            raw = input(f"    up/down cents{hint} (Enter=reuse, s=skip): ").strip()
        except EOFError:
            return self.last
        if raw.lower() == "s":
            return None
        if not raw:
            return self.last
        try:
            up, down = (float(x) for x in raw.replace(",", " ").split()[:2])
        except ValueError:
            print("    ! couldn't parse, reusing last")
            return self.last
        self.last = Quote(up_cost_cents=up, down_cost_cents=down)
        return self.last


def watch_window(
    strike: Optional[float] = None,
    expiry: Optional[datetime] = None,
    quote_fn: Optional[Callable[[dict], Optional[Quote]]] = None,
    vol_lookback_min: int = 90,
    path: Path = LOG_PATH,
    recal: Optional[Recalibrator] = None,
) -> list[Decision]:
    """
    Sample once a minute for the life of one 15-min contract window, logging a
    Decision each time. Returns the decisions it logged.

    strike:   contract strike. Defaults to the price at the moment we start,
              which is how these contracts are usually struck at window open.
    quote_fn: called each minute with a context dict to supply the market quote.
              Pass a `LivePrompter` to type them in, `kalshi_quote_fn(...)` to
              pull them, or omit for model-only samples.
    recal:    recalibrator applied to the model prob; loaded from RECAL_PATH by
              default so a fitted calibration is used automatically if present.
    """
    expiry = expiry or next_window_expiry()
    expiry_ts = expiry.timestamp()
    if recal is None:
        recal = Recalibrator.load(RECAL_PATH)

    candles = fetch_recent_1min_candles(minutes=vol_lookback_min)
    sigma = realized_vol_per_minute([c["close"] for c in candles])
    if strike is None:
        strike = round(current_price(), 2)

    tag = f"  recal a={recal.a:.2f}" if recal.n_fit else "  recal: none"
    print(f"window expiry {expiry.isoformat()}  strike {strike:,.2f}  "
          f"sigma/min {sigma:.6f}{tag}")

    decisions: list[Decision] = []
    # Sample at the top of each minute until expiry.
    tick = time.time()
    while True:
        minutes_left = (expiry_ts - time.time()) / 60.0
        if minutes_left <= 0:
            break

        try:
            price = current_price()
            # Refresh vol every 5 samples; it barely moves minute to minute.
            if len(decisions) % 5 == 0 and decisions:
                candles = fetch_recent_1min_candles(minutes=vol_lookback_min)
                sigma = realized_vol_per_minute([c["close"] for c in candles])
        except Exception as e:   # noqa: BLE001 - one bad tick shouldn't kill the run
            print(f"  ! tick failed: {e}")
            tick += 60
            _sleep_until(min(tick, expiry_ts))
            continue

        # Show the model's read *before* asking for the quote.
        prob_up = recal.apply(prob_finish_above(price, strike, minutes_left, sigma))
        print(f"  T-{minutes_left:5.2f}m  px {price:,.2f}  P(up) {prob_up:.1%}")
        ctx = {"price": price, "prob_up": prob_up, "minutes_left": minutes_left,
               "strike": strike, "expiry_ts": expiry_ts}
        quote = quote_fn(ctx) if quote_fn else None

        d = decide(price, strike, minutes_left, sigma, quote,
                   expiry_ts=expiry_ts, recal=recal)
        log_decision(d, path=path)
        decisions.append(d)
        if quote is not None:
            print(f"    -> {d.note}"
                  + (f"  [{d.recommended_side} k={d.kelly_fraction:.3f}]"
                     if d.recommended_side else ""))

        tick += 60
        _sleep_until(min(tick, expiry_ts))

    print(f"window closed — {len(decisions)} samples logged to {path}")
    return decisions


def watch_forever(quote_fn=None, path: Path = LOG_PATH):
    """Roll straight from one 15-min window into the next until interrupted."""
    while True:
        watch_window(quote_fn=quote_fn, path=path)
        fill_outcomes(path=path)


def watch_kalshi(path: Path = LOG_PATH, poll_seconds: int = 20,
                 vol_lookback_min: int = 90,
                 recal: Optional[Recalibrator] = None) -> None:
    """
    The real capture loop, and the one that produces the paired samples the edge
    question needs. Differs from watch_window in three ways that matter:

      * the contract is discovered from the exchange, so the strike, the expiry
        and the ticker are all authoritative rather than inferred;
      * it polls on a sub-minute cadence (the book moves far faster than once a
        minute, and a 15-min contract only offers ~15 minute-samples);
      * every row carries its ticker, so `fill` can settle from the venue.

    Runs until interrupted. Any single failure — network, empty book, a window
    with no open market — is logged and skipped, never fatal, because this is
    meant to be left running overnight.
    """
    if recal is None:
        recal = Recalibrator.load(RECAL_PATH)

    sigma: Optional[float] = None
    sigma_ts = 0.0
    seen_tickers: set[str] = set()
    n_rows = 0

    print(f"kalshi capture -> {path}   poll {poll_seconds}s   (ctrl-c to stop)")
    while True:
        try:
            # ORDERING IS LOAD-BEARING. Read spot BEFORE asking for the quote.
            #
            # The natural order (quote, then spot) silently manufactures edge:
            # the logged price would be fresher than the logged quote, so the
            # model gets to "see" a move the recorded market hasn't reacted to
            # yet. On steep deep-ITM probabilities that is worth several cents
            # a bet — easily an entire phantom edge.
            #
            # Reading spot first inverts the bias: the quote is now at least as
            # fresh as the price, so any edge we measure is understated rather
            # than invented. We also re-read spot afterwards and store both, so
            # the size of the staleness effect is measurable after the fact
            # instead of being an article of faith.
            t0 = time.time()
            price = current_price()
            mkt = fetch_kalshi_market(verbose=False)
            if mkt is None:
                print("  no open 15-min market right now; waiting")
                time.sleep(poll_seconds)
                continue
            price_post = current_price()
            quote_lag_ms = (time.time() - t0) * 1000.0

            minutes_left = (mkt.close_ts - time.time()) / 60.0
            if minutes_left <= 0:
                time.sleep(min(poll_seconds, 5))
                continue

            # Vol is a 90-minute statistic; refreshing it every poll would be
            # 3 wasted API calls a minute for a number that barely moves.
            if sigma is None or time.time() - sigma_ts > 300:
                candles = fetch_recent_1min_candles(minutes=vol_lookback_min)
                sigma = realized_vol_per_minute([c["close"] for c in candles])
                sigma_ts = time.time()

            if mkt.ticker not in seen_tickers:
                seen_tickers.add(mkt.ticker)
                print(f"\n{mkt.ticker}  strike {mkt.strike:,.2f}  "
                      f"closes {datetime.fromtimestamp(mkt.close_ts, tz=timezone.utc).strftime('%H:%M:%SZ')}"
                      f"  sigma/min {sigma:.6f}")
                fill_outcomes(path=path)   # settle the window that just ended

            d = decide(price, mkt.strike, minutes_left, sigma, mkt.quote,
                       expiry_ts=mkt.close_ts, recal=recal,
                       kalshi_ticker=mkt.ticker, price_post=price_post,
                       quote_lag_ms=quote_lag_ms)
            log_decision(d, path=path)
            n_rows += 1

            mkt_str = ("--" if mkt.quote is None
                       else f"{mkt.quote.up_cost_cents:.0f}/{mkt.quote.down_cost_cents:.0f}c"
                            f" vig {mkt.vig_cents:+.0f}c")
            flag = f"  << {d.recommended_side} k={d.kelly_fraction:.3f}" if d.recommended_side else ""
            print(f"  T-{minutes_left:5.2f}m  px {price:,.2f}  model {d.model_prob_up:5.1%}"
                  f"  mkt {mkt_str}{flag}   [{n_rows}]")

        except KeyboardInterrupt:
            print(f"\nstopped — {n_rows} rows logged to {path}")
            fill_outcomes(path=path)
            return
        except Exception as e:   # noqa: BLE001 - an overnight run must survive anything
            print(f"  ! poll failed: {e}")

        time.sleep(poll_seconds)
