"""
BTC 15-min binary contract edge detector.

Phase 1: pull recent BTC 1-min candles from Coinbase's public API
Phase 2: compute a fair-value probability that BTC will finish above a strike
         at expiry, using a GBM (log-normal) model with realized volatility
Phase 3: paper-trade logger — compare model prob to market quote and
         decide whether there's edge
Phase 4: scheduler — sample once a minute across a 15-min contract window
Phase 5: settlement — fill in outcomes/PnL once windows expire
Phase 6: backtest — replay the model over historical candles so you can
         measure calibration without waiting weeks for live samples

No auth needed. No money at risk. Run it, log results, evaluate after
200+ observations before considering real money.

CLI:
    python btc_prediction_edge.py once     --strike 65181.56 --minutes-left 2.5 \
                                           --up 4.7 --down 95.4
    python btc_prediction_edge.py watch    [--strike K] [--prompt-quotes]
    python btc_prediction_edge.py fill
    python btc_prediction_edge.py backtest --days 7
"""

import argparse
import csv
from bisect import bisect_left, bisect_right
import math
import os
import ssl
import sys
import time
from dataclasses import dataclass, asdict, fields as dataclass_fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import stdev
from typing import Callable, Optional

import urllib.request
import json


# ---------- Phase 1: data ----------

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
USER_AGENT = "edge-detector/0.2"

def _ssl_context() -> ssl.SSLContext:
    """
    Framework Python on macOS ships without root certs until you run
    "Install Certificates.command". Prefer certifi's bundle when it's there.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_HINT = (
    "TLS certificate verification failed. This Python has no root certs.\n"
    '  Fix: run "/Applications/Python 3.13/Install Certificates.command"\n'
    "  or:  python3 -m pip install certifi"
)


def _get_json(url: str, retries: int = 3):
    """GET with a couple of retries — Coinbase rate-limits public endpoints."""
    last_err = None
    ctx = _ssl_context()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                raise RuntimeError(_SSL_HINT) from e
            last_err = e
            time.sleep(1.0 + attempt)
        except Exception as e:   # noqa: BLE001 - network flakiness, retry and move on
            last_err = e
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"request failed after {retries} tries: {url}") from last_err


def _rows_from_raw(raw) -> list[dict]:
    # Coinbase order: [time, low, high, open, close, volume], newest-first
    rows = [
        {"ts": int(r[0]), "low": r[1], "high": r[2],
         "open": r[3], "close": r[4], "volume": r[5]}
        for r in raw
    ]
    rows.sort(key=lambda r: r["ts"])
    return rows


def fetch_recent_1min_candles(minutes: int = 90) -> list[dict]:
    """
    Pull the last `minutes` of 1-minute BTC-USD candles from Coinbase.
    Returns list of dicts sorted oldest -> newest.
    Each row: {ts, open, high, low, close, volume}
    """
    # Coinbase returns up to 300 candles per call
    minutes = min(minutes, 300)
    end = datetime.now(timezone.utc)
    start = end.timestamp() - minutes * 60
    url = (
        f"{COINBASE_CANDLES}?granularity=60"
        f"&start={datetime.fromtimestamp(start, tz=timezone.utc).isoformat()}"
        f"&end={end.isoformat()}"
    )
    return _rows_from_raw(_get_json(url))


def fetch_candle_range(start_ts: float, end_ts: float,
                       granularity: int = 60) -> list[dict]:
    """
    Pull an arbitrarily long span of candles by paging through Coinbase's
    300-candle-per-request limit. Sorted oldest -> newest, deduped by ts.
    """
    span = granularity * 300
    out: dict[int, dict] = {}
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + span, end_ts)
        url = (
            f"{COINBASE_CANDLES}?granularity={granularity}"
            f"&start={datetime.fromtimestamp(cursor, tz=timezone.utc).isoformat()}"
            f"&end={datetime.fromtimestamp(chunk_end, tz=timezone.utc).isoformat()}"
        )
        for row in _rows_from_raw(_get_json(url)):
            out[row["ts"]] = row
        cursor = chunk_end
        time.sleep(0.35)   # stay under the public rate limit
    return [out[k] for k in sorted(out)]


def current_price() -> float:
    """Latest trade price from Coinbase ticker."""
    url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    return float(_get_json(url)["price"])


# ---------- Phase 2: fair-value model ----------

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
) -> float:
    """
    Under geometric Brownian motion with zero drift over short horizons,
    P(settle > K) = N( (ln(P/K) - 0.5 * sigma^2 * tau) / (sigma * sqrt(tau)) )
    where tau is in the same time units as sigma (minutes here).

    `tau` is the *effective* time from `effective_tau`, which accounts for the
    contract settling on a 60-second average. Pass avg_minutes=0 to recover the
    naive point-settlement model (used by tests that generate point outcomes).
    """
    if minutes_to_expiry <= 0:
        return 1.0 if price > strike else 0.0
    tau = effective_tau(minutes_to_expiry, avg_minutes)
    total_var = sigma_per_minute ** 2 * tau
    total_sd = math.sqrt(total_var)
    if total_sd == 0:
        return 1.0 if price > strike else 0.0
    z = (math.log(price / strike) - 0.5 * total_var) / total_sd
    return _norm_cdf(z)


# ---------- Phase 3: paper-trade decision ----------

@dataclass
class Quote:
    """Market quote pulled from Robinhood screen (enter by hand for now)."""
    up_cost_cents: float     # cost in cents to bet "up", pays 100 cents
    down_cost_cents: float   # cost in cents to bet "down"


# ---------- probability recalibration ----------
#
# The 30-day backtest showed the raw GBM model is slightly *under-dispersed*:
# reality is more decisive than a driftless random walk (below 0.5 it
# over-predicts, above 0.5 it under-predicts — BTC's short-horizon momentum).
# A one-parameter Platt scaling in logit space fixes it: sharpen the odds.
#
#     p_cal = sigmoid(a * logit(p) + b)         a>1 sharpens toward the extremes
#
# `a` and `b` are fit on historical backtest data by minimising log-loss, then
# frozen to a JSON file and reused live. Identity (a=1, b=0) is a no-op, so the
# whole path is safe before any fit exists.

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


@dataclass
class Decision:
    ts: str
    window_id: str                 # ISO expiry — groups all samples of one contract
    expiry_ts: float               # unix seconds, used by the settlement pass
    strike: float
    price: float
    minutes_left: float
    sigma_per_min: float
    model_prob_up: float           # what we trade on: recalibrated if a fit exists
    raw_prob_up: float             # unadjusted GBM prob, kept for auditing
    market_prob_up: Optional[float]      # implied from up cost, ignoring spread
    market_prob_down: Optional[float]
    edge_up: Optional[float]             # model_prob_up - market_prob_up
    edge_down: Optional[float]
    recommended_side: Optional[str]
    kelly_fraction: float          # of bankroll, if you took the bet
    note: str
    kalshi_ticker: Optional[str] = None   # lets the fill pass settle from source
    price_post: Optional[float] = None    # spot re-read AFTER the quote came back
    quote_lag_ms: Optional[float] = None  # spot->quote->spot round trip


MIN_EDGE = 0.05   # require 5%+ edge before considering a bet
KELLY_CAP = 0.02  # never risk more than 2% of bankroll per contract


def decide(
    price: float,
    strike: float,
    minutes_left: float,
    sigma_per_minute: float,
    quote: Optional[Quote] = None,
    expiry_ts: Optional[float] = None,
    recal: Optional[Recalibrator] = None,
    kalshi_ticker: Optional[str] = None,
    price_post: Optional[float] = None,
    quote_lag_ms: Optional[float] = None,
) -> Decision:
    raw_up = prob_finish_above(price, strike, minutes_left, sigma_per_minute)
    p_up = recal.apply(raw_up) if recal else raw_up
    p_down = 1.0 - p_up

    if expiry_ts is None:
        expiry_ts = time.time() + minutes_left * 60
    window_id = datetime.fromtimestamp(expiry_ts, tz=timezone.utc).isoformat()

    side = None
    kelly = 0.0

    if quote is None:
        # Model-only observation: still worth logging, it feeds calibration.
        market_up = market_down = edge_up = edge_down = None
        note = "no quote (model-only sample)"
    else:
        market_up = quote.up_cost_cents / 100.0
        market_down = quote.down_cost_cents / 100.0
        edge_up = p_up - market_up
        edge_down = p_down - market_down
        note = "no edge"

        # Kelly for a bet costing c that pays $1: f* = (p - c) / (1 - c)
        if edge_up >= MIN_EDGE and market_up < 1.0:
            side = "UP"
            kelly = (p_up - market_up) / (1.0 - market_up)
            note = f"model {p_up:.1%} vs market {market_up:.1%}"
        elif edge_down >= MIN_EDGE and market_down < 1.0:
            side = "DOWN"
            kelly = (p_down - market_down) / (1.0 - market_down)
            note = f"model {p_down:.1%} vs market {market_down:.1%}"

    # Fractional Kelly (0.25x) and hard cap
    kelly = max(0.0, min(kelly * 0.25, KELLY_CAP))

    return Decision(
        ts=datetime.now(timezone.utc).isoformat(),
        window_id=window_id,
        expiry_ts=expiry_ts,
        strike=strike,
        price=price,
        minutes_left=minutes_left,
        sigma_per_min=sigma_per_minute,
        model_prob_up=p_up,
        raw_prob_up=raw_up,
        market_prob_up=market_up,
        market_prob_down=market_down,
        edge_up=edge_up,
        edge_down=edge_down,
        recommended_side=side,
        kelly_fraction=kelly,
        note=note,
        kalshi_ticker=kalshi_ticker,
        price_post=price_post,
        quote_lag_ms=quote_lag_ms,
    )


# ---------- logging ----------

LOG_PATH = Path("paper_trades.csv")

OUTCOME_FIELDS = ["settle_price", "outcome_up", "pnl_cents"]
CSV_FIELDS = [f.name for f in dataclass_fields(Decision)] + OUTCOME_FIELDS


def log_decision(d: Decision, path: Path = LOG_PATH) -> None:
    row = asdict(d)
    row.update({k: "" for k in OUTCOME_FIELDS})   # filled in after expiry
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------- Phase 4: scheduler ----------

WINDOW_MINUTES = 15

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


# ---------- Kalshi capture stub (activate in the US) ----------
#
# Robinhood's 15-min BTC Up/Down contracts settle on Kalshi's market. Kalshi's
# public REST API carries live yes/no prices for those exact contracts — no auth
# needed — but it is geoblocked from some networks (it currently times out from
# here). This function is written to drop straight in when you're back on a
# network that can reach it: it returns a Quote or None, never crashes the run.
#
# Kalshi is the venue behind these contracts. The series that actually matches
# the 15-minute Up/Down product is KXBTC15M ("Bitcoin price up down",
# frequency fifteen_min) — NOT the hourly KXBTCD this file originally targeted.
#
# Two things about the live API that the earlier stub got wrong, both verified
# against real responses:
#   1. The integer-cent fields (yes_ask, no_ask, volume, open_interest) are now
#      returned as null. Live prices live in string dollar fields such as
#      "yes_ask_dollars": "0.2800". We parse those and fall back to the legacy
#      cent fields so this keeps working if they are ever repopulated.
#   2. There is no `close_ts` key; the close time is ISO-8601 in `close_time`.
#
# Kalshi also publishes the authoritative strike (`floor_strike`, echoed in
# `yes_sub_title` as "Target Price"), so we no longer have to infer it from a
# Coinbase candle at window open — a real source of error, since the true strike
# is a 60-second BRTI average and our candle close is a point sample.

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_BTC_SERIES = "KXBTC15M"


def _iso_to_ts(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _market_cents(m: dict, side: str) -> Optional[float]:
    """
    Price in cents for 'yes_ask' / 'no_ask' / 'yes_bid' / 'no_bid', reading the
    dollar-string field first and the legacy integer-cent field as a fallback.
    """
    raw = m.get(f"{side}_dollars")
    if raw not in (None, ""):
        try:
            return float(raw) * 100.0
        except (TypeError, ValueError):
            pass
    legacy = m.get(side)
    if legacy in (None, ""):
        return None
    try:
        return float(legacy)
    except (TypeError, ValueError):
        return None


@dataclass
class KalshiMarket:
    """One open 15-min BTC up/down contract, as the exchange describes it."""
    ticker: str
    strike: float           # authoritative: 60s BRTI average before window open
    close_ts: float
    quote: Optional[Quote]  # None when the book is empty (no two-sided market)
    yes_bid_cents: Optional[float]
    no_bid_cents: Optional[float]

    @property
    def vig_cents(self) -> Optional[float]:
        """How much the two asks overprice a sure thing. 100 = frictionless."""
        if self.quote is None:
            return None
        return self.quote.up_cost_cents + self.quote.down_cost_cents - 100.0


def fetch_kalshi_market(expiry_ts: Optional[float] = None,
                        series: str = KALSHI_BTC_SERIES,
                        verbose: bool = True) -> Optional[KalshiMarket]:
    """
    Fetch the open 15-min market, preferring the one closing nearest expiry_ts.
    Returns None on any network/parse failure so a capture loop degrades to
    model-only rather than dying overnight.
    """
    try:
        url = f"{KALSHI_BASE}/markets?series_ticker={series}&status=open&limit=50"
        data = _get_json(url, retries=1)
    except Exception as e:   # noqa: BLE001 - a blocked/flaky network is expected here
        if verbose:
            print(f"    (kalshi unreachable: {e}; logging model-only)")
        return None
    markets = data.get("markets", []) if isinstance(data, dict) else []
    candidates = []
    for m in markets:
        ct = _iso_to_ts(m.get("close_time"))
        strike = m.get("floor_strike")
        if ct is None or strike is None:
            continue
        candidates.append((ct, float(strike), m))
    if not candidates:
        return None

    if expiry_ts is not None:
        ct, strike, m = min(candidates, key=lambda c: abs(c[0] - expiry_ts))
    else:
        ct, strike, m = min(candidates, key=lambda c: c[0])

    yes_ask = _market_cents(m, "yes_ask")
    no_ask = _market_cents(m, "no_ask")
    quote = (Quote(up_cost_cents=yes_ask, down_cost_cents=no_ask)
             if yes_ask is not None and no_ask is not None else None)
    return KalshiMarket(
        ticker=m.get("ticker", ""),
        strike=strike,
        close_ts=ct,
        quote=quote,
        yes_bid_cents=_market_cents(m, "yes_bid"),
        no_bid_cents=_market_cents(m, "no_bid"),
    )


def fetch_kalshi_settlement(ticker: str) -> Optional[bool]:
    """
    Ground-truth settlement straight from the exchange: True if the contract
    resolved Yes (price up), False if No, None if not settled yet / unknown.

    This is strictly better than reconstructing the outcome from a Coinbase
    candle, because the contract settles on a 60-second CF Benchmarks BRTI
    average — a different number from any single exchange's minute close, and
    the two genuinely disagree on near-the-money windows.
    """
    if not ticker:
        return None
    try:
        data = _get_json(f"{KALSHI_BASE}/markets/{ticker}", retries=1)
    except Exception:   # noqa: BLE001
        return None
    m = data.get("market") if isinstance(data, dict) else None
    if not isinstance(m, dict):
        return None
    result = (m.get("result") or "").strip().lower()
    if result == "yes":
        return True
    if result == "no":
        return False
    return None


def fetch_kalshi_quote(strike: float, expiry_ts: float,
                       series: str = KALSHI_BTC_SERIES) -> Optional[Quote]:
    """Back-compat shim: just the two ask prices for the nearest open market."""
    mkt = fetch_kalshi_market(expiry_ts=expiry_ts, series=series)
    return mkt.quote if mkt else None


def kalshi_quote_fn(strike: float, expiry_ts: float):
    """Wrap fetch_kalshi_quote as a ctx-taking quote source for watch_window."""
    def fn(ctx: dict) -> Optional[Quote]:
        return fetch_kalshi_quote(ctx.get("strike", strike),
                                  ctx.get("expiry_ts", expiry_ts))
    return fn


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


# ---------- Phase 5: outcome filling ----------

def settlement_price(expiry_ts: float) -> Optional[float]:
    """
    Close of the 1-min candle that the contract settles on. Returns None if the
    candle isn't published yet (Coinbase lags by a minute or two).
    """
    minute = int(expiry_ts // 60) * 60
    rows = fetch_candle_range(minute - 300, minute + 300)
    exact = [r for r in rows if r["ts"] == minute]
    if exact:
        return float(exact[0]["close"])
    before = [r for r in rows if r["ts"] < minute]
    return float(before[-1]["close"]) if before else None


def _pnl_cents(side: Optional[str], up_cost: Optional[float],
               down_cost: Optional[float], outcome_up: bool) -> Optional[float]:
    """PnL in cents for one paper contract at the logged side."""
    if not side:
        return 0.0            # we passed — flat, and that counts as a result
    if side == "UP":
        if up_cost is None:
            return None
        return (100.0 - up_cost * 100.0) if outcome_up else -up_cost * 100.0
    if down_cost is None:
        return None
    return (100.0 - down_cost * 100.0) if not outcome_up else -down_cost * 100.0


def fill_outcomes(path: Path = LOG_PATH, grace_seconds: int = 120) -> int:
    """
    Backfill settle_price / outcome_up / pnl_cents for every logged row whose
    window has closed. Idempotent — rows already filled are left alone.
    Returns the number of rows updated.
    """
    if not path.exists():
        print(f"no log at {path}")
        return 0

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0

    now = time.time()
    settle_cache: dict[int, Optional[float]] = {}
    result_cache: dict[str, Optional[bool]] = {}
    updated = 0

    for row in rows:
        if row.get("outcome_up"):
            continue
        try:
            expiry_ts = float(row["expiry_ts"])
        except (KeyError, ValueError):
            continue
        if now < expiry_ts + grace_seconds:
            continue   # not settled yet (or candle not published)

        # Ground truth first: if the row carries a Kalshi ticker, ask the venue
        # how it actually resolved. Only fall back to reconstructing the outcome
        # from a Coinbase candle when the exchange can't tell us — the two
        # disagree on near-the-money windows, because the contract settles on a
        # 60-second BRTI average rather than one exchange's minute close.
        ticker = (row.get("kalshi_ticker") or "").strip()
        outcome_up: Optional[bool] = None
        if ticker:
            if ticker not in result_cache:
                result_cache[ticker] = fetch_kalshi_settlement(ticker)
            outcome_up = result_cache[ticker]

        key = int(expiry_ts // 60)
        if key not in settle_cache:
            settle_cache[key] = settlement_price(expiry_ts)
        settle = settle_cache[key]

        if outcome_up is None:
            if settle is None:
                continue    # neither source can settle this row yet
            outcome_up = settle > float(row["strike"])
        up_cost = float(row["market_prob_up"]) if row.get("market_prob_up") else None
        down_cost = float(row["market_prob_down"]) if row.get("market_prob_down") else None
        pnl = _pnl_cents(row.get("recommended_side") or None, up_cost, down_cost, outcome_up)

        row["settle_price"] = "" if settle is None else f"{settle:.2f}"
        row["outcome_up"] = "1" if outcome_up else "0"
        row["pnl_cents"] = "" if pnl is None else f"{pnl:.2f}"
        updated += 1

    if updated:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)

    print(f"filled {updated} row(s) in {path}")
    return updated


def log_summary(path: Path = LOG_PATH) -> None:
    """Calibration + PnL over whatever has settled so far."""
    if not path.exists():
        print(f"no log at {path}")
        return
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("outcome_up")]
    if not rows:
        print("nothing settled yet")
        return
    probs = [float(r["model_prob_up"]) for r in rows]
    outcomes = [int(r["outcome_up"]) for r in rows]
    pnl = sum(float(r["pnl_cents"]) for r in rows if r.get("pnl_cents"))
    taken = sum(1 for r in rows if r.get("recommended_side"))
    print(f"\nsettled samples: {len(rows)}   bets taken: {taken}   "
          f"paper PnL: {pnl:+.1f}c")
    print(_calibration_report(probs, outcomes))


# ---------- model-vs-market edge report ----------
#
# This is the report that answers the real question, and it only becomes
# meaningful once rows carry BOTH a market quote and a settled outcome — i.e.
# once quote capture (Kalshi, back in the US) has been running. Until then it
# just says "no settled rows with quotes yet". Tested against synthetic quotes
# so the day real ones arrive, the readout is already trustworthy.

def edge_report(path: Path = LOG_PATH) -> Optional[dict]:
    """
    Over every settled sample that also had a market quote:
      - how the model's probability compared to the market's,
      - whether acting on our edge threshold actually made money,
      - a breakdown by how big the edge looked at decision time.
    """
    if not path.exists():
        print(f"no log at {path}")
        return None
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("outcome_up") and r.get("market_prob_up")]
    if not rows:
        print("no settled rows with quotes yet — capture quotes first "
              "(watch --prompt-quotes / --kalshi), then re-run")
        return None

    # Calibration of the model against realized outcomes, on quoted samples only.
    m_probs = [float(r["model_prob_up"]) for r in rows]
    outcomes = [int(r["outcome_up"]) for r in rows]

    # Did the market itself predict well? (its implied prob vs realized)
    mkt_probs = [float(r["market_prob_up"]) for r in rows]

    taken = [r for r in rows if r.get("recommended_side")]
    settled_bets = [r for r in taken if r.get("pnl_cents")]
    pnl = sum(float(r["pnl_cents"]) for r in settled_bets)
    wins = sum(1 for r in settled_bets
               if (r["recommended_side"] == "UP") == (r["outcome_up"] == "1"))

    # ---- the part that decides anything: one independent bet per window ----
    #
    # Every sample inside a 15-min window resolves on the SAME settlement, so
    # the ~45 rows a window produces are one observation wearing 45 hats. Summing
    # their PnL and treating the total as 45 independent bets understates the
    # standard error by ~sqrt(45), which is how a single lucky window turns into
    # a "significant" edge. So the headline below aggregates to one bet per
    # window — the earliest sample that cleared the threshold, which is also the
    # only one you could realistically have acted on in real time.
    per_window: dict[str, dict] = {}
    for r in settled_bets:
        wid = r.get("window_id") or r.get("expiry_ts", "")
        prev = per_window.get(wid)
        if prev is None or r["ts"] < prev["ts"]:
            per_window[wid] = r
    win_bets = list(per_window.values())
    win_pnls = [float(r["pnl_cents"]) for r in win_bets]

    # Bucket taken bets by the edge we thought we had (window-level).
    def edge_of(r) -> float:
        return (float(r["edge_up"]) if r["recommended_side"] == "UP"
                else float(r["edge_down"]))
    buckets = {"5-10%": [], "10-20%": [], "20%+": []}
    for r in win_bets:
        e = edge_of(r)
        key = "5-10%" if e < 0.10 else "10-20%" if e < 0.20 else "20%+"
        buckets[key].append(float(r["pnl_cents"]))

    n_windows = len({r.get("window_id") for r in rows})
    print(f"\nquoted & settled: {len(rows)} samples across {n_windows} windows")
    print(f"model  Brier vs outcomes: {_brier(m_probs, outcomes):.4f}")
    print(f"market Brier vs outcomes: {_brier(mkt_probs, outcomes):.4f}  "
          f"(if the market beats the model, there's no edge to take)")

    if not settled_bets:
        print("\nno bets cleared the edge threshold yet")
        return {"n": len(rows), "pnl": 0.0, "bets": 0, "window_bets": 0,
                "mean_c": None, "ci95": None, "significant": False}

    print(f"\nsample-level (correlated — do NOT read as significance):")
    print(f"  bets {len(settled_bets)}   hit {wins / len(settled_bets):.1%}   "
          f"PnL {pnl:+,.0f}c   avg {pnl / len(settled_bets):+.2f}c")

    mean_c = sum(win_pnls) / len(win_pnls)
    win_hits = sum(1 for r in win_bets
                   if (r["recommended_side"] == "UP") == (r["outcome_up"] == "1"))
    print(f"\nWINDOW-LEVEL (independent — this is the number that counts):")
    print(f"  bets {len(win_pnls)}   hit {win_hits / len(win_pnls):.1%}   "
          f"PnL {sum(win_pnls):+,.0f}c   avg {mean_c:+.2f}c/bet")

    ci = None
    significant = False
    if len(win_pnls) >= 2:
        sd = stdev(win_pnls)
        se = sd / math.sqrt(len(win_pnls))
        lo, hi = mean_c - 1.96 * se, mean_c + 1.96 * se
        ci = (lo, hi)
        significant = lo > 0 or hi < 0
        print(f"  95% CI: [{lo:+.2f}c, {hi:+.2f}c]   (sd {sd:.1f}c, se {se:.2f}c)")
        if not significant:
            print("  -> consistent with ZERO edge; keep collecting")
            # Sizing off the observed mean is the classic trap: that mean is
            # itself noisy, so near the boundary it reports "almost done" no
            # matter how little you know. Size off the pessimistic end of the
            # CI as well — that is the number that survives the estimate moving.
            if abs(mean_c) > 1e-9:
                need = (1.96 * sd / abs(mean_c)) ** 2
                print(f"  -> if the true edge really is {mean_c:+.1f}c: "
                      f"~{need:,.0f} window-bets total "
                      f"({max(0.0, need - len(win_pnls)):,.0f} more)")
            worst = min(abs(lo), abs(hi))
            if worst > 1e-9:
                need_w = (1.96 * sd / worst) ** 2
                print(f"  -> if it is only {worst:+.1f}c (pessimistic end of the CI): "
                      f"~{need_w:,.0f} window-bets total "
                      f"({max(0.0, need_w - len(win_pnls)):,.0f} more)")
            print("  -> plan against the second number; the first assumes the "
                  "point estimate is exact, which is what got you here")
        else:
            direction = "POSITIVE" if lo > 0 else "NEGATIVE"
            print(f"  -> {direction} edge, significant at 95%")
    else:
        print("  (need >=2 window-bets before a confidence interval means anything)")

    print("\nby perceived edge at entry (window-level):")
    print("  edge band     n     PnL      avg/bet")
    for k, v in buckets.items():
        if v:
            print(f"  {k:<10} {len(v):4d} {sum(v):+8.0f}c {sum(v)/len(v):+8.2f}c")

    # Staleness diagnostic. If spot moved materially between reading the price
    # and getting the quote back, some of the "edge" above is just us holding a
    # newer clock than the market we are scoring ourselves against.
    drifts = [abs(float(r["price_post"]) - float(r["price"]))
              for r in win_bets if r.get("price_post") and r.get("price")]
    if drifts:
        lags = [float(r["quote_lag_ms"]) for r in win_bets if r.get("quote_lag_ms")]
        avg_drift = sum(drifts) / len(drifts)
        px = sum(float(r["price"]) for r in win_bets if r.get("price")) / len(win_bets)
        print(f"\nquote staleness: spot moved {avg_drift:.2f} on average "
              f"({avg_drift / px:.4%}) during a "
              f"{sum(lags) / len(lags):.0f}ms round trip"
              if lags else f"\nquote staleness: spot moved {avg_drift:.2f} on average")
        print("  (spot is read BEFORE the quote, so this biases against us)")
    else:
        print("\nquote staleness: not measured on these rows (pre-fix capture) — "
              "treat the edge above as an UPPER bound")

    return {"n": len(rows), "pnl": pnl, "bets": len(settled_bets),
            "window_bets": len(win_pnls), "mean_c": mean_c, "ci95": ci,
            "significant": significant}


# ---------- Phase 6: backtest harness ----------

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


# ---------- entry points ----------

def one_shot(strike: float, minutes_left: float,
             up_cents: Optional[float] = None, down_cents: Optional[float] = None):
    candles = fetch_recent_1min_candles(minutes=90)
    closes = [c["close"] for c in candles]
    sigma = realized_vol_per_minute(closes)
    price = current_price()

    quote = (Quote(up_cost_cents=up_cents, down_cost_cents=down_cents)
             if up_cents is not None and down_cents is not None else None)
    d = decide(price, strike, minutes_left, sigma, quote,
               recal=Recalibrator.load(RECAL_PATH))
    log_decision(d)
    print(json.dumps(asdict(d), indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_once = sub.add_parser("once", help="single decision, logged")
    p_once.add_argument("--strike", type=float, required=True)
    p_once.add_argument("--minutes-left", type=float, required=True)
    p_once.add_argument("--up", type=float, help="Up cost in cents")
    p_once.add_argument("--down", type=float, help="Down cost in cents")

    p_watch = sub.add_parser("watch", help="sample every minute across a window")
    p_watch.add_argument("--strike", type=float, default=None,
                         help="default: spot at window start")
    p_watch.add_argument("--prompt-quotes", action="store_true",
                         help="type Up/Down cents at each sample (carry-forward)")
    p_watch.add_argument("--kalshi", action="store_true",
                         help="live capture off Kalshi KXBTC15M: authoritative "
                              "strike/expiry/ticker, sub-minute polling, runs "
                              "until interrupted (US networks only)")
    p_watch.add_argument("--poll", type=int, default=20,
                         help="seconds between samples in --kalshi mode")
    p_watch.add_argument("--forever", action="store_true",
                         help="roll into each successive window")

    sub.add_parser("fill", help="backfill outcomes for expired windows")
    sub.add_parser("summary", help="calibration + PnL over settled rows")
    sub.add_parser("edge", help="model-vs-market report over quoted+settled rows")

    p_bt = sub.add_parser("backtest", help="replay the model on historical candles")
    p_bt.add_argument("--days", type=float, default=7)
    p_bt.add_argument("--vol-lookback", type=int, default=90)
    p_bt.add_argument("--sample-every", type=int, default=1)
    p_bt.add_argument("--simulate-market", action="store_true",
                      help="add a toy counterparty for PnL sensitivity")
    p_bt.add_argument("--vig", type=float, default=0.04)
    p_bt.add_argument("--recal", action="store_true",
                      help="apply the saved recalibrator during scoring")

    p_rc = sub.add_parser("recalibrate",
                          help="fit the recalibrator on history, save it")
    p_rc.add_argument("--days", type=float, default=30)
    p_rc.add_argument("--vol-lookback", type=int, default=90)
    p_rc.add_argument("--sample-every", type=int, default=1)
    p_rc.add_argument("--split", type=float, default=0.7,
                      help="train fraction (earlier windows); rest is held out")
    p_rc.add_argument("--save", action="store_true",
                      help="write recalibrator.json if it improves held-out loss")

    args = ap.parse_args(argv)

    if args.cmd == "once":
        one_shot(args.strike, args.minutes_left, args.up, args.down)
    elif args.cmd == "watch":
        if args.kalshi:
            # The exchange defines the contract, so this path ignores --strike
            # and rolls windows itself; --forever is implied.
            watch_kalshi(poll_seconds=args.poll)
        else:
            qfn = LivePrompter() if args.prompt_quotes else None
            if args.forever:
                watch_forever(quote_fn=qfn)
            else:
                watch_window(strike=args.strike, quote_fn=qfn)
                fill_outcomes()
    elif args.cmd == "fill":
        fill_outcomes()
    elif args.cmd == "summary":
        log_summary()
    elif args.cmd == "edge":
        edge_report()
    elif args.cmd == "backtest":
        candles = load_candles_cached(args.days)
        recal = Recalibrator.load(RECAL_PATH) if args.recal else None
        if args.recal and not recal.n_fit:
            print("(no fitted recalibrator found; run `recalibrate --save` first)")
        r = backtest(
            candles,
            vol_lookback=args.vol_lookback,
            sample_every=args.sample_every,
            market_fn=vig_market(args.vig) if args.simulate_market else None,
            recal=recal,
            verbose=True,
        )
        print_backtest(r)
    elif args.cmd == "recalibrate":
        candles = load_candles_cached(args.days)
        e = fit_and_eval_recalibration(
            candles, split=args.split,
            vol_lookback=args.vol_lookback, sample_every=args.sample_every,
            verbose=True,
        )
        print_recal_eval(e)
        if args.save:
            if e.calibrated.log_loss < e.raw.log_loss:
                e.recal.save(RECAL_PATH)
                print(f"\nsaved -> {RECAL_PATH} (improves held-out log-loss)")
            else:
                print("\nnot saved: recalibration did not improve held-out log-loss")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
