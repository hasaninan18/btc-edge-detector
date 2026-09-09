"""Market data: Coinbase 1-minute candles + spot, and the Kalshi KXBTC15M book.

Phase 1 of the original file plus the Kalshi capture helpers. Everything here
is read-only public-endpoint I/O. No auth. Network failures degrade to None /
retry rather than raising into a capture loop that is meant to run overnight.
"""
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# ---------- Coinbase ----------

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


# ---------- market quote ----------

@dataclass
class Quote:
    """Market quote pulled from Robinhood screen (enter by hand for now)."""
    up_cost_cents: float     # cost in cents to bet "up", pays 100 cents
    down_cost_cents: float   # cost in cents to bet "down"


# ---------- Kalshi KXBTC15M ----------
#
# Robinhood's 15-min BTC Up/Down contracts settle on Kalshi's market. Kalshi's
# public REST API carries live yes/no prices for those exact contracts — no auth
# needed — but it is geoblocked from some networks (it currently times out from
# here). These helpers are written to drop straight in when back on a network
# that can reach it: they return a Quote / KalshiMarket / bool, or None, and
# never crash the run.
#
# The series that matches the 15-minute Up/Down product is KXBTC15M ("Bitcoin
# price up down", frequency fifteen_min) — NOT the hourly KXBTCD.
#
# Two things about the live API that an earlier stub got wrong, both verified
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
