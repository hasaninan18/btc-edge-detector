"""Kalshi history: settled KXBTC15M markets and their per-minute quote candles.

This is what turns the edge question from "wait weeks for live capture" into an
offline replay. Kalshi's public REST API serves, with no auth:

  * `/markets?series_ticker=..&status=settled&min_close_ts=..&max_close_ts=..`
    every settled contract with its `result` and authoritative `floor_strike`
    (60+ days back at 96 windows a day, as of Sept 2026);
  * `/series/{series}/markets/{ticker}/candlesticks?period_interval=1`
    one candle per minute with the yes bid and yes ask OHLC, the mean traded
    price and volume.

Both are immutable once a window has settled, so everything is cached on disk
under `.kalshi_cache/` and a second run over the same span makes no requests.

Only the parts of the response this package uses are kept, in dollars-as-floats
(0.0-1.0), with the dollar-string parsing done once here.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from btc_edge.data import KALSHI_BASE, KALSHI_BTC_SERIES, _get_json, _iso_to_ts
from btc_edge.model import WINDOW_MINUTES

CACHE_DIR = Path(".kalshi_cache")
REQUEST_PAUSE = 0.12       # seconds between candle requests; public rate limit


@dataclass(frozen=True)
class SettledMarket:
    ticker: str
    strike: float          # floor_strike: the 60s BRTI average before the open
    open_ts: int           # close_ts - 15 minutes
    close_ts: int
    outcome_up: bool       # result == "yes"


@dataclass(frozen=True)
class MarketMinute:
    """One Kalshi 1-minute candle: the state of the book at the END of the minute.

    `end_ts` closes the interval (end_ts - 60, end_ts]. Prices are in dollars
    (0.0-1.0); `yes_ask`/`yes_bid` are the closing quotes of the minute, `mean`
    the volume-weighted traded price, `volume` in contracts.
    """
    end_ts: int
    yes_ask: float
    yes_bid: float
    mean: Optional[float]
    volume: float

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def mid(self) -> float:
        return (self.yes_ask + self.yes_bid) / 2.0

    @property
    def no_ask(self) -> float:
        """Cost of a Down contract. In a binary book, no_ask == 1 - yes_bid."""
        return 1.0 - self.yes_bid

    def has_book(self, max_spread: float) -> bool:
        """A real two-sided market, not the empty 0.001/1.000 placeholder."""
        return (self.yes_ask < 1.0 and self.yes_bid > 0.001
                and self.spread <= max_spread)


# ---------------------------------------------------------------- parsing --

def _dollars(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_settled_market(m: dict,
                         window_minutes: int = WINDOW_MINUTES) -> Optional[SettledMarket]:
    """A market dict from the list endpoint -> SettledMarket, or None if it is
    unusable (unsettled, voided, no strike, unparseable close time)."""
    result = (m.get("result") or "").strip().lower()
    if result not in ("yes", "no"):
        return None
    strike = _dollars(m.get("floor_strike"))
    close_ts = _iso_to_ts(m.get("close_time"))
    ticker = m.get("ticker")
    if strike is None or close_ts is None or not ticker:
        return None
    close_i = int(round(close_ts))
    return SettledMarket(ticker=ticker, strike=strike,
                         open_ts=close_i - window_minutes * 60, close_ts=close_i,
                         outcome_up=(result == "yes"))


def parse_candle(c: dict) -> Optional[MarketMinute]:
    """A candlestick dict -> MarketMinute, or None if the quote fields are absent."""
    try:
        end_ts = int(c["end_period_ts"])
        ask = _dollars((c.get("yes_ask") or {}).get("close_dollars"))
        bid = _dollars((c.get("yes_bid") or {}).get("close_dollars"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if ask is None or bid is None:
        return None
    mean = _dollars((c.get("price") or {}).get("mean_dollars"))
    vol = _dollars(c.get("volume_fp")) or 0.0
    return MarketMinute(end_ts=end_ts, yes_ask=ask, yes_bid=bid, mean=mean, volume=vol)


# ---------------------------------------------------------------- fetching --

def _day_start(ts: float) -> int:
    return int(ts // 86400) * 86400


def fetch_settled_markets(start_ts: float, end_ts: float,
                          series: str = KALSHI_BTC_SERIES,
                          cache_dir: Path = CACHE_DIR,
                          verbose: bool = False) -> list[SettledMarket]:
    """
    Every settled market in `series` closing in [start_ts, end_ts), oldest
    first. Cached one UTC day at a time; a day is only written to the cache once
    it is fully in the past, so a partial day is re-fetched next time.
    """
    out: list[SettledMarket] = []
    seen: set[str] = set()
    day = _day_start(start_ts)
    now = time.time()
    (cache_dir / "settled").mkdir(parents=True, exist_ok=True)
    while day < end_ts:
        day_end = day + 86400
        cache = cache_dir / "settled" / f"{series}_{day}.json"
        if cache.exists():
            raw = json.loads(cache.read_text())
        else:
            raw, cursor = [], None
            while True:
                url = (f"{KALSHI_BASE}/markets?series_ticker={series}&status=settled"
                       f"&limit=200&min_close_ts={day}&max_close_ts={day_end}"
                       + (f"&cursor={cursor}" if cursor else ""))
                data = _get_json(url)
                # A 2xx body without the expected key is a schema change or an
                # error page, not an empty day. Never freeze that to disk.
                if not isinstance(data, dict) or "markets" not in data:
                    raise RuntimeError(f"unexpected settled-markets payload: {url}")
                page = data["markets"] or []
                raw.extend(page)
                cursor = data.get("cursor")
                if not cursor or not page:
                    break
                time.sleep(REQUEST_PAUSE)
            # only freeze a day that can no longer gain settlements
            if day_end + 3600 < now:
                cache.write_text(json.dumps(raw))
            if verbose:
                print(f"  settled {datetime.fromtimestamp(day, tz=timezone.utc):%Y-%m-%d}: "
                      f"{len(raw)} markets")
        for m in raw:
            sm = parse_settled_market(m)
            if sm is None or sm.ticker in seen:
                continue
            if start_ts <= sm.close_ts < end_ts:
                seen.add(sm.ticker)
                out.append(sm)
        day = day_end
    out.sort(key=lambda m: m.close_ts)
    return out


def fetch_market_minutes(market: SettledMarket,
                         series: str = KALSHI_BTC_SERIES,
                         cache_dir: Path = CACHE_DIR) -> list[MarketMinute]:
    """The 1-minute candles for one settled market, cached per ticker forever."""
    (cache_dir / "candles").mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "candles" / f"{market.ticker}.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        url = (f"{KALSHI_BASE}/series/{series}/markets/{market.ticker}/candlesticks"
               f"?start_ts={market.open_ts - 60}&end_ts={market.close_ts + 60}"
               f"&period_interval=1")
        data = _get_json(url)
        if not isinstance(data, dict) or "candlesticks" not in data:
            # Do not cache: an empty list on disk would be indistinguishable
            # from a genuinely quiet window on every future run.
            raise RuntimeError(f"unexpected candlesticks payload: {url}")
        raw = data["candlesticks"] or []
        cache.write_text(json.dumps(raw))
        time.sleep(REQUEST_PAUSE)
    minutes = [mm for mm in (parse_candle(c) for c in raw) if mm is not None]
    minutes.sort(key=lambda mm: mm.end_ts)
    return minutes


@dataclass
class MarketHistory:
    market: SettledMarket
    minutes: list[MarketMinute]


def load_history(days: float, end_ts: Optional[float] = None,
                 series: str = KALSHI_BTC_SERIES,
                 cache_dir: Path = CACHE_DIR,
                 verbose: bool = True) -> list[MarketHistory]:
    """
    `days` of settled windows ending at `end_ts` (default: the last window that
    can have settled), each with its minute candles. Oldest first.
    """
    if end_ts is None:
        # settlement lands ~5 minutes after close; stay clear of it
        end_ts = (int(time.time()) // 900) * 900 - 600
    start_ts = end_ts - days * 86400
    markets = fetch_settled_markets(start_ts, end_ts, series=series,
                                    cache_dir=cache_dir, verbose=verbose)
    if verbose:
        print(f"{len(markets)} settled {series} windows "
              f"{datetime.fromtimestamp(start_ts, tz=timezone.utc):%Y-%m-%d} -> "
              f"{datetime.fromtimestamp(end_ts, tz=timezone.utc):%Y-%m-%d}; "
              f"loading minute candles (cached after first run)...")
    out: list[MarketHistory] = []
    failed: list[str] = []
    for i, m in enumerate(markets):
        try:
            minutes = fetch_market_minutes(m, series=series, cache_dir=cache_dir)
        except Exception as e:   # noqa: BLE001 - one bad ticker must not kill 5,000
            failed.append(m.ticker)
            if verbose:
                print(f"  ! {m.ticker}: {e}; skipping")
            continue
        out.append(MarketHistory(m, minutes))
        if verbose and i and i % 200 == 0:
            print(f"  ...{i}/{len(markets)}")
    if failed and verbose:
        print(f"  {len(failed)} window(s) skipped after fetch failures "
              f"(not cached; re-run to retry)")
    return out
