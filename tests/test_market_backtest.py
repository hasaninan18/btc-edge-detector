"""The real-quote market backtest: fees, history parsing/caching, the
spot<->quote pairing rule, one-bet-per-window selection, and the report.

Everything here is offline. Kalshi responses are hand-built dicts in the shape
the live API returns (verified against real payloads, Sept 2026), and the
network layer is monkeypatched to fail loudly if anything tries to reach it.
"""
import json
import math
from pathlib import Path

import pytest

import btc_edge as E
from btc_edge import history as H
from btc_edge import market_backtest as MB
from btc_edge.fees import kalshi_fee_cents


# ---------------------------------------------------------------------- fees --

@pytest.mark.parametrize("price_cents, fee", [
    (50, 2),     # 0.07*0.25 = 1.75c -> 2c
    (30, 2),     # 0.07*0.21 = 1.47c -> 2c
    (10, 1),     # 0.07*0.09 = 0.63c -> 1c
    (99, 1),     # 0.07*0.0099 = 0.07c -> 1c: any fee rounds UP to a cent
    (0, 0),
    (100, 0),
])
def test_kalshi_fee_rounds_up_to_the_cent(price_cents, fee):
    assert kalshi_fee_cents(price_cents) == fee


def test_kalshi_fee_scales_with_contracts_before_rounding():
    # 100 contracts at 50c: 0.07*100*0.25 = $1.75 exactly -> 175c, not 200c.
    assert kalshi_fee_cents(50, contracts=100) == 175
    assert kalshi_fee_cents(50, contracts=0) == 0


def test_kalshi_fee_is_symmetric_in_price():
    assert kalshi_fee_cents(20) == kalshi_fee_cents(80)


# ------------------------------------------------------------------ parsing --

def _market_dict(ticker="KXBTC15M-26SEP141045-45", result="no",
                 strike=78372.39, close="2026-09-14T14:45:00Z"):
    return {"ticker": ticker, "result": result, "floor_strike": strike,
            "close_time": close, "yes_ask": None, "yes_ask_dollars": "0.0010"}


def test_parse_settled_market_reads_result_strike_and_window_bounds():
    sm = H.parse_settled_market(_market_dict())
    assert sm.ticker == "KXBTC15M-26SEP141045-45"
    assert sm.outcome_up is False
    assert sm.strike == pytest.approx(78372.39)
    assert sm.close_ts - sm.open_ts == 15 * 60
    assert sm.close_ts == int(E._iso_to_ts("2026-09-14T14:45:00Z"))
    assert H.parse_settled_market(_market_dict(result="yes")).outcome_up is True


@pytest.mark.parametrize("bad", [
    _market_dict(result=""),            # unsettled
    _market_dict(result="void"),        # voided market
    _market_dict(strike=None),
    _market_dict(close="not a time"),
    {**_market_dict(), "ticker": ""},
])
def test_parse_settled_market_drops_unusable_rows(bad):
    assert H.parse_settled_market(bad) is None


def _candle(end_ts, ask="0.5300", bid="0.5200", mean="0.5250", vol="120.5"):
    return {"end_period_ts": end_ts, "volume_fp": vol,
            "price": {"mean_dollars": mean},
            "yes_ask": {"close_dollars": ask, "open_dollars": "1.0000"},
            "yes_bid": {"close_dollars": bid, "open_dollars": "0.0010"}}


def test_parse_candle_reads_closing_quotes_in_dollars():
    mm = H.parse_candle(_candle(1000))
    assert (mm.end_ts, mm.yes_ask, mm.yes_bid) == (1000, 0.53, 0.52)
    assert mm.mean == pytest.approx(0.525)
    assert mm.volume == pytest.approx(120.5)
    assert mm.spread == pytest.approx(0.01)
    assert mm.mid == pytest.approx(0.525)
    assert mm.no_ask == pytest.approx(0.48)


def test_parse_candle_tolerates_missing_fields():
    assert H.parse_candle({"end_period_ts": 5}) is None
    assert H.parse_candle({"end_period_ts": 5, "yes_ask": {}, "yes_bid": {}}) is None
    assert H.parse_candle({"end_period_ts": 5, "yes_ask": {"close_dollars": "x"},
                           "yes_bid": {"close_dollars": "0.5"}}) is None
    mm = H.parse_candle({"end_period_ts": 5, "yes_ask": {"close_dollars": "0.6"},
                         "yes_bid": {"close_dollars": "0.5"}})
    assert mm.mean is None and mm.volume == 0.0


def test_has_book_rejects_the_empty_placeholder_and_wide_spreads():
    empty = H.parse_candle(_candle(1, ask="1.0000", bid="0.0010"))
    assert not empty.has_book(0.05)
    wide = H.parse_candle(_candle(1, ask="0.70", bid="0.40"))
    assert not wide.has_book(0.05) and wide.has_book(0.50)
    assert H.parse_candle(_candle(1)).has_book(0.05)


# ------------------------------------------------------------------ caching --

def _no_network(*a, **k):
    raise AssertionError("network call attempted")


def test_fetch_market_minutes_serves_from_cache_without_network(tmp_path, monkeypatch):
    sm = H.parse_settled_market(_market_dict())
    (tmp_path / "candles").mkdir()
    (tmp_path / "candles" / f"{sm.ticker}.json").write_text(
        json.dumps([_candle(sm.open_ts + 120), _candle(sm.open_ts + 60)]))
    monkeypatch.setattr(H, "_get_json", _no_network)
    mins = H.fetch_market_minutes(sm, cache_dir=tmp_path)
    assert [m.end_ts for m in mins] == [sm.open_ts + 60, sm.open_ts + 120]   # sorted


def test_fetch_market_minutes_writes_cache_on_first_fetch(tmp_path, monkeypatch):
    sm = H.parse_settled_market(_market_dict())
    calls = []

    def fake(url, retries=3):
        calls.append(url)
        return {"candlesticks": [_candle(sm.open_ts + 60)]}
    monkeypatch.setattr(H, "_get_json", fake)
    monkeypatch.setattr(H, "REQUEST_PAUSE", 0.0)
    assert len(H.fetch_market_minutes(sm, cache_dir=tmp_path)) == 1
    assert len(calls) == 1
    assert "period_interval=1" in calls[0] and sm.ticker in calls[0]
    monkeypatch.setattr(H, "_get_json", _no_network)
    assert len(H.fetch_market_minutes(sm, cache_dir=tmp_path)) == 1   # cached


def test_fetch_settled_markets_pages_filters_and_caches_only_complete_days(
        tmp_path, monkeypatch):
    day = 1_789_000_000 // 86400 * 86400            # a UTC day boundary in the past
    close_a = day + 900
    close_b = day + 1800
    a = _market_dict(ticker="A", close=_iso(close_a))
    b = _market_dict(ticker="B", close=_iso(close_b), result="yes")
    pages = {None: {"markets": [a], "cursor": "p2"}, "p2": {"markets": [b], "cursor": ""}}
    urls = []

    def fake(url, retries=3):
        urls.append(url)
        cur = url.split("cursor=")[1] if "cursor=" in url else None
        return pages[cur]
    monkeypatch.setattr(H, "_get_json", fake)
    monkeypatch.setattr(H, "REQUEST_PAUSE", 0.0)
    got = H.fetch_settled_markets(day, day + 86400, cache_dir=tmp_path)
    assert [m.ticker for m in got] == ["A", "B"]
    assert len(urls) == 2 and "min_close_ts" in urls[0]
    # the day is long past, so it was frozen to disk
    assert (tmp_path / "settled" / f"{E.KALSHI_BTC_SERIES}_{day}.json").exists()
    # ...and a narrower span reads the cache, filters by close time, no network
    monkeypatch.setattr(H, "_get_json", _no_network)
    got2 = H.fetch_settled_markets(day + 1000, day + 86400, cache_dir=tmp_path)
    assert [m.ticker for m in got2] == ["B"]


def _iso(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------ pairing rule --

OPEN = 1_789_000_000 // 900 * 900
CLOSE = OPEN + 900


def _history(outcome_up=True, strike=65000.0, quotes=None):
    """One window. `quotes` maps minute-of-window -> (ask, bid)."""
    quotes = quotes or {k: (0.55, 0.54) for k in range(1, 15)}
    sm = H.SettledMarket(ticker="W1", strike=strike, open_ts=OPEN, close_ts=CLOSE,
                         outcome_up=outcome_up)
    mins = [H.MarketMinute(end_ts=OPEN + k * 60, yes_ask=a, yes_bid=b, mean=None,
                           volume=1.0) for k, (a, b) in quotes.items()]
    return [H.MarketHistory(sm, mins)]


def _candles(start, n, base=65000.0, step=0.0):
    """Flat-ish 1-min bars with a distinct close per bar so pairing is checkable."""
    out = []
    for i in range(n):
        px = base + step * i
        out.append({"ts": start + i * 60, "open": px, "high": px + 1,
                    "low": px - 1, "close": px + (i % 7) * 0.5, "volume": 1.0})
    return out


def test_pairing_uses_the_coinbase_bar_that_closed_at_the_candle_end():
    candles = _candles(OPEN - 120 * 60, 140)
    by_ts = {c["ts"]: c for c in candles}
    samples = MB.pair_history(_history(), candles)
    assert len(samples) == 14
    for s in samples:
        # Kalshi candle end T  <->  Coinbase bar ts = T-60 (which closes at T)
        assert s.price == by_ts[s.end_ts - 60]["close"]
        assert s.minutes_left == pytest.approx((CLOSE - s.end_ts) / 60)
        assert s.raw_prob == s.model_prob          # no recalibrator supplied
    assert [s.minutes_left for s in samples] == sorted(
        [s.minutes_left for s in samples], reverse=True)


def test_pairing_never_uses_closes_after_the_paired_bar():
    """Vol is computed from closes up to and including the paired bar only.
    Poison every later bar; sigma must not move."""
    clean = _candles(OPEN - 120 * 60, 140)
    poisoned = [dict(c) for c in clean]
    for c in poisoned:
        if c["ts"] > OPEN + 5 * 60 - 60:        # everything after minute 5's bar
            c["close"] *= 3.0
    s_clean = [s for s in MB.pair_history(_history(), clean) if s.end_ts == OPEN + 300]
    s_pois = [s for s in MB.pair_history(_history(), poisoned) if s.end_ts == OPEN + 300]
    assert s_clean[0].sigma == pytest.approx(s_pois[0].sigma)
    assert s_clean[0].model_prob == pytest.approx(s_pois[0].model_prob)


def test_pairing_drops_minutes_without_a_book_bar_or_vol_history():
    quotes = {k: (0.55, 0.54) for k in range(1, 15)}
    quotes[3] = (1.0, 0.001)         # empty placeholder book
    quotes[4] = (0.80, 0.30)         # 50c wide
    hist = _history(quotes=quotes)
    candles = [c for c in _candles(OPEN - 120 * 60, 140) if c["ts"] != OPEN + 6 * 60 - 60]
    samples = MB.pair_history(hist, candles)
    minutes = sorted((s.end_ts - OPEN) // 60 for s in samples)
    assert 3 not in minutes and 4 not in minutes and 6 not in minutes
    assert len(samples) == 11
    # too little spot history -> nothing at all, rather than a crash
    assert MB.pair_history(hist, _candles(OPEN - 3 * 60, 18)) == []
    # ...and "too little" means short of the FULL vol lookback (minus a small
    # tolerance), not merely the 20-close floor: 40 bars before the open must
    # not price minute 1 off a 40-close sigma.
    short = _candles(OPEN - 40 * 60, 60)
    assert MB.pair_history(hist, short, vol_lookback=90) == []
    assert len(MB.pair_history(hist, short, vol_lookback=40)) > 0
    # minutes at/after the close or at/before the open are never paired
    late = _history(quotes={0: (0.5, 0.49), 15: (0.5, 0.49), 16: (0.5, 0.49)})
    assert MB.pair_history(late, _candles(OPEN - 120 * 60, 140)) == []


def test_pairing_applies_the_recalibrator_and_keeps_the_raw_prob():
    candles = _candles(OPEN - 120 * 60, 140)
    raw = MB.pair_history(_history(), candles)
    cal = MB.pair_history(_history(), candles, recal=E.Recalibrator(a=2.0, b=0.0, n_fit=1))
    for r, c in zip(raw, cal):
        assert c.raw_prob == pytest.approx(r.raw_prob)      # raw survives for refits
        assert c.model_prob == pytest.approx(E.Recalibrator(a=2.0).apply(r.raw_prob))


# -------------------------------------------------------- bet selection --

def _sample(ticker, minute, model, ask, bid, outcome_up):
    return MB.PairedSample(ticker=ticker, end_ts=OPEN + minute * 60,
                           minutes_left=15 - minute, price=1.0, sigma=0.0005,
                           raw_prob=model, model_prob=model, yes_ask=ask, yes_bid=bid,
                           outcome_up=outcome_up)


def test_choose_side_is_the_single_rule_live_and_replay_share():
    from btc_edge.decision import choose_side
    assert choose_side(0.58, 0.50, 0.51, 0.05) == ("UP", 0.50, pytest.approx(0.08))
    assert choose_side(0.20, 0.31, 0.70, 0.05) == ("DOWN", 0.70, pytest.approx(0.10))
    assert choose_side(0.52, 0.50, 0.51, 0.05) is None          # 2% is not 5%
    assert choose_side(0.99, 1.00, 0.01, 0.05) is None          # cannot buy at 100c
    # decide() goes through the same function: same side, same edge, same note
    d = E.decide(65200, 64800, 5.0, 0.0006, E.Quote(up_cost_cents=20, down_cost_cents=82))
    assert d.recommended_side == "UP" and d.edge_up == pytest.approx(d.model_prob_up - 0.20)


def test_one_bet_per_window_at_the_first_minute_that_clears_the_edge():
    rows = [
        _sample("W", 1, 0.52, 0.50, 0.49, 1),    # 2% edge: no
        _sample("W", 2, 0.58, 0.50, 0.49, 1),    # 8% edge UP: bet here
        _sample("W", 3, 0.90, 0.50, 0.49, 1),    # bigger edge later: hindsight, ignored
    ]
    bets = MB.select_window_bets(rows, min_edge=0.05)
    assert len(bets) == 1
    b = bets[0]
    assert (b.side, b.minutes_left, b.cost) == ("UP", 13, 0.50)
    assert b.edge == pytest.approx(0.08)
    assert b.won and b.gross_cents == pytest.approx(50.0)
    assert b.fee_cents == kalshi_fee_cents(50) == 2
    assert b.net_cents == pytest.approx(48.0)


def test_down_bets_pay_the_no_ask_and_lose_when_it_settles_up():
    # model 20% up -> 80% down; no_ask = 1 - yes_bid = 0.70; edge 10%
    rows = [_sample("W", 5, 0.20, 0.31, 0.30, 1)]
    (b,) = MB.select_window_bets(rows, min_edge=0.05)
    assert b.side == "DOWN" and b.cost == pytest.approx(0.70)
    assert not b.won and b.gross_cents == pytest.approx(-70.0)
    assert b.net_cents == pytest.approx(-70.0 - kalshi_fee_cents(70))


def test_no_bet_when_the_edge_never_clears_and_fee_fn_is_pluggable():
    rows = [_sample("W", k, 0.52, 0.50, 0.49, 1) for k in range(1, 15)]
    assert MB.select_window_bets(rows, min_edge=0.05) == []
    rows = [_sample("A", 1, 0.60, 0.50, 0.49, 1), _sample("B", 1, 0.60, 0.50, 0.49, 0)]
    bets = MB.select_window_bets(rows, min_edge=0.05, fee_fn=lambda c: 0.0)
    assert sorted(b.ticker for b in bets) == ["A", "B"]
    assert all(b.fee_cents == 0.0 for b in bets)
    assert [b.won for b in sorted(bets, key=lambda b: b.ticker)] == [True, False]


def test_pnl_stats_interval_and_significance():
    assert MB.pnl_stats([]) is None
    st = MB.pnl_stats([10.0, 12.0, 11.0, 9.0, 13.0, 11.0] * 2)
    assert st.n == 12 and st.hit_rate == 1.0 and st.mean == pytest.approx(11.0)
    assert st.ci95[0] > 0 and st.significant
    flat = MB.pnl_stats([50.0, -50.0] * 10)
    assert flat.mean == 0 and not flat.significant


def test_pnl_stats_refuses_an_interval_below_the_minimum_n():
    # Two identical wins used to yield a zero-width interval that read as
    # "significant at 95%". Below MIN_N_FOR_CI there is no interval at all.
    st = MB.pnl_stats([43.0, 43.0])
    assert st.n == 2 and st.mean == 43.0 and st.ci95 is None
    assert st.significant is False
    assert MB.pnl_stats([43.0]).ci95 is None
    line = MB._pnl_line("x", st)
    assert "no interval" in line and "CI" not in line
    assert MB.pnl_stats([43.0] * MB.MIN_N_FOR_CI).ci95 is not None


def test_edge_bands_follow_the_threshold_actually_used():
    assert [b[0] for b in MB.edge_bands(0.05)] == ["5%-10%", "10%-20%", "20%+"]
    assert MB.edge_bands(0.05)[0][1] == 0.05
    assert [b[0] for b in MB.edge_bands(0.02)] == ["2%-10%", "10%-20%", "20%+"]
    assert [b[0] for b in MB.edge_bands(0.12)] == ["12%-20%", "20%+"]
    assert [b[0] for b in MB.edge_bands(0.25)] == ["25%+"]
    # a 3% bet under a 2% threshold lands in the first band, correctly labelled
    b = MB.WindowBet("W", "UP", 10.0, 0.5, 0.03, True, 50.0, 2.0)
    bands = MB._band([b] * MB.MIN_N_FOR_CI, lambda x: x.edge, MB.edge_bands(0.02))
    assert list(bands) == ["2%-10%"]


# ------------------------------------------------------------ end to end --

def _many_windows(n=40, seed=3):
    """n windows with a random walk and a market that quotes the model's own
    probability +/- noise, so bets exist but no edge does."""
    import random
    rng = random.Random(seed)
    hist, candles = [], []
    t0 = OPEN - 200 * 60
    px = 65000.0
    # 8 windows (120 min) of price history before the first scored window, so
    # every minute has the full 90-bar vol lookback behind it.
    for i in range(n + 8):
        # 15 bars per window; a continuous price path across windows
        for k in range(15):
            px *= math.exp(rng.gauss(0, 0.0005))
            candles.append({"ts": t0 + (i * 15 + k) * 60, "open": px, "high": px,
                            "low": px, "close": px, "volume": 1.0})
    for w in range(n):
        o = t0 + (w + 8) * 15 * 60
        c = o + 900
        strike = next(b["close"] for b in candles if b["ts"] == o - 60)
        settle = next(b["close"] for b in candles if b["ts"] == c - 60)
        mins = []
        for k in range(1, 15):
            p = 0.5 + rng.uniform(-0.3, 0.3)
            mins.append(H.MarketMinute(end_ts=o + k * 60, yes_ask=round(p + 0.005, 3),
                                       yes_bid=round(p - 0.005, 3), mean=None, volume=1))
        hist.append(H.MarketHistory(H.SettledMarket(f"W{w}", strike, o, c, settle >= strike),
                                    mins))
    return hist, candles


def test_run_market_backtest_end_to_end_is_consistent_and_deterministic():
    hist, candles = _many_windows()
    r = MB.run_market_backtest(hist, candles, n_boot=300)
    assert r.windows == 40 and r.quoted_windows == 40
    assert r.samples == 40 * 14
    assert 0 < len(r.bets) <= 40
    assert len({b.ticker for b in r.bets}) == len(r.bets)         # one per window
    assert r.net.mean == pytest.approx(r.gross.mean - sum(b.fee_cents for b in r.bets) / len(r.bets))
    assert r.brier_first["windows"] == 40 and r.brier_first["samples"] == 40
    assert r.brier_all["windows"] == 40 and r.brier_all["samples"] == 560
    assert r.brier_all["ci95"][0] <= r.brier_all["delta"] <= r.brier_all["ci95"][1]
    assert r.book_first_minute == [1.0] * 40
    r2 = MB.run_market_backtest(hist, candles, n_boot=300)
    assert r2.brier_all == r.brier_all and r2.net.ci95 == r.net.ci95
    # every bet lands in exactly one entry band and one edge band
    assert sum(s.n for s in r.by_entry_band.values()) == len(r.bets)
    assert sum(s.n for s in r.by_edge_band.values()) == len(r.bets)
    assert list(r.by_edge_band)[0].startswith("5%")


def test_run_market_backtest_rejects_a_zero_bootstrap_instead_of_crashing_late():
    hist, candles = _many_windows(n=5)
    with pytest.raises(ValueError):
        MB.run_market_backtest(hist, candles, n_boot=0)
    # and the CLI refuses it before loading anything
    from btc_edge.cli import _positive_int
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    assert _positive_int("7") == 7


def test_report_mentions_every_headline_and_never_claims_a_null_is_significant():
    hist, candles = _many_windows()
    r = MB.run_market_backtest(hist, candles, n_boot=300)
    txt = MB.format_market_report(r)
    for needle in ("WINDOW-LEVEL PnL", "net of fee", "BRIER", "MID",
                   "first quoted minute/window", "calibration", "resamples"):
        assert needle in txt
    if not r.net.significant:
        assert "no demonstrated edge" in txt


def test_report_survives_an_empty_result():
    r = MB.run_market_backtest([], [], n_boot=10)
    txt = MB.format_market_report(r)
    assert "(no bets)" in txt
    assert r.brier_all is None and r.bets == []


# ------------------------------------------------- history robustness --

def test_load_history_skips_a_window_whose_fetch_fails(tmp_path, monkeypatch):
    day = 1_789_000_000 // 86400 * 86400
    ms = [_market_dict(ticker=f"T{i}", close=_iso(day + 900 * (i + 1))) for i in range(3)]
    (tmp_path / "settled").mkdir()
    (tmp_path / "settled" / f"{E.KALSHI_BTC_SERIES}_{day}.json").write_text(json.dumps(ms))

    def flaky(url, retries=3):
        if "/markets?" in url:                      # an uncached neighbouring day
            return {"markets": [], "cursor": ""}
        if "T1" in url:
            raise RuntimeError("request failed after 3 tries")
        return {"candlesticks": [_candle(day + 60)]}
    monkeypatch.setattr(H, "_get_json", flaky)
    monkeypatch.setattr(H, "REQUEST_PAUSE", 0.0)
    got = H.load_history(days=0.5, end_ts=day + 4000, cache_dir=tmp_path, verbose=False)
    assert [h.market.ticker for h in got] == ["T0", "T2"]
    # the failure was NOT cached as an empty window
    assert not (tmp_path / "candles" / "T1.json").exists()
    assert (tmp_path / "candles" / "T0.json").exists()


def test_malformed_payloads_are_never_written_to_the_cache(tmp_path, monkeypatch):
    sm = H.parse_settled_market(_market_dict())
    monkeypatch.setattr(H, "_get_json", lambda url, retries=3: {"error": "gone"})
    monkeypatch.setattr(H, "REQUEST_PAUSE", 0.0)
    with pytest.raises(RuntimeError):
        H.fetch_market_minutes(sm, cache_dir=tmp_path)
    assert not (tmp_path / "candles" / f"{sm.ticker}.json").exists()
    day = 1_789_000_000 // 86400 * 86400
    with pytest.raises(RuntimeError):
        H.fetch_settled_markets(day, day + 86400, cache_dir=tmp_path)
    assert not (tmp_path / "settled" / f"{E.KALSHI_BTC_SERIES}_{day}.json").exists()
    # a genuinely empty, well-formed response IS cached (no markets that day)
    monkeypatch.setattr(H, "_get_json", lambda url, retries=3: {"markets": [], "cursor": ""})
    assert H.fetch_settled_markets(day, day + 86400, cache_dir=tmp_path) == []
    assert (tmp_path / "settled" / f"{E.KALSHI_BTC_SERIES}_{day}.json").exists()


def test_candle_span_cache_is_keyed_on_the_span_not_the_clock(tmp_path, monkeypatch):
    import sys
    B = sys.modules["btc_edge.backtest"]      # the module, not the re-exported function
    calls = []

    def fake_range(start, end, granularity=60):
        calls.append((start, end))
        return [{"ts": start, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]
    monkeypatch.setattr(B, "fetch_candle_range", fake_range)
    a = B.load_candle_span_cached(1_789_000_100, 1_789_010_000, cache_dir=tmp_path)
    b = B.load_candle_span_cached(1_789_000_900, 1_789_010_500, cache_dir=tmp_path)
    assert a == b and len(calls) == 1               # same hour-snapped span: one fetch
    B.load_candle_span_cached(1_789_000_100, 1_789_020_000, cache_dir=tmp_path)
    assert len(calls) == 2                          # a different hour: a new fetch
    assert calls[0][0] % 3600 == 0 and calls[0][1] % 3600 == 0
    assert calls[0][0] <= 1_789_000_100 and calls[0][1] >= 1_789_010_000
