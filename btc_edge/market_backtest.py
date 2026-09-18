"""Score the model against REAL Kalshi quotes over settled history.

This replaces waiting weeks of live capture. `btc_edge.history` supplies every
settled window with its per-minute yes bid/ask; Coinbase supplies the spot the
model would have seen at the same minute; the model prices each minute; and the
same rules as the live `edge_report` decide what would have been bet, at what
cost, and what it returned — net of Kalshi's fee.

Pairing rule, because it decides whether the result is real:

    Kalshi candle end_ts = T covers (T-60, T] and its closing yes_ask is the
    ask as of T.  Coinbase bar ts = T-60 closes at T.  They are paired.

So the spot the model sees is never fresher than the quote it is scored
against. If anything the quote is a hair fresher (it can react to trades up to
T while the Coinbase close is one print), which biases AGAINST finding edge.

Independence: one bet per window (the first minute clearing the threshold, the
only one that could have been acted on), PnL intervals over windows, and Brier
deltas block-bootstrapped over windows. Rows within a window share one
settlement and never count as separate observations.
"""
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import stdev
from typing import Callable, Optional

from btc_edge.calibration import Recalibrator
from btc_edge.decision import MIN_EDGE, choose_side
from btc_edge.fees import kalshi_fee_cents
from btc_edge.history import MarketHistory
from btc_edge.live.fill import _pnl_cents
from btc_edge.metrics import _calibration_report
from btc_edge.model import prob_finish_above, realized_vol_per_minute
from btc_edge.report import _brier_level

MAX_SPREAD = 0.05        # wider than 5c = no real two-sided book, skip the minute
BOOTSTRAP_RESAMPLES = 4_000
BOOTSTRAP_SEED = 20260914
# Below this many bets a normal-approximation interval is not worth printing:
# two identical wins give a zero-width "significant" interval.
MIN_N_FOR_CI = 10
# A minute needs (nearly) the full vol lookback behind it, as collect_samples
# requires; a few missing Coinbase bars inside the lookback are tolerated.
LOOKBACK_TOLERANCE = 5


@dataclass(frozen=True)
class PairedSample:
    """One minute of one window: what the model said, what the book said."""
    ticker: str
    end_ts: int
    minutes_left: float
    price: float            # Coinbase close of the bar ending at end_ts
    sigma: float
    raw_prob: float         # unrecalibrated GBM probability, kept for refits
    model_prob: float       # recalibrated if a recalibrator was supplied
    yes_ask: float          # dollars 0-1: cost of Up
    yes_bid: float
    outcome_up: int

    @property
    def mid(self) -> float:
        return (self.yes_ask + self.yes_bid) / 2.0

    @property
    def no_ask(self) -> float:
        return 1.0 - self.yes_bid


@dataclass(frozen=True)
class WindowBet:
    ticker: str
    side: str               # "UP" | "DOWN"
    minutes_left: float
    cost: float             # dollars 0-1, the ask actually paid
    edge: float             # model_prob(side) - cost
    won: bool
    gross_cents: float
    fee_cents: float

    @property
    def net_cents(self) -> float:
        return self.gross_cents - self.fee_cents


@dataclass
class PnlStats:
    n: int
    hit_rate: float
    mean: float
    sd: float
    ci95: Optional[tuple[float, float]]   # None below MIN_N_FOR_CI

    @property
    def significant(self) -> bool:
        if self.ci95 is None:
            return False
        return self.ci95[0] > 0 or self.ci95[1] < 0


def pnl_stats(values: list[float], min_n_for_ci: int = MIN_N_FOR_CI) -> Optional[PnlStats]:
    n = len(values)
    if n < 1:
        return None
    mean = sum(values) / n
    sd = stdev(values) if n >= 2 else 0.0
    ci = None
    if n >= min_n_for_ci:
        se = sd / math.sqrt(n)
        ci = (mean - 1.96 * se, mean + 1.96 * se)
    return PnlStats(n=n, hit_rate=sum(1 for v in values if v > 0) / n,
                    mean=mean, sd=sd, ci95=ci)


@dataclass
class MarketBacktestResult:
    windows: int                     # settled windows supplied
    quoted_windows: int              # windows with at least one real-book minute
    samples: int                     # paired minutes
    min_edge: float
    max_spread: float
    recal: Recalibrator
    bets: list[WindowBet]
    gross: Optional[PnlStats]
    net: Optional[PnlStats]
    by_edge_band: dict[str, PnlStats]
    by_entry_band: dict[str, PnlStats]
    brier_all: Optional[dict]        # every paired minute, blocks = windows
    brier_first: Optional[dict]      # first quoted minute per window
    book_first_minute: list[float]   # minute-of-window the book first appears
    calibration_model: str
    calibration_market: str
    bootstrap: dict = field(default_factory=dict)


# ---------------------------------------------------------------- pairing --

def pair_history(
    history: list[MarketHistory],
    candles: list[dict],
    recal: Optional[Recalibrator] = None,
    vol_lookback: int = 90,
    max_spread: float = MAX_SPREAD,
    vol_fn: Callable[[list[float]], float] = realized_vol_per_minute,
    prob_fn: Callable[..., float] = prob_finish_above,
) -> list[PairedSample]:
    """
    Join each window's Kalshi minutes to the Coinbase bar that closed at the same
    instant, and price each with the model using only closes up to that bar.
    Minutes without a real two-sided book, without a matching Coinbase bar, or
    without (nearly) the full `vol_lookback` of spot history are dropped —
    the same lookback requirement `collect_samples` enforces, so a short
    candle span cannot silently price the first windows off a stub sigma.
    """
    by_ts = {int(c["ts"]): c for c in candles}
    ts_sorted = sorted(by_ts)
    closes = [float(by_ts[t]["close"]) for t in ts_sorted]
    need = max(2, vol_lookback - LOOKBACK_TOLERANCE)

    out: list[PairedSample] = []
    for h in history:
        m = h.market
        for mm in h.minutes:
            T = mm.end_ts
            if T <= m.open_ts or T >= m.close_ts:
                continue
            if not mm.has_book(max_spread):
                continue
            bar = by_ts.get(T - 60)
            if bar is None:
                continue
            hi = bisect_right(ts_sorted, T - 60)          # includes the bar itself
            lo = bisect_left(ts_sorted, T - 60 - vol_lookback * 60)
            hist = closes[lo:hi]
            if len(hist) < need:
                continue
            sigma = vol_fn(hist)
            price = float(bar["close"])
            minutes_left = (m.close_ts - T) / 60.0
            raw = prob_fn(price, m.strike, minutes_left, sigma)
            p = recal.apply(raw) if recal else raw
            out.append(PairedSample(
                ticker=m.ticker, end_ts=T, minutes_left=minutes_left,
                price=price, sigma=sigma, raw_prob=raw, model_prob=p,
                yes_ask=mm.yes_ask, yes_bid=mm.yes_bid,
                outcome_up=1 if m.outcome_up else 0))
    return out


# ------------------------------------------------------------- bet selection --

def select_window_bets(samples: list[PairedSample],
                       min_edge: float = MIN_EDGE,
                       fee_fn: Callable[[float], float] = kalshi_fee_cents,
                       ) -> list[WindowBet]:
    """
    The live rule, replayed: in each window take the FIRST minute at which the
    model's probability beats the ask by `min_edge`, on whichever side clears
    it, and hold to settlement. One bet per window at most. `fee_fn` maps the
    entry price in cents to the fee in cents.
    """
    by_window: dict[str, list[PairedSample]] = defaultdict(list)
    for s in samples:
        by_window[s.ticker].append(s)

    bets: list[WindowBet] = []
    for ticker, rows in by_window.items():
        rows.sort(key=lambda s: s.end_ts)
        for s in rows:
            # Same rule object the live decide() uses, same PnL arithmetic the
            # live fill uses — the replay cannot drift from the real thing.
            pick = choose_side(s.model_prob, s.yes_ask, s.no_ask, min_edge)
            if pick is None:
                continue
            side, cost, edge = pick
            won = (s.outcome_up == 1) if side == "UP" else (s.outcome_up == 0)
            gross = _pnl_cents(side, s.yes_ask, s.no_ask, bool(s.outcome_up))
            bets.append(WindowBet(ticker=ticker, side=side, minutes_left=s.minutes_left,
                                  cost=cost, edge=edge, won=won, gross_cents=gross,
                                  fee_cents=fee_fn(cost * 100.0)))
            break
    return bets


# ---------------------------------------------------------------- scoring --

ENTRY_BANDS = (("T-10..15m", 10.0, 15.0), ("T-5..10m", 5.0, 10.0),
               ("T-2..5m", 2.0, 5.0), ("T-0..2m", 0.0, 2.0))


def edge_bands(min_edge: float) -> tuple[tuple[str, float, float], ...]:
    """Edge bands whose first bucket starts at the threshold actually used."""
    cuts = [c for c in (0.10, 0.20) if c > min_edge]
    bands = []
    lo = min_edge
    for c in cuts:
        bands.append((f"{lo:.0%}-{c:.0%}", lo, c))
        lo = c
    bands.append((f"{lo:.0%}+", lo, 9.0))
    return tuple(bands)


def _band(bets: list[WindowBet], key: Callable[[WindowBet], float],
          bands) -> dict[str, PnlStats]:
    out: dict[str, PnlStats] = {}
    for label, lo, hi in bands:
        vals = [b.net_cents for b in bets if lo <= key(b) < hi]
        st = pnl_stats(vals)
        if st:
            out[label] = st
    return out


def run_market_backtest(
    history: list[MarketHistory],
    candles: list[dict],
    recal: Optional[Recalibrator] = None,
    min_edge: float = MIN_EDGE,
    max_spread: float = MAX_SPREAD,
    vol_lookback: int = 90,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    fee_fn: Callable[[float], float] = kalshi_fee_cents,
) -> MarketBacktestResult:
    recal = recal or Recalibrator()
    samples = pair_history(history, candles, recal=recal,
                           vol_lookback=vol_lookback, max_spread=max_spread)
    bets = select_window_bets(samples, min_edge=min_edge, fee_fn=fee_fn)

    # when does a real book first appear, in minutes after the open?
    first_book: dict[str, float] = {}
    open_by_ticker = {h.market.ticker: h.market.open_ts for h in history}
    for s in samples:
        t = (s.end_ts - open_by_ticker[s.ticker]) / 60.0
        if s.ticker not in first_book or t < first_book[s.ticker]:
            first_book[s.ticker] = t

    # Brier vs the MID — scoring the market at its ask would charge it half a
    # spread on every row and tilt the comparison toward the model.
    by_window: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    first_min: dict[str, PairedSample] = {}
    for s in samples:
        by_window[s.ticker].append((s.model_prob, s.mid, s.outcome_up))
        if s.ticker not in first_min or s.end_ts < first_min[s.ticker].end_ts:
            first_min[s.ticker] = s

    brier_all = _brier_level(list(by_window.values()), n_resamples=n_boot, seed=seed)
    brier_first = _brier_level([[(s.model_prob, s.mid, s.outcome_up)]
                                for s in first_min.values()],
                               n_resamples=n_boot, seed=seed)

    outcomes = [s.outcome_up for s in samples]
    return MarketBacktestResult(
        windows=len(history),
        quoted_windows=len(by_window),
        samples=len(samples),
        min_edge=min_edge,
        max_spread=max_spread,
        recal=recal,
        bets=bets,
        gross=pnl_stats([b.gross_cents for b in bets]),
        net=pnl_stats([b.net_cents for b in bets]),
        by_edge_band=_band(bets, lambda b: b.edge, edge_bands(min_edge)),
        by_entry_band=_band(bets, lambda b: b.minutes_left, ENTRY_BANDS),
        brier_all=brier_all,
        brier_first=brier_first,
        book_first_minute=sorted(first_book.values()),
        calibration_model=_calibration_report([s.model_prob for s in samples], outcomes)
        if samples else "",
        calibration_market=_calibration_report([s.mid for s in samples], outcomes)
        if samples else "",
        bootstrap={"n_resamples": n_boot, "seed": seed},
    )


# ---------------------------------------------------------------- report --

def _pnl_line(label: str, st: Optional[PnlStats]) -> str:
    if st is None:
        return f"  {label:<12} (no bets)"
    head = (f"  {label:<12} n={st.n:5d}  hit {st.hit_rate:5.1%}  "
            f"mean {st.mean:+6.2f}c/bet")
    if st.ci95 is None:
        return head + f"   (n < {MIN_N_FOR_CI}: no interval)"
    lo, hi = st.ci95
    return head + f"   95% CI [{lo:+6.2f}, {hi:+6.2f}]"


def format_market_report(r: MarketBacktestResult) -> str:
    L: list[str] = []
    L.append(f"settled windows {r.windows}   with a real book {r.quoted_windows}   "
             f"paired minutes {r.samples}")
    if r.book_first_minute:
        b = r.book_first_minute
        L.append(f"two-sided book first appears +{b[len(b) // 2]:.0f}m into the window "
                 f"(median; p90 +{b[min(len(b) - 1, 9 * len(b) // 10)]:.0f}m)")
    tag = (f"recal a={r.recal.a:.3f} b={r.recal.b:+.3f}" if r.recal.n_fit
           else "no recalibration")
    L.append(f"model: {tag}   edge threshold {r.min_edge:.0%}   "
             f"book filter spread <= {r.max_spread * 100:.0f}c")

    L.append("")
    L.append("WINDOW-LEVEL PnL — one bet per window at the ask, first minute the "
             "model cleared the threshold, held to settlement:")
    L.append(_pnl_line("gross", r.gross))
    L.append(_pnl_line("net of fee", r.net))
    if r.net is not None and r.net.ci95 is not None:
        if r.net.significant:
            d = "POSITIVE" if r.net.ci95[0] > 0 else "NEGATIVE"
            L.append(f"  -> {d} net edge, significant at 95%")
        else:
            L.append("  -> net of fees the interval straddles zero: no demonstrated edge")
    elif r.net is not None:
        L.append(f"  -> fewer than {MIN_N_FOR_CI} bets: no verdict")
    if r.by_edge_band:
        L.append("  by perceived edge at entry (net):")
        for k, st in r.by_edge_band.items():
            L.append("  " + _pnl_line(k, st))
    if r.by_entry_band:
        L.append("  by time to expiry at entry (net):")
        for k, st in r.by_entry_band.items():
            L.append("  " + _pnl_line(k, st))

    L.append("")
    L.append("BRIER — model vs market MID, paired delta (negative = model better), "
             "block bootstrap over windows:")
    L.append(f"  {'level':<28}{'windows':>8}{'samples':>8}{'model':>8}{'market':>8}"
             f"{'delta':>9}   95% CI")
    for label, lvl in (("first quoted minute/window", r.brier_first),
                       ("all paired minutes", r.brier_all)):
        if lvl:
            lo, hi = lvl["ci95"]
            L.append(f"  {label:<28}{lvl['windows']:>8}{lvl['samples']:>8}"
                     f"{lvl['model']:>8.4f}{lvl['market']:>8.4f}{lvl['delta']:>+9.4f}"
                     f"   [{lo:+.4f}, {hi:+.4f}]  model better in "
                     f"{lvl['p_model_better']:.0%} of resamples")
    L.append(f"  ({r.bootstrap.get('n_resamples', 0):,} resamples, "
             f"seed {r.bootstrap.get('seed')})")

    if r.calibration_model:
        L.append("")
        L.append("calibration, per paired minute (diagnostic — rows are correlated):")
        L.append("  MODEL")
        L.append(r.calibration_model)
        L.append("  MARKET mid")
        L.append(r.calibration_market)
    return "\n".join(L)


def print_market_report(r: MarketBacktestResult) -> None:
    print(format_market_report(r))
