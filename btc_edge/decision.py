"""Phase 3: turn a model probability and a market quote into a paper decision.

`decide()` is the join point — it takes spot, strike, time left, vol and an
optional quote, applies the recalibrator, and returns a fully-populated
`Decision` row ready for the CSV log.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from btc_edge.calibration import Recalibrator
from btc_edge.data import Quote
from btc_edge.model import prob_finish_above


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
