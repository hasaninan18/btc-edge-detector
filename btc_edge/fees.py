"""Kalshi trading fees.

The series endpoint reports KXBTC15M as `fee_type: quadratic, fee_multiplier: 1`,
which is Kalshi's standard schedule:

    fee = 0.07 * C * P * (1 - P)        (dollars, P the contract price in dollars)

rounded UP to the next cent, charged on the taker side at execution. There is
no fee at settlement. At the prices this model tends to enter (30c-70c) that is
1.5c-1.75c per contract, which rounds to 2c — a large fraction of any edge
measured in single cents, and the reason PnL must be reported net.
"""
import math

KALSHI_FEE_RATE = 0.07


def kalshi_fee_cents(price_cents: float, contracts: int = 1,
                     rate: float = KALSHI_FEE_RATE) -> float:
    """
    Fee in cents for an order of `contracts` at `price_cents`, rounded up to the
    next whole cent as Kalshi does. Prices outside (0, 100) carry no fee: a
    contract at 0c or 100c has P*(1-P) = 0.
    """
    if contracts <= 0:
        return 0.0
    p = min(max(price_cents / 100.0, 0.0), 1.0)
    cents = rate * contracts * p * (1.0 - p) * 100.0
    # Round up to the cent. The epsilon keeps an exact whole number of cents
    # (e.g. 100 contracts at 50c = 175.0c) from floating up to the next cent.
    return float(math.ceil(cents - 1e-9))
