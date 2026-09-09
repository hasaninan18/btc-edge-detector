"""
Dumb, dependency-free raw recorder for Kalshi's 15-min BTC up/down market.

Its only job is to bank ground truth while smarter code is being written: every
POLL_SECONDS it appends the full market JSON plus a Coinbase spot tick to a
JSONL file. Model probabilities are NOT computed here — they can be recomputed
offline from the recorded (timestamp, spot, strike) triples, so nothing about
this file needs to be right except the writing.

Never places orders. Read-only public endpoints.
"""
import json
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
COINBASE_TICKER = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
OUT = Path(__file__).with_name("kalshi_raw.jsonl")
POLL_SECONDS = 10


def _ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ctx()


def _get(url: str, timeout: int = 12):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-edge-recorder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode())


def snapshot() -> dict:
    """One timestamped observation: every open 15-min market + spot. None-safe."""
    row = {"ts": time.time(), "iso": datetime.now(timezone.utc).isoformat()}
    try:
        d = _get(f"{KALSHI_BASE}/markets?series_ticker={SERIES}&status=open&limit=20")
        row["markets"] = d.get("markets", [])
    except Exception as e:  # noqa: BLE001 - a bad poll must never end the run
        row["markets_error"] = repr(e)
    try:
        row["spot"] = float(_get(COINBASE_TICKER)["price"])
    except Exception as e:  # noqa: BLE001
        row["spot_error"] = repr(e)
    return row


def main() -> None:
    print(f"recording {SERIES} -> {OUT}  every {POLL_SECONDS}s   (ctrl-c to stop)")
    n = 0
    while True:
        row = snapshot()
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        n += 1
        ms = row.get("markets") or []
        if ms:
            m = ms[0]
            print(f"  {row['iso'][11:19]}  {m.get('ticker')}  "
                  f"strike {m.get('floor_strike')}  spot {row.get('spot')}  "
                  f"yes_ask {m.get('yes_ask_dollars')}  no_ask {m.get('no_ask_dollars')}"
                  f"   [{n} rows]")
        else:
            print(f"  {row['iso'][11:19]}  no open market  [{n} rows]")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
