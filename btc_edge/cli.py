"""BTC 15-min binary contract edge detector — command line.

    python -m btc_edge once     --strike 65181.56 --minutes-left 2.5 \
                                --up 4.7 --down 95.4
    python -m btc_edge watch    [--strike K] [--prompt-quotes] [--kalshi]
    python -m btc_edge fill
    python -m btc_edge summary
    python -m btc_edge edge
    python -m btc_edge backtest --days 7
    python -m btc_edge market-backtest --days 14   # vs REAL Kalshi quotes
    python -m btc_edge recalibrate --days 30 [--save]
    python -m btc_edge vol-tails   --days 30

No auth. No money at risk. Log results, evaluate after 200+ observations before
considering real money.
"""
import argparse
import json
from dataclasses import asdict
from typing import Optional

from btc_edge.backtest import (
    backtest,
    fit_and_eval_recalibration,
    load_candle_span_cached,
    load_candles_cached,
    print_backtest,
    print_recal_eval,
    vig_market,
)
from btc_edge.calibration import RECAL_PATH, Recalibrator
from btc_edge.experiments import build_variants, print_report, run_experiment
from btc_edge.data import Quote, current_price, fetch_recent_1min_candles
from btc_edge.decision import MIN_EDGE, decide
from btc_edge.history import load_history
from btc_edge.market_backtest import (
    BOOTSTRAP_RESAMPLES,
    MAX_SPREAD,
    print_market_report,
    run_market_backtest,
)
from btc_edge.model import realized_vol_per_minute
from btc_edge.report import edge_report
from btc_edge.live.fill import fill_outcomes, log_summary
from btc_edge.live.paperlog import log_decision
from btc_edge.live.watch import LivePrompter, watch_forever, watch_kalshi, watch_window


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


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


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
                      help="add a toy counterparty for PnL sensitivity "
                           "(use market-backtest for real quotes)")
    p_bt.add_argument("--vig", type=float, default=0.04)
    p_bt.add_argument("--recal", action="store_true",
                      help="apply the saved recalibrator during scoring")

    p_mb = sub.add_parser(
        "market-backtest",
        help="score the model against real Kalshi quotes over settled history "
             "(window-level PnL net of fees, Brier vs the mid)")
    p_mb.add_argument("--days", type=float, default=14)
    p_mb.add_argument("--min-edge", type=float, default=MIN_EDGE,
                      help="probability edge over the ask required to bet")
    p_mb.add_argument("--max-spread", type=float, default=MAX_SPREAD,
                      help="minutes with a wider yes bid/ask (dollars) are "
                           "treated as having no book")
    p_mb.add_argument("--vol-lookback", type=int, default=90)
    p_mb.add_argument("--no-recal", action="store_true",
                      help="score the raw GBM probability instead of the "
                           "saved recalibration")
    p_mb.add_argument("--n-boot", type=_positive_int, default=BOOTSTRAP_RESAMPLES,
                      help="block-bootstrap resamples for the Brier delta CI (>= 1)")

    p_rc = sub.add_parser("recalibrate",
                          help="fit the recalibrator on history, save it")
    p_rc.add_argument("--days", type=float, default=30)
    p_rc.add_argument("--vol-lookback", type=int, default=90)
    p_rc.add_argument("--sample-every", type=int, default=1)
    p_rc.add_argument("--split", type=float, default=0.7,
                      help="train fraction (earlier windows); rest is held out")
    p_rc.add_argument("--save", action="store_true",
                      help="write recalibrator.json if it improves held-out loss")

    p_vt = sub.add_parser(
        "vol-tails",
        help="held-out bake-off of EWMA/GARCH vol and Student-t tails "
             "against the baseline")
    p_vt.add_argument("--days", type=float, default=30)
    p_vt.add_argument("--vol-lookback", type=int, default=90)
    p_vt.add_argument("--sample-every", type=int, default=1)
    p_vt.add_argument("--split", type=float, default=0.7,
                      help="train fraction (earlier windows); rest is held out")
    p_vt.add_argument("--no-garch", action="store_true",
                      help="skip GARCH(1,1); the rest of the grid is much faster")
    p_vt.add_argument("--n-boot", type=int, default=2000,
                      help="block-bootstrap resamples for the Brier delta CI")

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
    elif args.cmd == "market-backtest":
        history = load_history(args.days)
        if not history:
            print("no settled windows in that span")
            return 1
        # Spot for exactly the windows loaded, reaching back one vol lookback
        # before the first open — keyed on those bounds, not on the clock, so a
        # re-run is a cache hit and a slow history load cannot shorten it.
        candles = load_candle_span_cached(
            history[0].market.open_ts - args.vol_lookback * 60,
            history[-1].market.close_ts)
        recal = Recalibrator() if args.no_recal else Recalibrator.load(RECAL_PATH)
        r = run_market_backtest(history, candles, recal=recal,
                                min_edge=args.min_edge, max_spread=args.max_spread,
                                vol_lookback=args.vol_lookback, n_boot=args.n_boot)
        print()
        print_market_report(r)
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
    elif args.cmd == "vol-tails":
        candles = load_candles_cached(args.days)
        rep = run_experiment(
            candles, days=args.days, split=args.split,
            vol_lookback=args.vol_lookback, sample_every=args.sample_every,
            variants=build_variants(garch=not args.no_garch),
            n_boot=args.n_boot,
        )
        print()
        print_report(rep)
    return 0
