# btc-edge-detector

A paper-trading harness that asks one question: **does a simple fair-value model
beat the market's own price on Kalshi's 15-minute BTC "up/down" contracts
(series `KXBTC15M`)?**

The model is deliberately plain — geometric Brownian motion with zero drift,
realized 1-minute volatility, and a closed-form `P(settle > strike)` — adjusted
for the fact that these contracts settle on a **60-second BRTI time-average**,
not a point price. A one-parameter Platt recalibrator, fit once on historical
backtest data and then frozen, corrects a small under-dispersion in the raw
model. Everything is logged; nothing places an order.

## Current result — the market beats the model, on 5,678 real windows

Kalshi's public API serves every settled window with its result and strike,
and one candle per minute with the yes bid/ask, going back 60+ days at 96
windows a day. `market-backtest` replays the model against those real quotes.
Run on **2026-09-18** over the 60 days to that date (2026-07-20 → 09-18):

```
python -m btc_edge market-backtest --days 60
```

| | windows | paired minutes |
|---|---:|---:|
| settled windows in span | 5,678 | 76,512 |
| windows with a two-sided book | 5,672 (book forms at +1m in every window) | |

One bet per window, at the ask, on the first minute the model's probability
beat the ask by 5%, held to settlement. Kalshi's fee (7% × P × (1−P) per
contract, rounded up to the cent, ≈2c at these prices) is charged on entry:

| window-level PnL | bets | hit rate | mean / bet | 95% CI |
|---|---:|---:|---:|---|
| gross | 4,852 | 47.5% | +0.71c | [−0.55c, +1.98c] |
| **net of fee** | 4,852 | 47.5% | **−1.16c** | **[−2.43c, +0.10c]** |
| net, edge 5–10% at entry | 4,251 | 46.8% | −1.00c | [−2.34c, +0.35c] |
| net, edge 10–20% at entry | 579 | 52.2% | −2.82c | [−6.59c, +0.94c] |
| net, entered T-10..15m | 3,357 | 52.9% | −1.21c | [−2.78c, +0.36c] |
| net, entered T-2..5m | 358 | 27.4% | −1.33c | [−5.34c, +2.68c] |

Even gross of fees the model does not make money on 60 days; net of fees it
loses about a cent a contract, and the interval only just reaches zero. The
larger the model thought its edge was, the worse it did, with the exception
of 22 bets at 20%+ that are too few to read.

Scoring is unambiguous. The market is scored at its **mid** (scoring it at
the ask would charge it half a spread on every row):

| Brier, model − market mid | windows | samples | model | market | delta | 95% CI (block bootstrap) |
|---|---:|---:|---:|---:|---:|---|
| first quoted minute per window | 5,672 | 5,672 | 0.2385 | 0.2358 | +0.0027 | [+0.0013, +0.0040] |
| all paired minutes (correlated) | 5,672 | 76,512 | 0.1610 | 0.1565 | +0.0045 | [+0.0033, +0.0057] |

**The market is the better forecaster at 95% on both levels**, and the model
scored better in 0 of 4,000 bootstrap resamples. Per-decile calibration shows
the mechanism: the market mid is within 2 points of realised frequency in
every decile, while the model realises 2–7 points *more Up* than it predicts
in every decile but the top one. A driftless random walk from Coinbase spot
is missing something persistent that the book prices in — short-horizon
momentum, order flow, or the index the contract actually settles on.

A note on sample size, since this project has been burned by it before: the
same command over only the 14 days to 2026-09-14 gave net **+1.44c/bet on
1,120 bets, CI [−1.20c, +4.08c]**, and a per-minute Brier delta the model
nearly won. Two weeks of real quotes still looked like a coin flip in the
model's favour. Sixty days settled it, and a re-run four days later on the
rolled-forward span (the table above) moved nothing by more than a few
hundredths of a cent.

Two things about the replay that are deliberate:

- **Spot is never fresher than the quote.** A Kalshi candle ending at T is
  paired with the Coinbase bar that closed at T, so any staleness works
  against the model, not for it.
- **The threshold is not a filter.** The 5% edge fires in 84% of windows at
  around T-11m, because a driftless GBM disagrees with the book by 5% almost
  always at that horizon. That is a fact about the model's noise, not about
  opportunity.

Threats: one 60-day span in a rising market (BTC ran from ~$63k to ~$78k; the
model's uniform Up-side miss is consistent with that, and a falling regime
could flip its sign without changing the conclusion that the book prices it
and the model does not); the Coinbase-to-BRTI basis measured −0.5bp over
these windows, negligible against a 16bp typical 15-minute move; the
last-minute approximation in `effective_tau` is coarse exactly where the
trade tape is busiest.

## The live-capture run (August 2026) — superseded

This was the evidence before `market-backtest` existed. It is kept because it
is what the `edge` command still reports on, and because it shows how a
32-sample point estimate of +14.5c/bet dissolves at n=4,852.

Live quote capture ran from **2026-08-12 to 2026-08-13**, producing 217 paper
samples, of which **204 had both a market quote and a settled outcome**, across
**54 fifteen-minute windows**.

Every sample inside a window resolves on the *same* settlement, so the 204 rows
are not 204 independent observations. Collapsing to one bet per window — the
earliest sample that cleared the 5% edge threshold, the only one you could
actually have acted on — leaves **32 independent window-level bets**.

| view | bets | hit rate | total PnL | mean PnL / bet | 95% CI on the mean |
|---|---:|---:|---:|---:|---|
| **window-level (independent — the number that counts)** | 32 | 65.6% | +465c | **+14.54c** | **[−0.15c, +29.22c]** |
| sample-level (correlated — *not* valid for significance) | 82 | 78.0% | +1,201c | +14.65c | — |

Scoring gets the same treatment. Two separate Brier numbers hide the fact that
both are scored against one shared set of settlements, so the report gives the
**paired delta** (model − market; negative means the model won) with a 95%
interval from a bootstrap that resamples **whole windows**, 10,000 times, seeded:

| level | windows | samples | model | market | delta | 95% CI (block bootstrap) |
|---|---:|---:|---:|---:|---:|---|
| **window-level (traded — the 32 bets above)** | 32 | 32 | 0.1973 | 0.1978 | **−0.0005** | **[−0.0481, +0.0606]** |
| window-level (all quoted windows) | 54 | 54 | 0.1977 | 0.1946 | +0.0031 | [−0.0243, +0.0413] |
| per-sample (correlated — *diagnostic only*) | 54 | 204 | 0.1035 | 0.1075 | −0.0040 | [−0.0156, +0.0111] |

Window-level Brier is higher than per-sample Brier because the actable row sits
~14 minutes from expiry, where the honest answer is near 50/50; the later rows
that drag the per-sample average down are near-certain by then and could not
have been traded.

**Read this honestly:** the window-level 95% confidence interval on PnL
**straddles zero**, and so does the interval on the Brier delta. The −0.0040
per-sample Brier gap this project used to quote does not survive aggregation:
on the 32 traded windows the delta is −0.0005, effectively nothing, and across
all 54 quoted windows it **flips sign** — the market scores slightly better once
the comparison is not restricted to windows where the model disagreed with it.
On 32 independent bets, all of this is statistically indistinguishable from no
edge. The model has *not* been shown to beat the market.

Two caveats that both push the true number *lower* than the table:

- **Quote staleness is not instrumented on these rows.** Spot was read before
  the quote on later captures but the timing delta was not recorded here, so any
  measured edge is an **upper bound** — some of it may just be holding a fresher
  clock than the quote being scored against.
- **Survivorship / selection.** The captured windows are whatever was liquid
  during a one-day run, not a random sample of market conditions.

How much more data would settle it, from `edge_report`:

- if the true edge really is +14.5c/bet → ~33 window-bets total (≈ 1 more)
- if it is only +0.1c (the pessimistic end of the CI) → ~320,000 window-bets

i.e. the honest answer is "keep collecting", and plan against the second number.

## Install

```bash
git clone <this repo>
cd btc-edge-detector
python3 -m pip install -r requirements.txt          # optional: certifi only
python3 -m pip install -r requirements-dev.txt      # to run the tests
```

Python 3.11+ (uses `X | Y` type syntax and `list[...]` generics). The only
optional runtime dependency is `certifi`, and only on Python builds that ship
without root certificates (framework Python on macOS).

## Use

```bash
# one manual decision, logged
python -m btc_edge once --strike 65181.56 --minutes-left 2.5 --up 4.7 --down 95.4

# sample a live window every minute, typing quotes in as you go
python -m btc_edge watch --prompt-quotes

# the real capture loop: contract, strike, expiry and ticker all from Kalshi,
# sub-minute polling, runs until Ctrl-C (needs a network that can reach Kalshi)
python -m btc_edge watch --kalshi

python -m btc_edge fill        # backfill outcomes / PnL for expired windows
python -m btc_edge summary     # calibration + PnL over everything settled
python -m btc_edge edge        # the model-vs-market report above

python -m btc_edge backtest --days 7             # calibration on Coinbase candles
python -m btc_edge market-backtest --days 14     # the model vs REAL Kalshi quotes
python -m btc_edge recalibrate --days 30 --save   # refit the Platt scaler
python -m btc_edge vol-tails --days 30            # EWMA/GARCH/Student-t bake-off
```

`market-backtest` is the command that answers the question. It pulls every
settled window in the span from Kalshi's public API with its per-minute yes
bid/ask (cached under `.kalshi_cache/`, so a re-run is instant), pairs each
minute with the Coinbase bar that closed at the same instant, and replays the
live betting rule net of Kalshi's fee. The first run over 14 days makes ~1,350
requests and takes a few minutes.

The paper log is `paper_trades.csv` in the working directory; the frozen
recalibration is `recalibrator.json` (`a = 1.034`, `b = −0.025`, fit on 30,160
backtest samples — barely sharpening, kept only because it does not hurt
held-out log-loss).

## Layout

```
btc_edge/
  data.py         Coinbase candles/spot + the Kalshi book — all network I/O
  model.py        realized_vol_per_minute, effective_tau, prob_finish_above
  calibration.py  the frozen one-parameter Platt recalibrator
  decision.py     decide(): model prob + market quote -> a logged Decision
  metrics.py      Brier / log-loss / calibration table / block bootstrap
  vol.py          EWMA + GARCH(1,1) alternatives to the flat stdev estimator
  tails.py        standardised Student-t innovations (incomplete beta, no scipy)
  backtest.py     historical replay + the time-ordered recal fit/eval harness
  experiments.py  vol x tail bake-off scored on held-out windows
  history.py      settled Kalshi windows + per-minute bid/ask, disk-cached
  fees.py         Kalshi's quadratic taker fee, rounded up to the cent
  market_backtest.py  the model vs real quotes: pairing, one bet per window,
                  PnL net of fees, Brier vs the mid, block bootstrap
  report.py       edge_report(): model vs market over quoted+settled rows
  live/
    paperlog.py   the CSV schema and append
    fill.py       settlement (exchange result first, candle proxy fallback)
    watch.py      the per-minute and Kalshi capture loops
  cli.py          argument parsing and command dispatch
tests/
  test_golden_master.py   pins edge_report()/backtest() output against refactors
  test_vol_and_tails.py   the vol/tail experiment and its statistics
  test_market_backtest.py fees, history parsing/caching, the pairing rule,
                          bet selection, the report — all offline
  test_*.py               ~40 unit tests (converted from the original scripts)
  fixtures/               a frozen copy of the paper log for the golden master
docs/
  vol-and-tails.md        write-up of the volatility/tail experiment
```

## Tests

```bash
python -m pytest                 # offline, deterministic, ~15s
python -m pytest --run-network   # also hit live Kalshi/Coinbase
```

`tests/test_golden_master.py` locks in the exact numbers `edge_report()` and
`backtest()` produce today. If you change modelling logic and a number moves,
that test is supposed to fail — update its constants in the same commit and say
why.

## Volatility and tail assumptions — tested, no improvement

The pricer assumes constant volatility over the lookback and Gaussian log
returns. Both are false for BTC, so sixteen combinations of EWMA / GARCH(1,1)
volatility and Student-t (ν ∈ {4, 6, 10}) tails were evaluated against the
baseline through the existing held-out recalibration harness — 30 days of
candles, 862 held-out windows, a 2,000-resample block bootstrap over whole
windows.

**None of them beat the baseline by a distinguishable margin.** The best point
estimate was GARCH(1,1) at −0.00040 Brier (0.27% relative), CI
[−0.00130, +0.00048]. Every interval straddles zero, so the flat stdev and the
Gaussian remain the defaults.

The Student-t result has a reason worth knowing: at unit variance a Student-t is
*more* peaked than a normal inside |z| ≈ 1.9 and only fatter beyond it, and 88%
of these 15-minute samples sit inside that crossover. So a t tail acts as a
sharpener here — the same job the Platt recalibrator already does. It improves
raw Brier and then the recalibrator's slope drops from 1.162 to 0.986 and the
gain cancels. Downstream of recalibration the tail choice is close to
unidentified.

Full write-up, including threats to the conclusion: **[docs/vol-and-tails.md](docs/vol-and-tails.md)**.

## Status and what's next

The package is a faithful decomposition of the original single file, with a test
suite and a golden-master safety net around it. Two measurement changes have
landed since, and neither alters the default model:

- the window-level Brier delta with a block bootstrap (above) — the apparent
  per-sample scoring edge does not survive aggregation to independent windows
- the volatility / tail bake-off (above) — no variant beats the baseline on
  held-out windows, so the baseline stays the default

Transaction costs: entries are charged the ask (`yes_ask` / `no_ask`, see
`btc_edge/data.py`) and contracts settle at 0/100 with no exit trade, so the
+14.5c/window figure is net of the spread.

A third measurement change is the one that settles it: `market-backtest`
(above) scores the model against real quotes on 5,678 windows, net of fees,
and finds the market is significantly better. Live capture is no longer the
bottleneck and no longer the evidence; its remaining job is to check that
live fills look like the historical asks.

Known open work:

- [#2](https://github.com/hasaninan18/btc-edge-detector/issues/2):
  `collect_samples` prices bar *t* with a close only known at *t+60* — a
  one-minute look-ahead in the replay the recalibrator was fit on. The
  real-quote replay above is aligned correctly; the misalignment reaches it
  only through the Platt slope (a = 1.034). Fixing it moves the golden-master
  numbers, so it gets its own PR.
- re-run `market-backtest --days 60` a month from now, ideally across a
  falling regime, to see whether the model's Up-side miss is drift or
  something structural
- the `edge` command still scores the market at its ask; move it to the mid
  and charge fees, so the live and historical reports agree by construction
- if any edge is worth chasing it is in the final two minutes, where the trade
  tape shows nearly all volume: a real-time replica of the 60-second BRTI
  settlement average, rather than a 1-minute Coinbase close, is the model that
  could have information the book does not
