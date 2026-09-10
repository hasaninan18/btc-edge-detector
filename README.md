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

## Current result — no demonstrated edge

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

| calibration (quoted samples) | Brier vs realized |
|---|---:|
| model | 0.1035 |
| market | 0.1075 |

**Read this honestly:** the window-level 95% confidence interval **straddles
zero**. The point estimate is positive and the model's Brier score edges the
market's, but on 32 independent bets that is statistically indistinguishable
from no edge at all. The model has *not* been shown to beat the market.

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

python -m btc_edge backtest --days 7
python -m btc_edge recalibrate --days 30 --save   # refit the Platt scaler
python -m btc_edge vol-tails --days 30            # EWMA/GARCH/Student-t bake-off
```

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
  metrics.py      Brier / log-loss / decile calibration table
  vol.py          EWMA + GARCH(1,1) alternatives to the flat stdev estimator
  tails.py        standardised Student-t innovations (incomplete beta, no scipy)
  backtest.py     historical replay + the time-ordered recal fit/eval harness
  experiments.py  vol x tail bake-off scored on held-out windows
  report.py       edge_report(): model vs market over quoted+settled rows
  live/
    paperlog.py   the CSV schema and append
    fill.py       settlement (exchange result first, candle proxy fallback)
    watch.py      the per-minute and Kalshi capture loops
  cli.py          argument parsing and command dispatch
tests/
  test_golden_master.py   pins edge_report()/backtest() output against refactors
  test_vol_and_tails.py   the vol/tail experiment and its statistics
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

The code is a faithful decomposition of the original single file, with a test
suite and a safety net around it. Known open work, not done here:

- window-level **Brier** comparison + a block bootstrap CI (the report
  aggregates PnL per window but still scores Brier on all correlated samples)
- a transaction-cost / quote-staleness audit to establish whether the
  +14.5c/window figure is gross or net
