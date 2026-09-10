"""BTC 15-min binary-contract edge detector.

A driftless-GBM fair-value model for Kalshi's KXBTC15M "Bitcoin up/down"
contracts, plus the paper-trading, settlement and backtest machinery to find
out whether the model actually beats the market's quote. It does not, yet — see
README.md.

The package is a straight decomposition of what was one 1,500-line file:

    data         Coinbase candles/spot + the Kalshi book (all network I/O)
    model        realized vol, effective_tau, prob_finish_above (pure)
    calibration  the frozen one-parameter Platt recalibrator
    decision     decide(): model prob + quote -> a logged Decision
    metrics      Brier / log-loss / calibration table
    vol          EWMA + GARCH(1,1) alternatives to the flat stdev estimator
    tails        Student-t innovations as an alternative to the Gaussian tail
    backtest     historical replay + the time-ordered recal harness
    experiments  vol x tail bake-off, scored on held-out windows
    report       edge_report(): model-vs-market over quoted+settled rows
    live         paper log, outcome fill, and the watch/capture loops
    cli          `python -m btc_edge ...`

Names are re-exported here so `import btc_edge as E` reaches everything the
old flat module exposed.
"""
from btc_edge.backtest import (
    BacktestResult,
    RecalEval,
    Sample,
    backtest,
    collect_samples,
    fit_and_eval_recalibration,
    load_candles_cached,
    print_backtest,
    print_recal_eval,
    score_samples,
    vig_market,
)
from btc_edge.calibration import (
    RECAL_PATH,
    Recalibrator,
    _logit,
    _sigmoid,
    fit_recalibrator,
)
from btc_edge.data import (
    KALSHI_BASE,
    KALSHI_BTC_SERIES,
    KalshiMarket,
    Quote,
    _iso_to_ts,
    _market_cents,
    current_price,
    fetch_candle_range,
    fetch_kalshi_market,
    fetch_kalshi_quote,
    fetch_kalshi_settlement,
    fetch_recent_1min_candles,
    kalshi_quote_fn,
)
from btc_edge.decision import KELLY_CAP, MIN_EDGE, Decision, decide
from btc_edge.metrics import (
    _brier,
    _calibration_report,
    _log_loss,
    block_bootstrap_brier_delta,
    brier_delta,
)
from btc_edge.model import (
    MIN_CLOSES_FOR_VOL,
    SETTLE_AVG_MINUTES,
    WINDOW_MINUTES,
    _norm_cdf,
    effective_tau,
    prob_finish_above,
    realized_vol_per_minute,
)
from btc_edge.experiments import (
    ExperimentReport,
    MemoVol,
    Variant,
    VariantResult,
    block_bootstrap_variant_delta,
    build_variants,
    format_report,
    print_report,
    run_experiment,
)
from btc_edge.report import edge_report
from btc_edge.tails import (
    normal_cdf,
    regularized_incomplete_beta,
    standardized_t_cdf,
    student_t_cdf,
)
from btc_edge.vol import (
    DEFAULT_LAMBDA,
    Garch11,
    GarchVol,
    ewma_vol_factory,
    ewma_vol_per_minute,
    fit_garch11,
    log_returns,
)
from btc_edge.live import (
    CSV_FIELDS,
    LOG_PATH,
    OUTCOME_FIELDS,
    LivePrompter,
    _pnl_cents,
    _sleep_until,
    fill_outcomes,
    log_decision,
    log_summary,
    next_window_expiry,
    settlement_price,
    watch_forever,
    watch_kalshi,
    watch_window,
)

__all__ = [
    # data
    "Quote", "KalshiMarket", "KALSHI_BASE", "KALSHI_BTC_SERIES",
    "current_price", "fetch_recent_1min_candles", "fetch_candle_range",
    "fetch_kalshi_market", "fetch_kalshi_quote", "fetch_kalshi_settlement",
    "kalshi_quote_fn", "_market_cents", "_iso_to_ts",
    # model
    "realized_vol_per_minute", "prob_finish_above", "effective_tau", "_norm_cdf",
    "SETTLE_AVG_MINUTES", "MIN_CLOSES_FOR_VOL", "WINDOW_MINUTES",
    # calibration
    "Recalibrator", "fit_recalibrator", "RECAL_PATH", "_logit", "_sigmoid",
    # decision
    "Decision", "decide", "MIN_EDGE", "KELLY_CAP",
    # metrics
    "_brier", "_log_loss", "_calibration_report",
    "brier_delta", "block_bootstrap_brier_delta",
    # backtest
    "Sample", "BacktestResult", "RecalEval", "collect_samples", "score_samples",
    "backtest", "fit_and_eval_recalibration", "load_candles_cached",
    "vig_market", "print_backtest", "print_recal_eval",
    # vol
    "ewma_vol_per_minute", "ewma_vol_factory", "GarchVol", "Garch11",
    "fit_garch11", "log_returns", "DEFAULT_LAMBDA",
    # tails
    "normal_cdf", "student_t_cdf", "standardized_t_cdf",
    "regularized_incomplete_beta",
    # experiments
    "Variant", "VariantResult", "ExperimentReport", "MemoVol",
    "build_variants", "run_experiment", "block_bootstrap_variant_delta",
    "format_report", "print_report",
    # report
    "edge_report",
    # live
    "LOG_PATH", "CSV_FIELDS", "OUTCOME_FIELDS", "log_decision",
    "settlement_price", "_pnl_cents", "fill_outcomes", "log_summary",
    "LivePrompter", "next_window_expiry", "_sleep_until",
    "watch_window", "watch_forever", "watch_kalshi",
]
