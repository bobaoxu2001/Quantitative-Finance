from __future__ import annotations

import numpy as np
import pandas as pd

from hourly_trading_system.analytics import compute_performance_report


def test_compute_performance_report_outputs_required_metrics() -> None:
    times = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    returns = pd.Series(np.random.default_rng(1).normal(0.0002, 0.003, len(times)))
    equity = 100_000 * (1 + returns).cumprod()
    curve = pd.DataFrame({"event_time": times, "equity": equity.values})
    benchmark = pd.Series(np.random.default_rng(2).normal(0.0001, 0.0025, len(times)), index=times)

    report = compute_performance_report(
        equity_curve=curve,
        benchmark_returns=benchmark,
        turnover_series=pd.Series(np.abs(np.random.default_rng(3).normal(0.1, 0.02, len(times)))),
        exposure_series=pd.Series(np.random.default_rng(4).uniform(0.3, 1.0, len(times))),
    )
    required = {
        "cagr",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "turnover",
        "exposure_pct",
        "beta_to_spy",
        "information_ratio",
        "tail_ratio",
        "skew",
        "kurtosis",
    }
    assert required.issubset(report.summary.keys())
