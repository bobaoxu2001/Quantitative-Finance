#!/usr/bin/env python3
"""Run baseline vs aggressive-v2 backtest comparison on the demo dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hourly_trading_system.analytics import compute_performance_report
from hourly_trading_system.backtest import BacktestArtifacts, HourlyBacktestEngine
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import ExecutionEngine
from hourly_trading_system.features import required_features
from hourly_trading_system.models import FactorLightGBMBaseline
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController
from hourly_trading_system.time_utils import to_utc_period_end, to_utc_timestamp
from run_demo_backtest import make_synthetic_market_data


@dataclass(slots=True)
class ProfileResult:
    profile: str
    full_summary: dict[str, float]
    full_total_return: float
    test_summary: dict[str, float]
    test_total_return: float
    config: dict[str, Any]


def _build_engine(
    config: SystemConfig,
    allocator: HourlyPortfolioAllocator,
) -> HourlyBacktestEngine:
    model = FactorLightGBMBaseline(feature_columns=required_features())
    executor = ExecutionEngine(cost_config=config.costs)
    risk = RiskController(
        max_drawdown=config.risk.max_drawdown_hard_stop,
        target_vol_low=config.portfolio.target_vol_lower,
        target_vol_high=config.portfolio.target_vol_upper,
    )
    return HourlyBacktestEngine(
        config=config,
        model=model,
        allocator=allocator,
        executor=executor,
        risk_controller=risk,
    )


def _test_window(config: SystemConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    window = config.backtest_window
    return to_utc_timestamp(window.test_start), to_utc_period_end(window.test_end)


def _benchmark_returns(market: pd.DataFrame, symbol: str) -> pd.Series:
    """Close-to-close benchmark returns indexed by event_time."""
    bench = market.loc[market["symbol"] == symbol, ["event_time", "close"]].copy()
    bench["event_time"] = pd.to_datetime(bench["event_time"], utc=True)
    closes = bench.sort_values("event_time").set_index("event_time")["close"]
    return closes.pct_change().dropna()


def _test_only_report(
    artifacts: BacktestArtifacts,
    config: SystemConfig,
    benchmark_returns: pd.Series | None = None,
) -> tuple[dict[str, float], float]:
    curve = artifacts.equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    test_start, test_end = _test_window(config)
    test_curve = curve.loc[(curve["event_time"] >= test_start) & (curve["event_time"] <= test_end)].copy()
    if len(test_curve) < 2:
        return {}, float("nan")

    test_report = compute_performance_report(
        equity_curve=test_curve[["event_time", "equity", "cash", "invested"]],
        benchmark_returns=benchmark_returns,
        turnover_series=test_curve["turnover"] if "turnover" in test_curve.columns else None,
        exposure_series=test_curve["exposure"] if "exposure" in test_curve.columns else None,
    )
    return test_report.summary, float(test_report.extras["total_return"])


def _buy_and_hold_report(market: pd.DataFrame, symbol: str, config: SystemConfig) -> tuple[dict[str, float], float]:
    """Test-window metrics of simply holding ``symbol`` (fully invested, no costs)."""
    test_start, test_end = _test_window(config)
    bench = market.loc[market["symbol"] == symbol, ["event_time", "close"]].copy()
    bench["event_time"] = pd.to_datetime(bench["event_time"], utc=True)
    bench = bench.loc[(bench["event_time"] >= test_start) & (bench["event_time"] <= test_end)]
    bench = bench.sort_values("event_time").rename(columns={"close": "equity"})
    report = compute_performance_report(
        equity_curve=bench[["event_time", "equity"]],
        exposure_series=pd.Series([1.0]),
    )
    return report.summary, float(report.extras["total_return"])


def _equal_weight_universe_report(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    config: SystemConfig,
) -> tuple[dict[str, float], float]:
    """Test-window metrics of an hourly-rebalanced equal-weight basket of investable names (no costs)."""
    test_start, test_end = _test_window(config)
    known = membership.loc[pd.to_datetime(membership["known_time"], utc=True) < test_start]
    latest = known.sort_values("known_time").groupby("symbol").tail(1)
    members = set(latest.loc[latest["is_member"].astype(bool), "symbol"])
    prices = market.loc[market["symbol"].isin(members), ["event_time", "symbol", "close"]].copy()
    prices["event_time"] = pd.to_datetime(prices["event_time"], utc=True)
    prices = prices.loc[(prices["event_time"] >= test_start) & (prices["event_time"] <= test_end)]
    wide = prices.pivot(index="event_time", columns="symbol", values="close").sort_index()
    basket_returns = wide.pct_change().mean(axis=1).fillna(0.0)
    equity = (1.0 + basket_returns).cumprod()
    report = compute_performance_report(
        equity_curve=pd.DataFrame({"event_time": equity.index, "equity": equity.to_numpy()}),
        exposure_series=pd.Series([1.0]),
    )
    return report.summary, float(report.extras["total_return"])


def _run_profile(
    profile_name: str,
    data: tuple[pd.DataFrame, ...],
    portfolio_overrides: dict[str, Any] | None = None,
    allocator_kwargs: dict[str, Any] | None = None,
) -> tuple[BacktestArtifacts, ProfileResult]:
    config = SystemConfig()
    portfolio_overrides = portfolio_overrides or {}
    allocator_kwargs = allocator_kwargs or {}
    for key, value in portfolio_overrides.items():
        setattr(config.portfolio, key, value)

    allocator = HourlyPortfolioAllocator(config=config.portfolio, **allocator_kwargs)
    engine = _build_engine(config=config, allocator=allocator)
    market, membership, fundamentals, macro, sentiment, analyst = data

    artifacts = engine.run(
        market=market,
        membership=membership,
        fundamentals=fundamentals,
        macro=macro,
        sentiment=sentiment,
        analyst=analyst,
        start="2025-11-01",
        end="2026-02-26",
        train_end="2025-10-31",
    )

    test_summary, test_total_return = _test_only_report(
        artifacts, config, benchmark_returns=_benchmark_returns(market, config.benchmark_symbol)
    )
    result = ProfileResult(
        profile=profile_name,
        full_summary=artifacts.performance.summary,
        full_total_return=float(artifacts.performance.extras["total_return"]),
        test_summary=test_summary,
        test_total_return=test_total_return,
        config={
            "portfolio": asdict(config.portfolio),
            "allocator_kwargs": allocator_kwargs,
        },
    )
    return artifacts, result


def _to_equity_series(artifacts: BacktestArtifacts, label: str) -> pd.Series:
    curve = artifacts.equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    curve = curve.sort_values("event_time").set_index("event_time")
    return curve["equity"].rename(label)


def _to_exposure_series(artifacts: BacktestArtifacts, label: str) -> pd.Series:
    curve = artifacts.equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    curve = curve.sort_values("event_time").set_index("event_time")
    if "exposure" not in curve.columns:
        return pd.Series(dtype=float, name=label)
    return curve["exposure"].rename(label)


def _save_figure(fig: go.Figure, path: Path, width: int = 1300, height: int = 760) -> None:
    fig.update_layout(template="plotly_white")
    fig.write_image(str(path), width=width, height=height, scale=2)


def _build_comparison_df(
    baseline: ProfileResult,
    aggressive: ProfileResult,
    spy_summary: dict[str, float],
    spy_total_return: float,
    ew_summary: dict[str, float],
    ew_total_return: float,
) -> pd.DataFrame:
    metrics = [
        "cagr",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "turnover",
        "exposure_pct",
        "tail_ratio",
        "beta_to_spy",
        "information_ratio",
    ]
    rows = []
    for metric in metrics:
        b = baseline.test_summary.get(metric, float("nan"))
        a = aggressive.test_summary.get(metric, float("nan"))
        rows.append(
            {
                "metric": metric,
                "baseline_test": b,
                "aggressive_v2_test": a,
                "delta": a - b,
                "spy_buy_hold_test": spy_summary.get(metric, float("nan")),
                "ew_universe_buy_hold_test": ew_summary.get(metric, float("nan")),
            }
        )

    rows.extend(
        [
            {
                "metric": "total_return",
                "baseline_test": baseline.test_total_return,
                "aggressive_v2_test": aggressive.test_total_return,
                "delta": aggressive.test_total_return - baseline.test_total_return,
                "spy_buy_hold_test": spy_total_return,
                "ew_universe_buy_hold_test": ew_total_return,
            },
            {
                "metric": "full_period_total_return",
                "baseline_test": baseline.full_total_return,
                "aggressive_v2_test": aggressive.full_total_return,
                "delta": aggressive.full_total_return - baseline.full_total_return,
                "spy_buy_hold_test": float("nan"),
                "ew_universe_buy_hold_test": float("nan"),
            },
        ]
    )
    return pd.DataFrame(rows)


def _plot_key_metrics(comparison_df: pd.DataFrame, output_path: Path) -> None:
    selected = comparison_df.loc[
        comparison_df["metric"].isin(["total_return", "cagr", "sharpe", "max_drawdown", "exposure_pct"])
    ].copy()
    selected["metric"] = selected["metric"].map(
        {
            "total_return": "Test total return",
            "cagr": "Test CAGR",
            "sharpe": "Test Sharpe",
            "max_drawdown": "Test max drawdown",
            "exposure_pct": "Test exposure",
        }
    )

    fig = make_subplots(rows=1, cols=1)
    fig.add_bar(name="Baseline", x=selected["metric"], y=selected["baseline_test"])
    fig.add_bar(name="Aggressive v2", x=selected["metric"], y=selected["aggressive_v2_test"])
    fig.add_bar(name="SPY buy & hold", x=selected["metric"], y=selected["spy_buy_hold_test"])
    fig.add_bar(name="EW universe buy & hold", x=selected["metric"], y=selected["ew_universe_buy_hold_test"])
    fig.update_layout(
        barmode="group",
        title="Baseline vs Aggressive v2 - Key Test Metrics",
        yaxis_title="Value",
    )
    _save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs aggressive-v2 comparison.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/v2_comparison"),
        help="Directory for comparison artifacts.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = make_synthetic_market_data()
    baseline_artifacts, baseline_result = _run_profile(profile_name="baseline", data=data)
    aggressive_artifacts, aggressive_result = _run_profile(
        profile_name="aggressive_v2",
        data=data,
        portfolio_overrides={
            "max_turnover_per_rebalance": 0.80,
            "max_sector_weight": 0.45,
            "min_dollar_volume": 1_000_000.0,
            "target_vol_lower": 0.16,
            "target_vol_upper": 0.22,
        },
        allocator_kwargs={
            "downside_penalty": 0.12,
            "cost_penalty": 0.05,
            "require_positive_scores": False,
            "selection_quantile": 0.0,
            "score_power": 1.6,
            "min_total_exposure": 0.75,
        },
    )

    spy_summary, spy_total_return = _buy_and_hold_report(data[0], SystemConfig().benchmark_symbol, SystemConfig())
    ew_summary, ew_total_return = _equal_weight_universe_report(data[0], data[1], SystemConfig())
    comparison_df = _build_comparison_df(
        baseline_result, aggressive_result, spy_summary, spy_total_return, ew_summary, ew_total_return
    )
    comparison_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    baseline_curve = _to_equity_series(baseline_artifacts, "baseline")
    aggressive_curve = _to_equity_series(aggressive_artifacts, "aggressive_v2")
    equity_df = pd.concat([baseline_curve, aggressive_curve], axis=1)
    equity_df.to_csv(out_dir / "equity_curves.csv")

    baseline_dd = baseline_curve / baseline_curve.cummax() - 1.0
    aggressive_dd = aggressive_curve / aggressive_curve.cummax() - 1.0

    fig_equity = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Equity", "Drawdown")
    )
    fig_equity.add_trace(go.Scatter(x=equity_df.index, y=equity_df["baseline"], name="Baseline"), row=1, col=1)
    fig_equity.add_trace(
        go.Scatter(x=equity_df.index, y=equity_df["aggressive_v2"], name="Aggressive v2"), row=1, col=1
    )
    fig_equity.add_trace(go.Scatter(x=baseline_dd.index, y=baseline_dd, name="Baseline DD"), row=2, col=1)
    fig_equity.add_trace(
        go.Scatter(x=aggressive_dd.index, y=aggressive_dd, name="Aggressive v2 DD"), row=2, col=1
    )
    fig_equity.update_layout(title="Baseline vs Aggressive v2 - Equity and Drawdown")
    _save_figure(fig_equity, out_dir / "equity_drawdown_comparison.png")

    exposure_df = pd.concat(
        [
            _to_exposure_series(baseline_artifacts, "baseline"),
            _to_exposure_series(aggressive_artifacts, "aggressive_v2"),
        ],
        axis=1,
    ).dropna(how="all")
    exposure_df.to_csv(out_dir / "exposure_series.csv")
    fig_exposure = go.Figure()
    fig_exposure.add_trace(go.Scatter(x=exposure_df.index, y=exposure_df["baseline"], name="Baseline exposure"))
    fig_exposure.add_trace(
        go.Scatter(x=exposure_df.index, y=exposure_df["aggressive_v2"], name="Aggressive v2 exposure")
    )
    fig_exposure.update_layout(
        title="Baseline vs Aggressive v2 - Exposure Utilization",
        yaxis_title="Exposure (0-1)",
        xaxis_title="Time",
    )
    _save_figure(fig_exposure, out_dir / "exposure_comparison.png")

    _plot_key_metrics(comparison_df, out_dir / "key_metrics_comparison.png")

    payload = {
        "baseline": asdict(baseline_result),
        "aggressive_v2": asdict(aggressive_result),
        "spy_buy_hold_test": {"summary": spy_summary, "total_return": spy_total_return},
        "ew_universe_buy_hold_test": {"summary": ew_summary, "total_return": ew_total_return},
        "delta_test_total_return": aggressive_result.test_total_return - baseline_result.test_total_return,
        "delta_test_sharpe": aggressive_result.test_summary.get("sharpe", float("nan"))
        - baseline_result.test_summary.get("sharpe", float("nan")),
        "delta_test_exposure": aggressive_result.test_summary.get("exposure_pct", float("nan"))
        - baseline_result.test_summary.get("exposure_pct", float("nan")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print("Saved comparison outputs to", out_dir)
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
