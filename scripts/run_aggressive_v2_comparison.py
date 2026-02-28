#!/usr/bin/env python3
"""Run baseline vs aggressive-v2 backtest comparison on the demo dataset."""

from __future__ import annotations

import json
import argparse
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


def _test_only_report(artifacts: BacktestArtifacts, config: SystemConfig) -> tuple[dict[str, float], float]:
    curve = artifacts.equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    test_start = pd.Timestamp(config.backtest_window.test_start, tz="UTC")
    test_end = pd.Timestamp(config.backtest_window.test_end, tz="UTC")
    test_curve = curve.loc[(curve["event_time"] >= test_start) & (curve["event_time"] <= test_end)].copy()
    if len(test_curve) < 2:
        return {}, float("nan")

    test_report = compute_performance_report(
        equity_curve=test_curve[["event_time", "equity", "cash", "invested"]],
        turnover_series=test_curve["turnover"] if "turnover" in test_curve.columns else None,
        exposure_series=test_curve["exposure"] if "exposure" in test_curve.columns else None,
    )
    return test_report.summary, float(test_report.extras["total_return"])


def _run_profile(
    profile_name: str,
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
    market, membership, fundamentals, macro, sentiment, analyst = make_synthetic_market_data()

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

    test_summary, test_total_return = _test_only_report(artifacts, config)
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


def _build_comparison_df(baseline: ProfileResult, aggressive: ProfileResult) -> pd.DataFrame:
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
    ]
    rows = []
    for metric in metrics:
        b = baseline.test_summary.get(metric, float("nan"))
        a = aggressive.test_summary.get(metric, float("nan"))
        rows.append({"metric": metric, "baseline_test": b, "aggressive_v2_test": a, "delta": a - b})

    rows.extend(
        [
            {
                "metric": "total_return",
                "baseline_test": baseline.test_total_return,
                "aggressive_v2_test": aggressive.test_total_return,
                "delta": aggressive.test_total_return - baseline.test_total_return,
            },
            {
                "metric": "full_period_total_return",
                "baseline_test": baseline.full_total_return,
                "aggressive_v2_test": aggressive.full_total_return,
                "delta": aggressive.full_total_return - baseline.full_total_return,
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

    baseline_artifacts, baseline_result = _run_profile(profile_name="baseline")
    aggressive_artifacts, aggressive_result = _run_profile(
        profile_name="aggressive_v2",
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

    comparison_df = _build_comparison_df(baseline_result, aggressive_result)
    comparison_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    baseline_curve = _to_equity_series(baseline_artifacts, "baseline")
    aggressive_curve = _to_equity_series(aggressive_artifacts, "aggressive_v2")
    equity_df = pd.concat([baseline_curve, aggressive_curve], axis=1)
    equity_df.to_csv(out_dir / "equity_curves.csv")

    baseline_dd = baseline_curve / baseline_curve.cummax() - 1.0
    aggressive_dd = aggressive_curve / aggressive_curve.cummax() - 1.0

    fig_equity = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Equity", "Drawdown"))
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
        [_to_exposure_series(baseline_artifacts, "baseline"), _to_exposure_series(aggressive_artifacts, "aggressive_v2")],
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
