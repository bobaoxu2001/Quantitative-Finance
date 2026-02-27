"""Demonstrate live workflow: registry approval, canary rollout, and order routing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hourly_trading_system.config import SystemConfig
from hourly_trading_system.features import required_features
from hourly_trading_system.governance import ModelRegistry
from hourly_trading_system.live import AlertRouter, FileBackedRealtimeQueue
from hourly_trading_system.models import FactorLightGBMBaseline
from hourly_trading_system.orchestration import LiveTradingRunner
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController
from run_demo_backtest import make_synthetic_market_data


def _train_model(
    market: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> FactorLightGBMBaseline:
    from hourly_trading_system.features import build_feature_panel
    from hourly_trading_system.labels import build_labels

    subset = market.loc[
        (market["event_time"] >= pd.Timestamp(train_start, tz="UTC"))
        & (market["event_time"] <= pd.Timestamp(train_end, tz="UTC"))
    ].copy()
    features = build_feature_panel(subset, benchmark_symbol="SPY")
    labels = build_labels(subset, benchmark_symbol="SPY")
    data = features.merge(labels, on=["symbol", "event_time"], how="inner").dropna(
        subset=["target_excess_return", "target_downside_event"]
    )
    model = FactorLightGBMBaseline(feature_columns=required_features())
    model.fit(
        data[["symbol", "event_time", *required_features()]],
        data["target_excess_return"],
        data["target_downside_event"],
    )
    return model


def main() -> None:
    market, _, _, _, _, _ = make_synthetic_market_data()
    market["event_time"] = pd.to_datetime(market["event_time"], utc=True)
    market = market.sort_values(["event_time", "symbol"])

    champion_model = _train_model(market, train_start="2025-06-01", train_end="2025-10-31")
    canary_model = _train_model(market, train_start="2025-07-01", train_end="2025-10-31")

    registry_dir = Path("outputs/model_registry")
    registry = ModelRegistry(root_dir=registry_dir)

    champion = registry.register_model(
        model_object=champion_model,
        model_name="hourly_alpha",
        route="B",
        feature_schema=required_features(),
        metrics={"sharpe": 1.2, "calmar": 0.7, "max_drawdown": 0.12},
        notes="Initial champion candidate.",
    )
    registry.submit_for_approval(champion.version_id, submitted_by="research")
    registry.approve(champion.version_id, approved_by="mrc")
    registry.deploy_champion(champion.version_id)

    canary = registry.register_model(
        model_object=canary_model,
        model_name="hourly_alpha",
        route="B",
        feature_schema=required_features(),
        metrics={"sharpe": 1.3, "calmar": 0.75, "max_drawdown": 0.11},
        notes="Canary candidate with adjusted hyperparameters.",
    )
    registry.submit_for_approval(canary.version_id, submitted_by="research")
    registry.approve(canary.version_id, approved_by="mrc")
    registry.deploy_canary(canary.version_id, allocation=0.15)

    config = SystemConfig()
    allocator = HourlyPortfolioAllocator(config=config.portfolio)
    risk_controller = RiskController(
        max_drawdown=config.risk.max_drawdown_hard_stop,
        target_vol_low=config.portfolio.target_vol_lower,
        target_vol_high=config.portfolio.target_vol_upper,
    )
    alert_router = AlertRouter.with_console_and_file("outputs/live_alerts.jsonl")
    queue = FileBackedRealtimeQueue("outputs/live_queue")
    runner = LiveTradingRunner(
        config=config,
        registry=registry,
        allocator=allocator,
        risk_controller=risk_controller,
        alert_router=alert_router,
        realtime_queue=queue,
    )

    decision_time = market["event_time"].iloc[-2]
    market_t = market.loc[market["event_time"] == decision_time].copy()
    from hourly_trading_system.features import build_feature_panel

    panel = build_feature_panel(
        market.loc[market["event_time"] <= decision_time].copy(),
        benchmark_symbol="SPY",
    )
    features_t = panel.loc[panel["event_time"] == decision_time, ["symbol", "event_time", *required_features()]]
    market_snapshot = panel.loc[
        panel["event_time"] == decision_time,
        ["symbol", "close", "volume", "sector", "rv_20h"],
    ].copy()

    result = runner.run_hour(
        decision_time=decision_time,
        features=features_t,
        market_snapshot=market_snapshot,
        benchmark_return=0.0,
    )
    print("Live cycle result:", result)
    print("Queue depth orders:", queue.size("orders"))
    print("Queue depth predictions:", queue.size("predictions"))


if __name__ == "__main__":
    main()
