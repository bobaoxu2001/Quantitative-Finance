from __future__ import annotations

import numpy as np
import pandas as pd

from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import PaperOMSClient
from hourly_trading_system.features import required_features
from hourly_trading_system.governance import ModelRegistry, RolloutManager
from hourly_trading_system.live import AlertRouter, InMemoryRealtimeQueue
from hourly_trading_system.models.base import PredictionFrame
from hourly_trading_system.orchestration import LiveTradingRunner
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController


class DummySignalModel:
    def __init__(self, alpha: float, downside: float, name: str) -> None:
        self.alpha = alpha
        self.downside = downside
        self.model_name = name

    def predict(self, features: pd.DataFrame) -> PredictionFrame:
        index = features.index
        return PredictionFrame(
            symbol=features["symbol"],
            event_time=pd.to_datetime(features["event_time"], utc=True),
            expected_excess_return=pd.Series(self.alpha, index=index),
            downside_probability=pd.Series(self.downside, index=index),
            model_name=self.model_name,
        )


def _register_approved(registry: ModelRegistry, model: object, metrics: dict[str, float]) -> str:
    record = registry.register_model(
        model_object=model,
        model_name="hourly_alpha",
        route="B",
        feature_schema=required_features(),
        metrics=metrics,
    )
    registry.submit_for_approval(record.version_id, submitted_by="research")
    registry.approve(record.version_id, approved_by="mrc")
    return record.version_id


def test_model_registry_canary_promote_and_rollback(tmp_path) -> None:
    registry = ModelRegistry(root_dir=tmp_path / "registry")
    champion_v = _register_approved(
        registry,
        DummySignalModel(alpha=0.03, downside=0.03, name="champion"),
        metrics={"sharpe": 1.2, "calmar": 0.8, "max_drawdown": 0.10},
    )
    registry.deploy_champion(champion_v)

    canary_v = _register_approved(
        registry,
        DummySignalModel(alpha=0.04, downside=0.025, name="canary"),
        metrics={"sharpe": 1.3, "calmar": 0.85, "max_drawdown": 0.09},
    )
    registry.deploy_canary(canary_v, allocation=0.2)

    manager = RolloutManager(registry=registry)
    idx = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    champion_returns = pd.Series(np.sin(np.arange(len(idx))) * 0.001 + 0.0001, index=idx)
    canary_returns = pd.Series(np.sin(np.arange(len(idx))) * 0.001 + 0.0005, index=idx)
    decision = manager.evaluate(champion_returns=champion_returns, canary_returns=canary_returns)
    assert decision.action in {"promote", "hold"}

    # Force rollback path directly to verify operational safety controls.
    state = registry.deployment_state()
    if state.previous_champion_version:
        rolled = registry.rollback(reason="manual_test_rollback")
        assert rolled.champion_version == state.previous_champion_version


def test_live_runner_sends_orders_with_registry_models(tmp_path) -> None:
    registry = ModelRegistry(root_dir=tmp_path / "registry")
    champion_v = _register_approved(
        registry,
        DummySignalModel(alpha=0.05, downside=0.02, name="champion"),
        metrics={"sharpe": 1.1, "calmar": 0.7, "max_drawdown": 0.11},
    )
    registry.deploy_champion(champion_v)

    canary_v = _register_approved(
        registry,
        DummySignalModel(alpha=0.06, downside=0.015, name="canary"),
        metrics={"sharpe": 1.2, "calmar": 0.75, "max_drawdown": 0.10},
    )
    registry.deploy_canary(canary_v, allocation=0.25)

    config = SystemConfig(initial_capital=100_000.0)
    allocator = HourlyPortfolioAllocator(config=config.portfolio)
    risk_controller = RiskController(
        max_drawdown=config.risk.max_drawdown_hard_stop,
        target_vol_low=config.portfolio.target_vol_lower,
        target_vol_high=config.portfolio.target_vol_upper,
    )
    queue = InMemoryRealtimeQueue()
    runner = LiveTradingRunner(
        config=config,
        registry=registry,
        allocator=allocator,
        risk_controller=risk_controller,
        alert_router=AlertRouter.with_console_and_file(tmp_path / "alerts.jsonl"),
        realtime_queue=queue,
        oms_client=PaperOMSClient(audit_path=tmp_path / "oms_audit.jsonl"),
    )

    decision_time = pd.Timestamp("2026-02-10 15:00:00+00:00")
    symbols = ["AAPL", "MSFT", "XOM", "JPM"]
    features = pd.DataFrame({"symbol": symbols, "event_time": [decision_time] * len(symbols)})
    for feature in required_features():
        features[feature] = 0.0

    market_snapshot = pd.DataFrame(
        {
            "symbol": symbols,
            "close": [180.0, 430.0, 105.0, 220.0],
            "volume": [1_200_000.0, 1_100_000.0, 2_200_000.0, 950_000.0],
            "sector": ["TECH", "TECH", "ENERGY", "FIN"],
            "rv_20h": [0.02, 0.018, 0.025, 0.022],
        }
    )
    result = runner.run_hour(
        decision_time=decision_time,
        features=features,
        market_snapshot=market_snapshot,
        benchmark_return=0.0,
    )
    assert result.orders_sent >= 1
    assert queue.size("orders") == 1
    assert queue.size("predictions") == 1
