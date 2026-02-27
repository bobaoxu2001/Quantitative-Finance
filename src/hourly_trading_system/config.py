"""System configuration objects and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BacktestWindow:
    train_start: str = "2023-01-01"
    train_end: str = "2025-10-31"
    val_start: str = "2025-11-01"
    val_end: str = "2025-12-31"
    test_start: str = "2026-01-01"
    test_end: str = "2026-02-26"


@dataclass(slots=True)
class CostModelConfig:
    slippage_bps: float = 1.0
    half_spread_bps: float = 0.5
    base_impact_eta: float = 8.0
    max_participation_rate: float = 0.05


@dataclass(slots=True)
class PortfolioConfig:
    max_position_weight: float = 0.10
    target_vol_lower: float = 0.15
    target_vol_upper: float = 0.20
    min_dollar_volume: float = 2_000_000.0
    max_turnover_per_rebalance: float = 0.25
    max_sector_weight: float = 0.30


@dataclass(slots=True)
class RiskConfig:
    max_drawdown_hard_stop: float = 0.20
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    beta_soft_limit: float = 1.10


@dataclass(slots=True)
class DataQualityConfig:
    max_missing_feature_fraction: float = 0.25
    max_data_latency_minutes: int = 10
    fail_on_timestamp_violation: bool = True


@dataclass(slots=True)
class SystemConfig:
    timezone: str = "America/New_York"
    initial_capital: float = 100_000.0
    benchmark_symbol: str = "SPY"
    backtest_window: BacktestWindow = field(default_factory=BacktestWindow)
    costs: CostModelConfig = field(default_factory=CostModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SystemConfig":
        return SystemConfig(
            timezone=payload.get("timezone", "America/New_York"),
            initial_capital=float(payload.get("initial_capital", 100_000.0)),
            benchmark_symbol=payload.get("benchmark_symbol", "SPY"),
            backtest_window=BacktestWindow(**payload.get("backtest_window", {})),
            costs=CostModelConfig(**payload.get("costs", {})),
            portfolio=PortfolioConfig(**payload.get("portfolio", {})),
            risk=RiskConfig(**payload.get("risk", {})),
            data_quality=DataQualityConfig(**payload.get("data_quality", {})),
        )


def load_config(path: str | Path) -> SystemConfig:
    """Load system configuration from YAML."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return SystemConfig.from_dict(payload)


def save_config(config: SystemConfig, path: str | Path) -> None:
    """Persist system configuration to YAML."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
