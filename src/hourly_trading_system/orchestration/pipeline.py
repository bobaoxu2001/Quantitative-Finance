"""Orchestration layer for research-to-production workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from hourly_trading_system.backtest import BacktestArtifacts, HourlyBacktestEngine
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.data import AdapterRegistry


@dataclass(slots=True)
class PipelineContext:
    run_id: str
    started_at: datetime
    config: SystemConfig
    mode: str  # research | paper | live


class TradingPipeline:
    """
    Production orchestration coordinator.

    In production this class should be invoked by scheduler/orchestrator
    (Airflow/Prefect/K8s CronJob) and wired to OMS/EMS gateways.
    """

    def __init__(
        self,
        context: PipelineContext,
        adapters: AdapterRegistry,
        engine: HourlyBacktestEngine,
    ) -> None:
        self.context = context
        self.adapters = adapters
        self.engine = engine

    def load_data(
        self,
        start: datetime,
        end: datetime,
        symbols: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        return self.adapters.load_all(start=start, end=end, symbols=symbols)

    def run_research_backtest(self, start: str, end: str) -> BacktestArtifacts:
        payload = self.load_data(
            start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
            end=pd.Timestamp(end, tz="UTC").to_pydatetime(),
            symbols=None,
        )
        required = {"market"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Missing required datasets for backtest: {sorted(missing)}")
        return self.engine.run(
            market=payload["market"],
            membership=payload.get("membership"),
            fundamentals=payload.get("fundamentals"),
            macro=payload.get("macro"),
            sentiment=payload.get("sentiment"),
            analyst=payload.get("analyst"),
            start=start,
            end=end,
        )

    def build_live_hour_payload(self, decision_time: pd.Timestamp) -> dict[str, Any]:
        """
        Build payload for live hourly inference.

        This method intentionally returns raw tables + metadata so production
        deployers can route them to a low-latency inference service.
        """
        payload = self.load_data(
            start=(decision_time - pd.Timedelta(days=30)).to_pydatetime(),
            end=decision_time.to_pydatetime(),
            symbols=None,
        )
        return {
            "decision_time": decision_time,
            "tables": payload,
            "run_id": self.context.run_id,
            "mode": self.context.mode,
        }
