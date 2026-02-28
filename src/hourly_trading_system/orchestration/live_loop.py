"""Hourly live loop orchestration with market-calendar scheduling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import pandas as pd

from hourly_trading_system.orchestration.live_runner import LiveRunResult, LiveTradingRunner


@dataclass(slots=True)
class LiveLoopConfig:
    """Runtime configuration for the hourly live loop."""

    timezone: str = "UTC"
    run_only_market_hours: bool = True
    market_open_hour_utc: int = 14
    market_close_hour_utc: int = 20
    sleep_check_seconds: int = 20
    max_cycles: int | None = None
    stop_on_exception: bool = True


class LiveDataProvider(ABC):
    """Abstract provider for live features and market snapshots."""

    @abstractmethod
    def get_features(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        """Return feature frame with symbol/event_time/features schema."""

    @abstractmethod
    def get_market_snapshot(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        """Return market snapshot with required columns for live runner."""

    def get_benchmark_return(self, decision_time: pd.Timestamp) -> float | None:
        """Optional benchmark return for risk update."""
        return None


def is_market_hour(
    timestamp_utc: pd.Timestamp,
    market_open_hour_utc: int = 14,
    market_close_hour_utc: int = 20,
) -> bool:
    if timestamp_utc.dayofweek >= 5:
        return False
    return market_open_hour_utc <= timestamp_utc.hour <= market_close_hour_utc


def floor_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out.floor("h")


def next_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return floor_to_hour(ts) + pd.Timedelta(hours=1)


class HourlyLiveLoop:
    """Scheduling wrapper that repeatedly triggers LiveTradingRunner."""

    def __init__(
        self,
        runner: LiveTradingRunner,
        provider: LiveDataProvider,
        config: LiveLoopConfig | None = None,
    ) -> None:
        self.runner = runner
        self.provider = provider
        self.config = config or LiveLoopConfig()
        self._last_executed_hour: pd.Timestamp | None = None

    def run_once(self, decision_time: pd.Timestamp | None = None) -> LiveRunResult:
        dt = floor_to_hour(decision_time or pd.Timestamp.utcnow())
        features = self.provider.get_features(dt)
        market_snapshot = self.provider.get_market_snapshot(dt)
        bench = self.provider.get_benchmark_return(dt)
        result = self.runner.run_hour(
            decision_time=dt,
            features=features,
            market_snapshot=market_snapshot,
            benchmark_return=bench,
        )
        self._last_executed_hour = dt
        return result

    def _should_run(self, current_utc: pd.Timestamp) -> bool:
        hour = floor_to_hour(current_utc)
        if self._last_executed_hour is not None and hour <= self._last_executed_hour:
            return False
        if not self.config.run_only_market_hours:
            return True
        return is_market_hour(
            hour,
            market_open_hour_utc=self.config.market_open_hour_utc,
            market_close_hour_utc=self.config.market_close_hour_utc,
        )

    def run_forever(self) -> list[LiveRunResult]:
        results: list[LiveRunResult] = []
        cycles = 0
        while True:
            now_utc = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow().tz_convert("UTC")
            if self._should_run(now_utc):
                try:
                    result = self.run_once(floor_to_hour(now_utc))
                    results.append(result)
                    cycles += 1
                except Exception:
                    if self.config.stop_on_exception:
                        raise
            if self.config.max_cycles is not None and cycles >= self.config.max_cycles:
                break
            time.sleep(max(self.config.sleep_check_seconds, 1))
        return results
