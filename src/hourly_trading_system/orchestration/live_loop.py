"""Hourly live loop orchestration with market-calendar scheduling."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import time as dt_time

import pandas as pd

from hourly_trading_system.orchestration.live_runner import LiveRunResult, LiveTradingRunner
from hourly_trading_system.time_utils import to_utc_timestamp


@dataclass(slots=True)
class LiveLoopConfig:
    """
    Runtime configuration for the hourly live loop.

    Market hours are evaluated in ``exchange_timezone`` local time (DST-aware):
    a decision hour runs when its local wall-clock time falls inside
    ``[session_open_local, session_close_local]`` on a weekday. Set
    ``exchange_timezone=None`` to fall back to the fixed UTC hour window
    (``market_open_hour_utc``..``market_close_hour_utc``), which drifts by one
    hour across daylight-saving transitions. Exchange holidays are not modelled.
    """

    timezone: str = "UTC"
    run_only_market_hours: bool = True
    market_open_hour_utc: int = 14
    market_close_hour_utc: int = 20
    exchange_timezone: str | None = "America/New_York"
    session_open_local: str = "09:30"
    session_close_local: str = "16:00"
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
    """Fixed-UTC-window check (legacy; not DST-aware)."""
    if timestamp_utc.dayofweek >= 5:
        return False
    return market_open_hour_utc <= timestamp_utc.hour <= market_close_hour_utc


def is_exchange_session(
    timestamp: pd.Timestamp,
    exchange_timezone: str = "America/New_York",
    session_open_local: str = "09:30",
    session_close_local: str = "16:00",
) -> bool:
    """Return True when ``timestamp`` falls inside the local weekday session (inclusive)."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(exchange_timezone)
    if local.dayofweek >= 5:
        return False
    open_t = dt_time.fromisoformat(session_open_local)
    close_t = dt_time.fromisoformat(session_close_local)
    return open_t <= local.time() <= close_t


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def floor_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return to_utc_timestamp(ts).floor("h")


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
        dt = floor_to_hour(decision_time if decision_time is not None else utc_now())
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
        if self.config.exchange_timezone:
            return is_exchange_session(
                hour,
                exchange_timezone=self.config.exchange_timezone,
                session_open_local=self.config.session_open_local,
                session_close_local=self.config.session_close_local,
            )
        return is_market_hour(
            hour,
            market_open_hour_utc=self.config.market_open_hour_utc,
            market_close_hour_utc=self.config.market_close_hour_utc,
        )

    def run_forever(self) -> list[LiveRunResult]:
        results: list[LiveRunResult] = []
        cycles = 0
        while True:
            now = utc_now()
            if self._should_run(now):
                hour = floor_to_hour(now)
                try:
                    result = self.run_once(hour)
                    results.append(result)
                    cycles += 1
                except Exception as exc:
                    # Mark the hour as attempted so a persistent failure is not
                    # retried every few seconds for the rest of the hour.
                    self._last_executed_hour = hour
                    self.runner.alert_router.critical(
                        source="live_loop",
                        message="Live cycle failed",
                        details={"decision_time": hour.isoformat(), "error": repr(exc)},
                    )
                    if self.config.stop_on_exception:
                        raise
            if self.config.max_cycles is not None and cycles >= self.config.max_cycles:
                break
            time.sleep(max(self.config.sleep_check_seconds, 1))
        return results
