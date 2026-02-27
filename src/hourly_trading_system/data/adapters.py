"""Vendor-agnostic data adapter interfaces and local implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hourly_trading_system.time_utils import to_utc_timestamp

from .contracts import validate_time_contract


class BaseDataAdapter(ABC):
    """Abstract adapter for all provider-specific connectors."""

    @abstractmethod
    def load(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load point-in-time data from provider."""


class CSVAdapter(BaseDataAdapter):
    """Simple local CSV adapter for reproducible research and testing."""

    def __init__(self, csv_path: str | Path, table_name: str) -> None:
        self.csv_path = Path(csv_path)
        self.table_name = table_name

    def load(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing data file: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)
        validate_time_contract(frame, self.table_name)

        for col in ("event_time", "public_release_time", "ingest_time", "known_time"):
            frame[col] = pd.to_datetime(frame[col], utc=True)

        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame["event_time"] >= to_utc_timestamp(start)
        if end is not None:
            mask &= frame["event_time"] <= to_utc_timestamp(end)
        if symbols is not None and "symbol" in frame.columns:
            mask &= frame["symbol"].isin(symbols)
        return frame.loc[mask].copy()


class AdapterRegistry:
    """Central registry for all data adapters used by the pipeline."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseDataAdapter] = {}

    def register(self, name: str, adapter: BaseDataAdapter) -> None:
        self._adapters[name] = adapter

    def get(self, name: str) -> BaseDataAdapter:
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not registered.")
        return self._adapters[name]

    def load_all(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        return {
            name: adapter.load(start=start, end=end, symbols=symbols)
            for name, adapter in self._adapters.items()
        }

    def describe(self) -> dict[str, Any]:
        return {name: adapter.__class__.__name__ for name, adapter in self._adapters.items()}
