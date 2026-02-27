"""Point-in-time data store and as-of join logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .contracts import enforce_information_set, validate_time_contract


@dataclass(slots=True)
class PITTable:
    name: str
    frame: pd.DataFrame


class PITStore:
    """In-memory point-in-time data store with strict anti-lookahead access."""

    def __init__(self) -> None:
        self.tables: dict[str, PITTable] = {}

    def register(self, name: str, frame: pd.DataFrame) -> None:
        validate_time_contract(frame, name)
        normalized = frame.copy()
        for col in ("event_time", "public_release_time", "ingest_time", "known_time"):
            normalized[col] = pd.to_datetime(normalized[col], utc=True)
        self.tables[name] = PITTable(name=name, frame=normalized.sort_values("known_time"))

    def get_snapshot(self, name: str, decision_time: datetime) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Unknown PIT table '{name}'")
        table = self.tables[name].frame
        snapshot = enforce_information_set(table, decision_time)
        return snapshot.copy()

    def latest_per_symbol(
        self,
        name: str,
        decision_time: datetime,
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        snapshot = self.get_snapshot(name, decision_time)
        if snapshot.empty:
            return snapshot
        if symbol_col not in snapshot.columns:
            return snapshot.tail(1).copy()
        return (
            snapshot.sort_values("known_time")
            .groupby(symbol_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    def asof_join(
        self,
        left: pd.DataFrame,
        right_name: str,
        left_time_col: str,
        right_by: str | None = "symbol",
        right_time_col: str = "known_time",
        suffix: str = "_r",
    ) -> pd.DataFrame:
        """
        Merge right PIT table onto left using as-of logic.

        Assumes left[left_time_col] is the decision timestamp and right records are
        only valid if right_time_col < left_time_col.
        """
        if right_name not in self.tables:
            raise KeyError(f"Unknown PIT table '{right_name}'")
        right = self.tables[right_name].frame.copy().sort_values(right_time_col)
        left_sorted = left.copy().sort_values(left_time_col)
        left_sorted[left_time_col] = pd.to_datetime(left_sorted[left_time_col], utc=True)
        right[right_time_col] = pd.to_datetime(right[right_time_col], utc=True)

        if right_by and right_by in left_sorted.columns and right_by in right.columns:
            merged = pd.merge_asof(
                left_sorted,
                right,
                left_on=left_time_col,
                right_on=right_time_col,
                by=right_by,
                direction="backward",
                allow_exact_matches=False,
                suffixes=("", suffix),
            )
        else:
            merged = pd.merge_asof(
                left_sorted,
                right,
                left_on=left_time_col,
                right_on=right_time_col,
                direction="backward",
                allow_exact_matches=False,
                suffixes=("", suffix),
            )
        return merged

    def tradable_universe(self, decision_time: datetime) -> set[str]:
        """Return dynamic S&P membership active at decision_time."""
        if "membership" not in self.tables:
            raise KeyError("PIT table 'membership' is required for dynamic universe.")
        membership = self.get_snapshot("membership", decision_time)
        if membership.empty:
            return set()
        if "is_member" not in membership.columns:
            raise ValueError("membership table must include is_member column.")
        active = membership.sort_values("known_time").groupby("symbol", as_index=False).tail(1)
        return set(active.loc[active["is_member"].astype(bool), "symbol"])
