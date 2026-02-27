"""Point-in-time data contracts and anti-lookahead guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

REQUIRED_TIME_COLUMNS = [
    "event_time",
    "public_release_time",
    "ingest_time",
    "known_time",
]


def _ensure_datetime(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_datetime(out[col], utc=True)
    return out


def validate_time_contract(frame: pd.DataFrame, table_name: str) -> None:
    """Validate required PIT columns and temporal ordering constraints."""
    missing = [col for col in REQUIRED_TIME_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{table_name}: missing required columns {missing}")

    dtf = _ensure_datetime(frame, REQUIRED_TIME_COLUMNS)
    invalid_public = dtf["public_release_time"] < dtf["event_time"]
    invalid_ingest = dtf["ingest_time"] < dtf["public_release_time"]
    invalid_known = dtf["known_time"] < dtf["ingest_time"]
    if invalid_public.any() or invalid_ingest.any() or invalid_known.any():
        raise ValueError(
            f"{table_name}: invalid temporal ordering in records "
            f"(event<=public<=ingest<=known violated)."
        )


def enforce_information_set(frame: pd.DataFrame, decision_time: datetime) -> pd.DataFrame:
    """
    Return rows available in information set I(t).

    I(t) = all records publicly known and ingested strictly before decision_time.
    """
    if frame.empty:
        return frame.copy()
    decision_ts = pd.Timestamp(decision_time, tz="UTC")
    dtf = frame.copy()
    dtf["known_time"] = pd.to_datetime(dtf["known_time"], utc=True)
    return dtf.loc[dtf["known_time"] < decision_ts].copy()


@dataclass(slots=True)
class DataSnapshot:
    """Container for all as-of datasets required at a specific decision timestamp."""

    decision_time: datetime
    market: pd.DataFrame
    fundamentals: pd.DataFrame
    macro: pd.DataFrame
    sentiment: pd.DataFrame
    analyst: pd.DataFrame
    membership: pd.DataFrame

    def enforce(self) -> "DataSnapshot":
        """Apply information-set filter to all contained datasets."""
        return DataSnapshot(
            decision_time=self.decision_time,
            market=enforce_information_set(self.market, self.decision_time),
            fundamentals=enforce_information_set(self.fundamentals, self.decision_time),
            macro=enforce_information_set(self.macro, self.decision_time),
            sentiment=enforce_information_set(self.sentiment, self.decision_time),
            analyst=enforce_information_set(self.analyst, self.decision_time),
            membership=enforce_information_set(self.membership, self.decision_time),
        )
