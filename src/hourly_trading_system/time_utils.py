"""Datetime normalization helpers."""

from __future__ import annotations

import pandas as pd


def to_utc_timestamp(value: object) -> pd.Timestamp:
    """Normalize datetime-like values to UTC pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
