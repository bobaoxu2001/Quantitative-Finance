"""Datetime normalization helpers."""

from __future__ import annotations

import re

import pandas as pd

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def to_utc_timestamp(value: object) -> pd.Timestamp:
    """Normalize datetime-like values to UTC pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_utc_period_end(value: object) -> pd.Timestamp:
    """
    Normalize an inclusive period end bound to a UTC timestamp.

    A date-only string such as ``"2026-02-26"`` is interpreted as the *whole*
    calendar day (``2026-02-26T23:59:59.999999999Z``) so that window ends behave
    inclusively, as users of ``train_end="2025-10-31"`` expect. Values that carry
    an explicit time component are returned unchanged.
    """
    ts = to_utc_timestamp(value)
    if isinstance(value, str) and _DATE_ONLY.match(value.strip()):
        return ts + pd.Timedelta(days=1) - pd.Timedelta(1, unit="ns")
    return ts
