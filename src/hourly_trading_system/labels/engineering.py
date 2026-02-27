"""Label construction with execution-consistent horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(
    market: pd.DataFrame,
    benchmark_symbol: str = "SPY",
    horizon_hours: int = 5,
    downside_threshold: float = 0.015,
) -> pd.DataFrame:
    """
    Build forward labels:
      - 5h excess return relative to benchmark.
      - downside event probability target.

    For signal at t:
      entry at open(t+1), exit at open(t+1+horizon_hours)
    """
    if market.empty:
        return market.copy()

    cols = {"symbol", "event_time", "open", "low"}
    missing = sorted(cols - set(market.columns))
    if missing:
        raise ValueError(f"market data missing columns for labels: {missing}")

    df = market.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.sort_values(["symbol", "event_time"]).reset_index(drop=True)
    grouped = df.groupby("symbol", group_keys=False)

    entry = grouped["open"].shift(-1)
    exit_ = grouped["open"].shift(-(1 + horizon_hours))
    df["target_forward_return"] = np.log(exit_ / entry)

    # Worst intrahorizon low relative to entry (drawdown proxy).
    worst_low = grouped["low"].transform(
        lambda s: s.shift(-1).rolling(window=horizon_hours, min_periods=horizon_hours).min().shift(-(horizon_hours - 1))
    )
    intrahorizon_dd = worst_low / entry - 1.0
    df["target_downside_event"] = (intrahorizon_dd <= -downside_threshold).astype(float)

    bench = (
        df.loc[df["symbol"] == benchmark_symbol, ["event_time", "target_forward_return"]]
        .rename(columns={"target_forward_return": "benchmark_forward_return"})
        .drop_duplicates("event_time")
    )
    df = df.merge(bench, on="event_time", how="left")
    df["target_excess_return"] = df["target_forward_return"] - df["benchmark_forward_return"]

    return df[
        [
            "symbol",
            "event_time",
            "target_forward_return",
            "benchmark_forward_return",
            "target_excess_return",
            "target_downside_event",
        ]
    ].copy()


def train_validation_test_split(
    frame: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, pd.DataFrame]:
    """Split a frame into train/validation/test by event_time."""
    out = frame.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    train = out.loc[out["event_time"] <= pd.Timestamp(train_end, tz="UTC")]
    val = out.loc[
        (out["event_time"] >= pd.Timestamp(val_start, tz="UTC"))
        & (out["event_time"] <= pd.Timestamp(val_end, tz="UTC"))
    ]
    test = out.loc[
        (out["event_time"] >= pd.Timestamp(test_start, tz="UTC"))
        & (out["event_time"] <= pd.Timestamp(test_end, tz="UTC"))
    ]
    return {"train": train.copy(), "validation": val.copy(), "test": test.copy()}


def purge_overlapping_windows(
    frame: pd.DataFrame,
    horizon_hours: int = 5,
) -> pd.DataFrame:
    """
    Purge overlapping labels to reduce leakage risk in cross-validation.

    Keeps one sample every `horizon_hours` bars per symbol.
    """
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["symbol", "event_time"]).copy()
    ordered["row_id"] = ordered.groupby("symbol").cumcount()
    filtered = ordered.loc[(ordered["row_id"] % horizon_hours) == 0].drop(columns=["row_id"])
    return filtered.reset_index(drop=True)
