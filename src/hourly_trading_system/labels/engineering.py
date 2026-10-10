"""Label construction with execution-consistent horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hourly_trading_system.time_utils import to_utc_period_end, to_utc_timestamp


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

    ``label_end_time`` is the timestamp of the exit bar. Any sample whose
    ``label_end_time`` is later than a training cutoff uses prices from after
    that cutoff and must be purged from the training set.
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
    # Keep the label undefined (NaN) when the horizon is incomplete instead of
    # silently labelling the tail of the sample as "no downside event".
    df["target_downside_event"] = np.where(
        intrahorizon_dd.notna(),
        (intrahorizon_dd <= -downside_threshold).astype(float),
        np.nan,
    )
    df["label_end_time"] = grouped["event_time"].shift(-(1 + horizon_hours))

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
            "label_end_time",
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
    """
    Split a frame into train/validation/test by event_time.

    Date-only end bounds are inclusive of the whole day. When the frame carries
    ``label_end_time`` (see :func:`build_labels`), samples whose label horizon
    reaches past the end of their split are purged so that no split is trained
    or scored on prices that belong to the next split.
    """
    out = frame.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    train_end_ts = to_utc_period_end(train_end)
    val_end_ts = to_utc_period_end(val_end)
    test_end_ts = to_utc_period_end(test_end)
    train = out.loc[out["event_time"] <= train_end_ts]
    val = out.loc[
        (out["event_time"] >= to_utc_timestamp(val_start))
        & (out["event_time"] <= val_end_ts)
    ]
    test = out.loc[
        (out["event_time"] >= to_utc_timestamp(test_start))
        & (out["event_time"] <= test_end_ts)
    ]
    if "label_end_time" in out.columns:
        train = purge_label_overlap(train, train_end_ts)
        val = purge_label_overlap(val, val_end_ts)
        test = purge_label_overlap(test, test_end_ts)
    return {"train": train.copy(), "validation": val.copy(), "test": test.copy()}


def purge_label_overlap(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Drop samples whose forward label is not fully observable by ``cutoff``.

    Rows with an unknown ``label_end_time`` (tail of the sample) are dropped too,
    because their labels are undefined.
    """
    if frame.empty or "label_end_time" not in frame.columns:
        return frame.copy()
    label_end = pd.to_datetime(frame["label_end_time"], utc=True)
    return frame.loc[label_end.notna() & (label_end <= cutoff)].copy()


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
