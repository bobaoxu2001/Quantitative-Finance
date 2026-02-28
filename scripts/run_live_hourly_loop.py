"""Run scheduled hourly live loop (single cycle or bounded loop)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hourly_trading_system.features import build_feature_panel, required_features
from hourly_trading_system.orchestration import HourlyLiveLoop, LiveDataProvider, LiveLoopConfig
from run_demo_backtest import make_synthetic_market_data
from run_live_from_config_template import build_live_runner


def _to_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(slots=True)
class SyntheticProvider(LiveDataProvider):
    panel: pd.DataFrame

    def get_features(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        dt = _to_utc(decision_time)
        rows = self.panel.loc[self.panel["event_time"] == dt]
        if rows.empty:
            rows = self.panel.loc[self.panel["event_time"] <= dt].sort_values("event_time").groupby("symbol", as_index=False).tail(1)
        return rows[["symbol", "event_time", *required_features()]].copy()

    def get_market_snapshot(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        dt = _to_utc(decision_time)
        rows = self.panel.loc[self.panel["event_time"] == dt]
        if rows.empty:
            rows = self.panel.loc[self.panel["event_time"] <= dt].sort_values("event_time").groupby("symbol", as_index=False).tail(1)
        return rows[["symbol", "close", "volume", "sector", "rv_20h"]].copy()


@dataclass(slots=True)
class CSVSnapshotProvider(LiveDataProvider):
    features_df: pd.DataFrame
    market_df: pd.DataFrame

    def _slice(self, frame: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
        dt = _to_utc(decision_time)
        rows = frame.loc[frame["event_time"] == dt]
        if rows.empty:
            rows = frame.loc[frame["event_time"] <= dt].sort_values("event_time").groupby("symbol", as_index=False).tail(1)
        return rows.copy()

    def get_features(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        rows = self._slice(self.features_df, decision_time)
        for col in required_features():
            if col not in rows.columns:
                rows[col] = 0.0
        return rows[["symbol", "event_time", *required_features()]].copy()

    def get_market_snapshot(self, decision_time: pd.Timestamp) -> pd.DataFrame:
        rows = self._slice(self.market_df, decision_time)
        if "rv_20h" not in rows.columns:
            rows["rv_20h"] = 0.02
        if "sector" not in rows.columns:
            rows["sector"] = "UNKNOWN"
        return rows[["symbol", "close", "volume", "sector", "rv_20h"]].copy()


def _build_synthetic_provider() -> SyntheticProvider:
    market, _, _, _, _, _ = make_synthetic_market_data()
    market["event_time"] = pd.to_datetime(market["event_time"], utc=True)
    panel = build_feature_panel(market.sort_values(["event_time", "symbol"]), benchmark_symbol="SPY")
    return SyntheticProvider(panel=panel)


def _build_csv_provider(features_path: Path, market_path: Path) -> CSVSnapshotProvider:
    features = pd.read_csv(features_path)
    market = pd.read_csv(market_path)
    features["event_time"] = pd.to_datetime(features["event_time"], utc=True)
    market["event_time"] = pd.to_datetime(market["event_time"], utc=True)
    return CSVSnapshotProvider(features_df=features, market_df=market)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly live loop.")
    parser.add_argument("--mode", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    parser.add_argument("--decision-time", default=None, help="UTC timestamp for one-cycle run.")
    parser.add_argument("--max-cycles", type=int, default=3, help="Max cycles for loop mode.")
    parser.add_argument("--sleep-seconds", type=int, default=20, help="Loop poll sleep seconds.")
    parser.add_argument("--features-csv", default="outputs/features_live.csv")
    parser.add_argument("--market-csv", default="outputs/market_live.csv")
    args = parser.parse_args()

    runner = build_live_runner()
    if args.mode == "synthetic":
        provider = _build_synthetic_provider()
    else:
        provider = _build_csv_provider(Path(args.features_csv), Path(args.market_csv))

    loop = HourlyLiveLoop(
        runner=runner,
        provider=provider,
        config=LiveLoopConfig(
            run_only_market_hours=True,
            sleep_check_seconds=max(args.sleep_seconds, 1),
            max_cycles=max(args.max_cycles, 1),
        ),
    )

    if args.once:
        dt = _to_utc(args.decision_time) if args.decision_time else None
        result = loop.run_once(dt)
        print("Single-cycle result:", result)
        return

    results = loop.run_forever()
    print(f"Completed cycles: {len(results)}")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
