"""Event-driven hourly backtest engine."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hourly_trading_system.analytics.metrics import PerformanceReport, compute_performance_report
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution.engine import ExecutionEngine
from hourly_trading_system.features.library import build_feature_panel, required_features
from hourly_trading_system.labels.engineering import build_labels, purge_label_overlap
from hourly_trading_system.models.base import BaseSignalModel
from hourly_trading_system.portfolio.optimizer import HourlyPortfolioAllocator
from hourly_trading_system.risk.controls import RiskController
from hourly_trading_system.time_utils import to_utc_period_end, to_utc_timestamp


@dataclass(slots=True)
class BacktestArtifacts:
    performance: PerformanceReport
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    weight_history: pd.DataFrame
    signal_history: pd.DataFrame
    risk_history: pd.DataFrame
    metadata: dict[str, Any]


class HourlyBacktestEngine:
    """End-to-end event-driven simulation for hourly strategy research."""

    def __init__(
        self,
        config: SystemConfig,
        model: BaseSignalModel,
        allocator: HourlyPortfolioAllocator,
        executor: ExecutionEngine,
        risk_controller: RiskController,
    ) -> None:
        self.config = config
        self.model = model
        self.allocator = allocator
        self.executor = executor
        self.risk_controller = risk_controller

    @staticmethod
    def _active_membership(membership: pd.DataFrame, decision_time: pd.Timestamp) -> set[str]:
        if membership is None or membership.empty:
            return set()
        m = membership.copy()
        m["known_time"] = pd.to_datetime(m["known_time"], utc=True)
        active = m.loc[m["known_time"] < decision_time].sort_values("known_time")
        if active.empty:
            return set()
        latest = active.groupby("symbol", as_index=False).tail(1)
        if "is_member" not in latest.columns:
            return set(latest["symbol"])
        return set(latest.loc[latest["is_member"].astype(bool), "symbol"])

    @staticmethod
    def _membership_timeline(
        membership: pd.DataFrame | None,
        times: list[pd.Timestamp],
    ) -> dict[pd.Timestamp, set[str]] | None:
        """
        Point-in-time membership for every decision time in one ordered pass.

        Equivalent to calling :meth:`_active_membership` per timestamp (records
        must be known strictly before the decision time) but O(M + T).
        """
        if membership is None or membership.empty:
            return None
        events = membership.copy()
        events["known_time"] = pd.to_datetime(events["known_time"], utc=True)
        events = events.sort_values("known_time", kind="stable")
        has_flag = "is_member" in events.columns
        symbols = events["symbol"].to_numpy()
        known = events["known_time"].dt.tz_localize(None).to_numpy()
        flags = events["is_member"].astype(bool).to_numpy() if has_flag else None

        state: dict[str, bool] = {}
        timeline: dict[pd.Timestamp, set[str]] = {}
        pointer = 0
        for t in times:
            t64 = to_utc_timestamp(t).tz_localize(None).to_datetime64()
            while pointer < len(events) and known[pointer] < t64:
                state[symbols[pointer]] = bool(flags[pointer]) if flags is not None else True
                pointer += 1
            timeline[t] = {sym for sym, active in state.items() if active}
        return timeline

    def _fit_model(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        train_end: pd.Timestamp,
    ) -> int:
        """
        Fit the model on samples whose labels are fully observable by ``train_end``.

        Samples are purged when their forward-label window (``label_end_time``)
        extends past ``train_end``; otherwise prices from the evaluation period
        would leak into training targets. Returns the number of training rows.
        """
        dataset = feature_panel.merge(label_panel, on=["symbol", "event_time"], how="inner")
        train = dataset.loc[dataset["event_time"] <= train_end].copy()
        train = train.dropna(subset=["target_excess_return", "target_downside_event"])
        train = purge_label_overlap(train, train_end)
        if train.empty:
            raise ValueError(
                "No valid training rows at or before train_end "
                f"({train_end.isoformat()}) after label purging. Provide more history "
                "before the backtest window or move train_end later."
            )
        feature_cols = [col for col in required_features() if col in train.columns]
        self.model.fit(
            train[["symbol", "event_time", *feature_cols]],
            train["target_excess_return"],
            train["target_downside_event"],
        )
        return len(train)

    def run(
        self,
        market: pd.DataFrame,
        membership: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        analyst: pd.DataFrame | None = None,
        train_end: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestArtifacts:
        """
        Run an event-driven hourly simulation.

        Timing: signal at bar ``t`` (close), execution at the open of bar
        ``t+1``, mark-to-market at the close of ``t+1``. Date-only ``end`` and
        ``train_end`` bounds include the whole day. ``membership`` restricts the
        *investable* universe for new positions; existing positions in names that
        leave the universe are still priced and liquidated from full market data.
        """
        if market.empty:
            raise ValueError("market data cannot be empty")
        if not {"symbol", "event_time", "open", "high", "low", "close", "volume"}.issubset(market.columns):
            raise ValueError("market data is missing required OHLCV columns")

        market = market.copy()
        market["event_time"] = pd.to_datetime(market["event_time"], utc=True)
        market = market.sort_values(["event_time", "symbol"]).reset_index(drop=True)
        if "sector" not in market.columns:
            market["sector"] = "UNKNOWN"

        feature_panel = build_feature_panel(
            market=market,
            fundamentals=fundamentals,
            macro=macro,
            sentiment=sentiment,
            analyst=analyst,
            benchmark_symbol=self.config.benchmark_symbol,
        )
        labels = build_labels(
            market=market,
            benchmark_symbol=self.config.benchmark_symbol,
            horizon_hours=5,
            downside_threshold=0.015,
        )
        fit_until = to_utc_period_end(train_end or self.config.backtest_window.train_end)

        times = sorted(pd.Timestamp(t) for t in feature_panel["event_time"].unique())
        if start is not None:
            start_ts = to_utc_timestamp(start)
            times = [t for t in times if t >= start_ts]
        if end is not None:
            end_ts = to_utc_period_end(end)
            times = [t for t in times if t <= end_ts]
        if len(times) < 2:
            raise ValueError("Backtest period requires at least 2 timestamps")
        in_sample_overlap = bool(fit_until >= times[0])
        if in_sample_overlap:
            warnings.warn(
                f"train_end ({fit_until.isoformat()}) is not before the first backtest "
                f"timestamp ({times[0].isoformat()}); results are in-sample.",
                stacklevel=2,
            )
        n_train_rows = self._fit_model(feature_panel, labels, train_end=fit_until)

        feature_cols = [col for col in required_features() if col in feature_panel.columns]
        market_by_time = {t: frame for t, frame in market.groupby("event_time", sort=True)}
        features_by_time = {
            t: frame[["symbol", "event_time", "rv_20h", *[c for c in feature_cols if c != "rv_20h"]]]
            for t, frame in feature_panel.groupby("event_time", sort=True)
        }
        universe_by_time = self._membership_timeline(membership, times)
        benchmark = self.config.benchmark_symbol

        positions: dict[str, float] = {}
        last_close: dict[str, float] = {}
        cash = float(self.config.initial_capital)
        equity = float(self.config.initial_capital)

        equity_records: list[dict[str, Any]] = [
            {
                "event_time": times[0],
                "equity": equity,
                "cash": cash,
                "invested": 0.0,
                "turnover": 0.0,
                "exposure": 0.0,
                "target_exposure": 0.0,
                "risk_regime": self.risk_controller.regime,
            }
        ]
        fill_records: list[dict[str, Any]] = []
        weight_records: list[dict[str, Any]] = []
        signal_records: list[pd.DataFrame] = []
        risk_records: list[dict[str, Any]] = []
        turnover_series: list[float] = []
        exposure_series: list[float] = []
        benchmark_returns: list[tuple[pd.Timestamp, float]] = []

        for idx in range(len(times) - 1):
            t = times[idx]
            t_next = times[idx + 1]
            market_t = market_by_time.get(t)
            market_next = market_by_time.get(t_next)
            if market_t is None or market_next is None or market_t.empty or market_next.empty:
                continue
            last_close.update(zip(market_t["symbol"], market_t["close"].astype(float), strict=True))

            investable = market_t
            if universe_by_time is not None:
                active_symbols = universe_by_time.get(t, set())
                if active_symbols:
                    investable = market_t.loc[market_t["symbol"].isin(active_symbols)]

            marked = self.executor.mark_to_market(
                positions=positions, cash=cash, market_snapshot=market_t, fallback_prices=last_close
            )
            equity = float(marked["equity"])
            notionals = marked["notionals"]
            if equity <= 0:
                break
            current_weights = pd.Series({s: n / equity for s, n in notionals.items()}, dtype=float)

            risk_state = self.risk_controller.update(equity)
            risk_records.append(
                {
                    "event_time": t,
                    "equity": equity,
                    "drawdown": risk_state.drawdown,
                    "regime": risk_state.regime,
                    "realized_vol_annualized": risk_state.realized_vol_annualized,
                    "beta_to_spy": risk_state.beta_to_spy,
                    "var_95": risk_state.var_95,
                    "cvar_95": risk_state.cvar_95,
                }
            )

            features_all = features_by_time.get(t)
            features_t = (
                features_all.loc[features_all["symbol"].isin(investable["symbol"])]
                if features_all is not None
                else pd.DataFrame()
            )
            if not features_t.empty:
                pred = self.model.predict(features_t[["symbol", "event_time", *feature_cols]]).to_frame()
                signal_records.append(pred.assign(signal_time=t))
                market_snapshot = investable[["symbol", "close", "volume", "sector"]].merge(
                    features_t[["symbol", "rv_20h"]], on="symbol", how="left"
                )
            else:
                # No scoreable names: hand the allocator an empty signal so it
                # de-risks instead of silently holding stale positions.
                pred = pd.DataFrame(columns=["symbol", "expected_excess_return", "downside_probability"])
                market_snapshot = investable[["symbol", "close", "volume", "sector"]].assign(rv_20h=np.nan)

            decision = self.allocator.allocate(
                timestamp=t,
                predictions=pred,
                market_snapshot=market_snapshot,
                current_weights=current_weights,
                risk_scaler=self.risk_controller.capital_scaler(),
            )

            result = self.executor.execute_target_weights(
                timestamp=t_next,
                equity=equity,
                current_positions=positions,
                cash=cash,
                target_weights=decision.weights,
                market_next_open=market_next[["symbol", "open", "volume"]],
            )
            positions = result.positions
            cash = float(result.cash)
            for fill in result.fills:
                fill_records.append(
                    {
                        "symbol": fill.symbol,
                        "event_time": t_next,
                        "quantity": fill.quantity,
                        "fill_price": fill.fill_price,
                        "notional": fill.notional,
                        "trading_cost": fill.trading_cost,
                        "partial": fill.partial,
                    }
                )

            last_close.update(zip(market_next["symbol"], market_next["close"].astype(float), strict=True))
            marked_next = self.executor.mark_to_market(
                positions=positions, cash=cash, market_snapshot=market_next, fallback_prices=last_close
            )
            equity_next = float(marked_next["equity"])
            period_return = (equity_next / equity - 1.0) if equity > 0 else 0.0
            bench_ret = self._benchmark_return(market_t, market_next, benchmark)
            self.risk_controller.register_return(period_return, benchmark_return=bench_ret)
            benchmark_returns.append((t_next, bench_ret))

            realized_exposure = float(marked_next["invested"]) / equity_next if equity_next > 0 else 0.0
            turnover_series.append(decision.turnover)
            exposure_series.append(realized_exposure)
            for symbol, w in decision.weights.items():
                weight_records.append({"event_time": t, "symbol": symbol, "weight": float(w)})
            equity_records.append(
                {
                    "event_time": t_next,
                    "equity": equity_next,
                    "cash": marked_next["cash"],
                    "invested": marked_next["invested"],
                    "turnover": decision.turnover,
                    "exposure": realized_exposure,
                    "target_exposure": 1.0 - decision.cash_weight,
                    "risk_regime": risk_state.regime,
                }
            )

        equity_curve = pd.DataFrame(equity_records)
        fills = pd.DataFrame(fill_records)
        weights = pd.DataFrame(weight_records)
        signals = pd.concat(signal_records, ignore_index=True) if signal_records else pd.DataFrame()
        risk_hist = pd.DataFrame(risk_records)

        bench_series = (
            pd.Series(dict(benchmark_returns), name="benchmark_return").sort_index()
            if benchmark_returns
            else None
        )
        perf = compute_performance_report(
            equity_curve=equity_curve[["event_time", "equity", "cash", "invested"]],
            benchmark_returns=bench_series,
            turnover_series=pd.Series(turnover_series, dtype=float),
            exposure_series=pd.Series(exposure_series, dtype=float),
            trades=fills,
        )
        return BacktestArtifacts(
            performance=perf,
            equity_curve=equity_curve,
            fills=fills,
            weight_history=weights,
            signal_history=signals,
            risk_history=risk_hist,
            metadata={
                "model_name": self.model.model_name,
                "n_periods": len(equity_curve) - 1,
                "n_fills": len(fills),
                "n_train_rows": n_train_rows,
                "train_end": fit_until.isoformat(),
                "in_sample_overlap": in_sample_overlap,
            },
        )

    @staticmethod
    def _benchmark_return(market_t: pd.DataFrame, market_next: pd.DataFrame, benchmark: str) -> float:
        """Close-to-close benchmark return from the *unfiltered* market data."""
        prev = market_t.loc[market_t["symbol"] == benchmark, "close"]
        nxt = market_next.loc[market_next["symbol"] == benchmark, "close"]
        if prev.empty or nxt.empty or float(prev.iloc[0]) == 0.0:
            return 0.0
        return float(nxt.iloc[0] / prev.iloc[0] - 1.0)
