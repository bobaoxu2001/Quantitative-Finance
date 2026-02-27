"""Event-driven hourly backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hourly_trading_system.analytics.metrics import PerformanceReport, compute_performance_report
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution.engine import ExecutionEngine
from hourly_trading_system.features.library import build_feature_panel, required_features
from hourly_trading_system.labels.engineering import build_labels
from hourly_trading_system.models.base import BaseSignalModel
from hourly_trading_system.portfolio.optimizer import HourlyPortfolioAllocator
from hourly_trading_system.risk.controls import RiskController
from hourly_trading_system.time_utils import to_utc_timestamp


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

    def _fit_model(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        train_end: pd.Timestamp,
    ) -> None:
        dataset = feature_panel.merge(label_panel, on=["symbol", "event_time"], how="inner")
        train = dataset.loc[dataset["event_time"] <= train_end].copy()
        train = train.dropna(subset=["target_excess_return", "target_downside_event"])
        if train.empty:
            # Fallback when configured train_end predates available observations.
            cutoff = max(int(len(dataset) * 0.6), 1)
            train = dataset.sort_values("event_time").iloc[:cutoff].dropna(
                subset=["target_excess_return", "target_downside_event"]
            )
        if train.empty:
            raise ValueError("No valid training rows after label alignment and filtering.")
        feature_cols = [col for col in required_features() if col in train.columns]
        self.model.fit(
            train[["symbol", "event_time", *feature_cols]],
            train["target_excess_return"],
            train["target_downside_event"],
        )

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
        if market.empty:
            raise ValueError("market data cannot be empty")
        if not {"symbol", "event_time", "open", "high", "low", "close", "volume"}.issubset(market.columns):
            raise ValueError("market data is missing required OHLCV columns")

        market = market.copy()
        market["event_time"] = pd.to_datetime(market["event_time"], utc=True)
        market = market.sort_values(["event_time", "symbol"]).reset_index(drop=True)

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
        fit_until = to_utc_timestamp(train_end) if train_end else to_utc_timestamp(self.config.backtest_window.train_end)
        self._fit_model(feature_panel, labels, train_end=fit_until)

        times = sorted(feature_panel["event_time"].unique())
        if start is not None:
            start_ts = to_utc_timestamp(start)
            times = [t for t in times if t >= start_ts]
        if end is not None:
            end_ts = to_utc_timestamp(end)
            times = [t for t in times if t <= end_ts]
        if len(times) < 2:
            raise ValueError("Backtest period requires at least 2 timestamps")

        positions: dict[str, float] = {}
        cash = float(self.config.initial_capital)
        equity = float(self.config.initial_capital)

        equity_records: list[dict[str, Any]] = []
        fill_records: list[dict[str, Any]] = []
        weight_records: list[dict[str, Any]] = []
        signal_records: list[pd.DataFrame] = []
        risk_records: list[dict[str, Any]] = []
        turnover_series: list[float] = []
        exposure_series: list[float] = []
        benchmark_returns: list[tuple[pd.Timestamp, float]] = []

        feature_cols = [col for col in required_features() if col in feature_panel.columns]
        for idx in range(len(times) - 1):
            t = to_utc_timestamp(times[idx])
            t_next = to_utc_timestamp(times[idx + 1])

            market_t = market.loc[market["event_time"] == t].copy()
            market_next = market.loc[market["event_time"] == t_next].copy()
            if market_t.empty or market_next.empty:
                continue

            if membership is not None and not membership.empty:
                active_symbols = self._active_membership(membership, t)
                if active_symbols:
                    market_t = market_t.loc[market_t["symbol"].isin(active_symbols)]
                    market_next = market_next.loc[market_next["symbol"].isin(active_symbols)]

            if market_t.empty or market_next.empty:
                continue

            marked = self.executor.mark_to_market(positions=positions, cash=cash, market_snapshot=market_t)
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

            features_t = feature_panel.loc[
                (feature_panel["event_time"] == t) & (feature_panel["symbol"].isin(market_t["symbol"]))
            ][["symbol", "event_time", *feature_cols]].copy()
            if features_t.empty:
                continue
            pred = self.model.predict(features_t).to_frame()
            signal_records.append(pred.assign(signal_time=t))

            rv_slice = feature_panel.loc[
                (feature_panel["event_time"] == t) & (feature_panel["symbol"].isin(market_t["symbol"])),
                ["symbol", "rv_20h"],
            ].copy()
            market_snapshot = market_t[["symbol", "close", "volume", "sector"]].merge(
                rv_slice, on="symbol", how="left"
            )
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

            marked_next = self.executor.mark_to_market(positions=positions, cash=cash, market_snapshot=market_next)
            equity_next = float(marked_next["equity"])
            period_return = (equity_next / equity - 1.0) if equity > 0 else 0.0
            spy_t = market_t.loc[market_t["symbol"] == self.config.benchmark_symbol, "close"]
            spy_next = market_next.loc[market_next["symbol"] == self.config.benchmark_symbol, "close"]
            bench_ret = float(spy_next.iloc[0] / spy_t.iloc[0] - 1.0) if len(spy_t) and len(spy_next) and spy_t.iloc[0] != 0 else 0.0
            self.risk_controller.register_return(period_return, benchmark_return=bench_ret)
            benchmark_returns.append((t_next, bench_ret))

            turnover_series.append(decision.turnover)
            exposure_series.append(1.0 - decision.cash_weight)
            for symbol, w in decision.weights.items():
                weight_records.append({"event_time": t, "symbol": symbol, "weight": float(w)})
            equity_records.append(
                {
                    "event_time": t_next,
                    "equity": equity_next,
                    "cash": marked_next["cash"],
                    "invested": marked_next["invested"],
                    "turnover": decision.turnover,
                    "exposure": 1.0 - decision.cash_weight,
                    "risk_regime": risk_state.regime,
                }
            )

        equity_curve = pd.DataFrame(equity_records)
        if equity_curve.empty:
            equity_curve = pd.DataFrame(
                [{"event_time": times[0], "equity": self.config.initial_capital, "cash": self.config.initial_capital, "invested": 0.0}]
            )

        fills = pd.DataFrame(fill_records)
        weights = pd.DataFrame(weight_records)
        signals = pd.concat(signal_records, ignore_index=True) if signal_records else pd.DataFrame()
        risk_hist = pd.DataFrame(risk_records)

        bench_series = (
            pd.Series({t: r for t, r in benchmark_returns}, name="benchmark_return").sort_index()
            if benchmark_returns
            else None
        )
        perf = compute_performance_report(
            equity_curve=equity_curve[["event_time", "equity", "cash", "invested"]],
            benchmark_returns=bench_series,
            turnover_series=pd.Series(turnover_series),
            exposure_series=pd.Series(exposure_series),
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
                "n_periods": len(equity_curve),
                "n_fills": len(fills),
            },
        )
