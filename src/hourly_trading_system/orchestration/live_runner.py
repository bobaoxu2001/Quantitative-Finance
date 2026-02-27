"""Live trading runner integrating model governance and OMS/EMS routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import EMSClient, OMSClient, PaperOMSClient
from hourly_trading_system.execution.broker_gateway import OrderBatchResult
from hourly_trading_system.governance import (
    CanaryPolicy,
    ModelRegistry,
    deterministic_canary_assignment,
)
from hourly_trading_system.live.alerting import AlertRouter
from hourly_trading_system.live.contracts import AlertSeverity, OrderRequest, QueueMessage, now_utc
from hourly_trading_system.live.realtime_queue import BaseRealtimeQueue, InMemoryRealtimeQueue
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController


@dataclass(slots=True)
class LiveRunResult:
    """Result of one live decision cycle."""

    decision_time: pd.Timestamp
    used_champion: bool
    used_canary: bool
    canary_allocation: float
    orders_sent: int
    orders_rejected: int
    risk_regime: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LiveTradingRunner:
    """
    Production live orchestrator.

    Workflow:
      1) Load champion/canary model versions from registry.
      2) Score incoming feature snapshot.
      3) Build target portfolio with risk controls.
      4) Submit orders to OMS/EMS.
      5) Emit events to queue and alerts.
    """

    def __init__(
        self,
        config: SystemConfig,
        registry: ModelRegistry,
        allocator: HourlyPortfolioAllocator,
        risk_controller: RiskController,
        alert_router: AlertRouter | None = None,
        realtime_queue: BaseRealtimeQueue | None = None,
        oms_client: OMSClient | None = None,
        ems_client: EMSClient | None = None,
        strategy_id: str = "hourly_system_live",
    ) -> None:
        self.config = config
        self.registry = registry
        self.allocator = allocator
        self.risk_controller = risk_controller
        self.alert_router = alert_router or AlertRouter.with_console_and_file("outputs/live_alerts.jsonl")
        self.realtime_queue = realtime_queue or InMemoryRealtimeQueue()
        self.oms_client = oms_client or PaperOMSClient(audit_path="outputs/paper_oms_audit.jsonl")
        self.ems_client = ems_client
        self.strategy_id = strategy_id
        self.current_positions: dict[str, float] = {}
        self.cash: float = float(config.initial_capital)
        self.canary_policy = CanaryPolicy()

    def _publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        self.realtime_queue.publish(QueueMessage(topic=topic, payload=payload))

    def _active_models(self) -> tuple[object, object | None, float]:
        state = self.registry.deployment_state()
        if not state.champion_version:
            raise RuntimeError("No champion model deployed in model registry.")
        champion = self.registry.load_model(state.champion_version)
        canary = self.registry.load_model(state.canary_version) if state.canary_version else None
        return champion, canary, state.canary_allocation

    def _score(
        self,
        features: pd.DataFrame,
        decision_time: pd.Timestamp,
    ) -> tuple[pd.DataFrame, bool, bool, float]:
        champion_model, canary_model, canary_allocation = self._active_models()
        pred_champion = champion_model.predict(features).to_frame()
        used_champion = True
        used_canary = False

        if canary_model is None or canary_allocation <= 0:
            return pred_champion, used_champion, used_canary, 0.0

        pred_canary = canary_model.predict(features).to_frame()
        pred_champion = pred_champion.rename(
            columns={
                "expected_excess_return": "expected_excess_return_champion",
                "downside_probability": "downside_probability_champion",
                "model_name": "model_name_champion",
            }
        )
        pred_canary = pred_canary.rename(
            columns={
                "expected_excess_return": "expected_excess_return_canary",
                "downside_probability": "downside_probability_canary",
                "model_name": "model_name_canary",
            }
        )
        merged = pred_champion.merge(pred_canary, on=["symbol", "event_time"], how="inner")
        if merged.empty:
            return champion_model.predict(features).to_frame(), used_champion, used_canary, 0.0

        choose_canary = merged["symbol"].map(
            lambda sym: deterministic_canary_assignment(
                key=f"{decision_time.isoformat()}:{sym}",
                canary_allocation=canary_allocation,
            )
        )
        merged["expected_excess_return"] = merged["expected_excess_return_champion"]
        merged["downside_probability"] = merged["downside_probability_champion"]
        merged.loc[choose_canary, "expected_excess_return"] = merged.loc[
            choose_canary, "expected_excess_return_canary"
        ]
        merged.loc[choose_canary, "downside_probability"] = merged.loc[
            choose_canary, "downside_probability_canary"
        ]
        merged["model_name"] = merged["model_name_champion"]
        merged.loc[choose_canary, "model_name"] = merged.loc[choose_canary, "model_name_canary"]
        used_canary = bool(choose_canary.any())
        return (
            merged[["symbol", "event_time", "expected_excess_return", "downside_probability", "model_name"]].copy(),
            used_champion,
            used_canary,
            canary_allocation,
        )

    def _current_weights(self, market_snapshot: pd.DataFrame, equity: float) -> pd.Series:
        prices = market_snapshot.set_index("symbol")["close"].to_dict()
        notionals = {sym: qty * float(prices.get(sym, 0.0)) for sym, qty in self.current_positions.items()}
        if equity <= 0:
            return pd.Series(dtype=float)
        return pd.Series({sym: notional / equity for sym, notional in notionals.items() if abs(notional) > 0}, dtype=float)

    def _build_orders(
        self,
        target_weights: pd.Series,
        market_snapshot: pd.DataFrame,
        equity: float,
    ) -> list[OrderRequest]:
        prices = market_snapshot.set_index("symbol")["close"].to_dict()
        current_notional = {
            sym: qty * float(prices.get(sym, 0.0)) for sym, qty in self.current_positions.items()
        }
        symbols = sorted(set(target_weights.index).union(current_notional.keys()))
        orders: list[OrderRequest] = []
        for symbol in symbols:
            price = float(prices.get(symbol, 0.0))
            if price <= 0:
                continue
            target_notional = float(target_weights.get(symbol, 0.0) * equity)
            current = float(current_notional.get(symbol, 0.0))
            delta = target_notional - current
            qty = abs(delta / price)
            if qty <= 1e-8:
                continue
            side = "BUY" if delta > 0 else "SELL"
            orders.append(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type="MKT",
                    tif="DAY",
                    strategy_id=self.strategy_id,
                    metadata={"target_weight": float(target_weights.get(symbol, 0.0))},
                )
            )
        return orders

    def _apply_fills_approximate(
        self,
        orders: list[OrderRequest],
        market_snapshot: pd.DataFrame,
    ) -> None:
        """Update local positions approximation after accepted market orders."""
        prices = market_snapshot.set_index("symbol")["close"].to_dict()
        for order in orders:
            px = float(prices.get(order.symbol, 0.0))
            if px <= 0:
                continue
            signed_qty = order.quantity if order.side.upper() == "BUY" else -order.quantity
            self.current_positions[order.symbol] = self.current_positions.get(order.symbol, 0.0) + signed_qty
            self.cash -= signed_qty * px
        self.current_positions = {k: v for k, v in self.current_positions.items() if abs(v) > 1e-10}

    def _send_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        if not orders:
            return OrderBatchResult(acks=[], rejected=[])
        if self.ems_client is not None:
            return self.ems_client.route_orders(orders)
        return self.oms_client.submit_orders(orders)

    def run_hour(
        self,
        decision_time: pd.Timestamp,
        features: pd.DataFrame,
        market_snapshot: pd.DataFrame,
        benchmark_return: float | None = None,
    ) -> LiveRunResult:
        """Run one live decision cycle."""
        decision_time = pd.Timestamp(decision_time)
        if decision_time.tzinfo is None:
            decision_time = decision_time.tz_localize("UTC")
        else:
            decision_time = decision_time.tz_convert("UTC")

        required_market_cols = {"symbol", "close", "volume", "sector", "rv_20h"}
        missing = sorted(required_market_cols - set(market_snapshot.columns))
        if missing:
            raise ValueError(f"market_snapshot missing required columns: {missing}")
        if not self.oms_client.health_check() or (self.ems_client is not None and not self.ems_client.health_check()):
            self.alert_router.critical(
                source="live_runner",
                message="OMS/EMS health check failed. Orders not sent.",
                details={"decision_time": decision_time.isoformat()},
            )
            self._publish_event(
                "alerts",
                {
                    "severity": str(AlertSeverity.CRITICAL),
                    "source": "live_runner",
                    "message": "OMS/EMS health check failed",
                },
            )
            return LiveRunResult(
                decision_time=decision_time,
                used_champion=False,
                used_canary=False,
                canary_allocation=0.0,
                orders_sent=0,
                orders_rejected=0,
                risk_regime="halted",
                metadata={"halt_reason": "oms_ems_unavailable"},
            )

        prices = market_snapshot.set_index("symbol")["close"].to_dict()
        invested = sum(self.current_positions.get(sym, 0.0) * float(px) for sym, px in prices.items())
        equity = float(self.cash + invested)
        if equity <= 0:
            raise RuntimeError("Equity is non-positive. Live runner cannot proceed.")

        risk_state = self.risk_controller.update(equity, benchmark_return=benchmark_return)
        prediction_frame, used_champion, used_canary, canary_allocation = self._score(
            features=features,
            decision_time=decision_time,
        )

        current_weights = self._current_weights(market_snapshot=market_snapshot, equity=equity)
        decision = self.allocator.allocate(
            timestamp=decision_time,
            predictions=prediction_frame,
            market_snapshot=market_snapshot,
            current_weights=current_weights,
            risk_scaler=self.risk_controller.capital_scaler(),
        )
        orders = self._build_orders(
            target_weights=decision.weights,
            market_snapshot=market_snapshot,
            equity=equity,
        )
        batch = self._send_orders(orders)
        accepted_order_ids = {ack.order_id for ack in batch.acks if ack.accepted}
        accepted_orders = orders[: len(accepted_order_ids)] if accepted_order_ids else []
        self._apply_fills_approximate(accepted_orders, market_snapshot)

        payload = {
            "decision_time": decision_time.isoformat(),
            "equity": equity,
            "orders_sent": len(batch.acks),
            "orders_rejected": len(batch.rejected),
            "risk_regime": risk_state.regime,
            "used_canary": used_canary,
            "canary_allocation": canary_allocation,
        }
        self._publish_event("orders", payload)
        self._publish_event(
            "predictions",
            {
                "decision_time": decision_time.isoformat(),
                "predictions": prediction_frame.to_dict(orient="records"),
            },
        )
        self.alert_router.info(
            source="live_runner",
            message="Live cycle completed",
            details=payload,
        )
        return LiveRunResult(
            decision_time=decision_time,
            used_champion=used_champion,
            used_canary=used_canary,
            canary_allocation=canary_allocation,
            orders_sent=len(batch.acks),
            orders_rejected=len(batch.rejected),
            risk_regime=risk_state.regime,
            metadata={"timestamp": now_utc().isoformat()},
        )
