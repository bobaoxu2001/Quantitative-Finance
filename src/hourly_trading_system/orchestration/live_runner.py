"""Live trading runner integrating model governance and OMS/EMS routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import pandas as pd

from hourly_trading_system.analytics import LiveTCAAttributor
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import EMSClient, OMSClient, PaperOMSClient
from hourly_trading_system.execution.broker_gateway import OrderBatchResult
from hourly_trading_system.governance import CanaryPolicy, ModelRegistry, deterministic_canary_assignment
from hourly_trading_system.live.alerting import AlertRouter
from hourly_trading_system.live.contracts import AlertSeverity, OrderRequest, QueueMessage, now_utc
from hourly_trading_system.live.reconciliation import FillReconciler
from hourly_trading_system.live.realtime_queue import BaseRealtimeQueue, InMemoryRealtimeQueue
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import LiveSafetyGuard, RiskController


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
    """Production live orchestrator with safety guards and reconciliation."""

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
        safety_guard: LiveSafetyGuard | None = None,
        reconciler: FillReconciler | None = None,
        tca_attributor: LiveTCAAttributor | None = None,
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
        self.safety_guard = safety_guard or LiveSafetyGuard()
        self.reconciler = reconciler or FillReconciler(cash=float(config.initial_capital))
        self.tca_attributor = tca_attributor or LiveTCAAttributor()
        self.strategy_id = strategy_id
        self.current_positions: dict[str, float] = {}
        self.cash: float = float(config.initial_capital)
        self.expected_positions: dict[str, float] = {}
        self.expected_cash: float = float(config.initial_capital)
        self.last_trade_day: pd.Timestamp | None = None
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

    def _build_client_order_id(self, decision_time: pd.Timestamp, symbol: str, side: str, quantity: float) -> str:
        key = f"{self.strategy_id}|{decision_time.isoformat()}|{symbol}|{side}|{quantity:.6f}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return f"cid-{digest}"

    def _build_orders(
        self,
        decision_time: pd.Timestamp,
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
            if price <= 0.0:
                continue
            target_notional = float(target_weights.get(symbol, 0.0) * equity)
            current = float(current_notional.get(symbol, 0.0))
            delta = target_notional - current
            quantity = abs(delta / price)
            if quantity <= 1e-8:
                continue
            side = "BUY" if delta > 0 else "SELL"
            orders.append(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="MKT",
                    tif="DAY",
                    client_order_id=self._build_client_order_id(decision_time, symbol, side, quantity),
                    strategy_id=self.strategy_id,
                    metadata={
                        "target_weight": float(target_weights.get(symbol, 0.0)),
                        "reference_price": price,
                        "decision_time": decision_time.isoformat(),
                    },
                )
            )
        return orders

    @staticmethod
    def _apply_expected_fill(
        expected_positions: dict[str, float],
        expected_cash: float,
        order: OrderRequest,
    ) -> float:
        px = float(order.limit_price) if order.limit_price is not None else float(order.metadata.get("reference_price", 0.0))
        if px <= 0:
            return expected_cash
        signed = order.quantity if order.side.upper() == "BUY" else -order.quantity
        expected_positions[order.symbol] = expected_positions.get(order.symbol, 0.0) + signed
        if abs(expected_positions[order.symbol]) <= 1e-12:
            expected_positions.pop(order.symbol, None)
        return expected_cash - signed * px

    def _send_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        if not orders:
            return OrderBatchResult(acks=[], rejected=[], status_events=[])
        if self.ems_client is not None:
            return self.ems_client.route_orders(orders)
        return self.oms_client.submit_orders(orders)

    def _apply_status_updates(self) -> list[dict[str, Any]]:
        updates = self.oms_client.poll_order_updates(strategy_id=self.strategy_id, max_events=500)
        payloads: list[dict[str, Any]] = []
        tca_rows: list[dict[str, Any]] = []
        for event in updates:
            self.reconciler.apply_status(event)
            payloads.append(event.to_dict())
            tca_row = self.tca_attributor.on_status(event)
            if tca_row is not None:
                tca_rows.append(tca_row.to_dict())
            if event.last_fill is not None:
                self._publish_event("fills", event.last_fill.to_dict())
        if payloads:
            self._publish_event(
                "order_updates",
                {"count": len(payloads), "events": payloads},
            )
        if tca_rows:
            self._publish_event(
                "tca",
                {
                    "count": len(tca_rows),
                    "rows": tca_rows,
                    "summary": self.tca_attributor.summary(last_n=200),
                },
            )
        return payloads

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

        current_day = decision_time.normalize()
        is_new_day = self.last_trade_day is None or current_day != self.last_trade_day
        self.last_trade_day = current_day
        self.safety_guard.update_equity_state(equity=equity, is_new_day=is_new_day)
        if self.safety_guard.kill_switch_engaged:
            self.oms_client.cancel_all(self.strategy_id)
            return LiveRunResult(
                decision_time=decision_time,
                used_champion=False,
                used_canary=False,
                canary_allocation=0.0,
                orders_sent=0,
                orders_rejected=0,
                risk_regime="halted",
                metadata={"halt_reason": self.safety_guard.kill_switch_reason},
            )

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
        raw_orders = self._build_orders(
            decision_time=decision_time,
            target_weights=decision.weights,
            market_snapshot=market_snapshot,
            equity=equity,
        )
        guard_decision = self.safety_guard.pre_trade_check(raw_orders, reference_prices=prices)
        if not guard_decision.allowed:
            self.alert_router.warning(
                source="live_runner",
                message="Order cycle blocked by safety guard",
                details={"reason": guard_decision.reason, "decision_time": decision_time.isoformat()},
            )
            self._publish_event(
                "alerts",
                {
                    "severity": str(AlertSeverity.WARNING),
                    "source": "live_runner",
                    "message": "order_cycle_blocked",
                    "details": {"reason": guard_decision.reason},
                },
            )
            return LiveRunResult(
                decision_time=decision_time,
                used_champion=used_champion,
                used_canary=used_canary,
                canary_allocation=canary_allocation,
                orders_sent=0,
                orders_rejected=0,
                risk_regime=risk_state.regime,
                metadata={"guard_block_reason": guard_decision.reason},
            )

        orders = guard_decision.adjusted_orders
        try:
            batch = self._send_orders(orders)
            self.safety_guard.register_batch_outcome(
                accepted_count=batch.accepted_count(),
                rejected_count=batch.rejected_count(),
                had_exception=False,
            )
        except Exception as exc:
            self.safety_guard.register_batch_outcome(accepted_count=0, rejected_count=0, had_exception=True)
            self.alert_router.critical(
                source="live_runner",
                message="Order submission exception",
                details={"error": str(exc), "decision_time": decision_time.isoformat()},
            )
            return LiveRunResult(
                decision_time=decision_time,
                used_champion=used_champion,
                used_canary=used_canary,
                canary_allocation=canary_allocation,
                orders_sent=0,
                orders_rejected=len(orders),
                risk_regime="halted" if self.safety_guard.kill_switch_engaged else risk_state.regime,
                metadata={"exception": str(exc)},
            )

        self.reconciler.register_submissions(requests=orders, acknowledgements=batch.acks)
        self.tca_attributor.register_submissions(requests=orders, acknowledgements=batch.acks)
        accepted_cids = {ack.client_order_id for ack in batch.acks if ack.accepted and ack.client_order_id}
        for order in orders:
            if order.client_order_id in accepted_cids:
                self.expected_cash = self._apply_expected_fill(self.expected_positions, self.expected_cash, order)

        updates = self._apply_status_updates()
        for status in batch.status_events:
            self.reconciler.apply_status(status)

        report = self.reconciler.reconcile_expected(
            expected_positions=self.expected_positions,
            expected_cash=self.expected_cash,
        )
        self.safety_guard.evaluate_reconciliation(
            unresolved_orders=len(report.unresolved_orders),
            duplicate_fills=report.duplicate_fill_count,
        )
        self.current_positions = dict(self.reconciler.positions)
        self.cash = float(self.reconciler.cash)

        if report.has_breaks:
            self.alert_router.warning(
                source="live_runner",
                message="Reconciliation breaks detected",
                details={
                    "position_breaks": report.position_breaks,
                    "cash_break": report.cash_break,
                    "unresolved_orders": len(report.unresolved_orders),
                },
            )

        if self.safety_guard.kill_switch_engaged:
            self.oms_client.cancel_all(self.strategy_id)

        payload = {
            "decision_time": decision_time.isoformat(),
            "equity": equity,
            "orders_sent": len(batch.acks),
            "orders_rejected": len(batch.rejected),
            "risk_regime": risk_state.regime,
            "used_canary": used_canary,
            "canary_allocation": canary_allocation,
            "kill_switch": self.safety_guard.kill_switch_engaged,
            "reconciliation_breaks": report.has_breaks,
        }
        self._publish_event(
            "orders",
            {
                **payload,
                "acks": [ack.to_dict() for ack in batch.acks],
                "rejected": [ack.to_dict() for ack in batch.rejected],
            },
        )
        self._publish_event(
            "predictions",
            {
                "decision_time": decision_time.isoformat(),
                "predictions": prediction_frame.to_dict(orient="records"),
            },
        )
        self._publish_event(
            "reconciliation",
            {
                "decision_time": decision_time.isoformat(),
                "summary": {
                    "position_breaks": report.position_breaks,
                    "cash_break": report.cash_break,
                    "unresolved_orders": len(report.unresolved_orders),
                    "duplicate_fill_count": report.duplicate_fill_count,
                    "status_events": len(updates),
                    "total_fees": report.total_fees,
                },
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
            risk_regime="halted" if self.safety_guard.kill_switch_engaged else risk_state.regime,
            metadata={
                "timestamp": now_utc().isoformat(),
                "kill_switch_reason": self.safety_guard.kill_switch_reason,
                "reconciliation_breaks": report.has_breaks,
            },
        )
