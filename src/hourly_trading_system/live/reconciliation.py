"""Order status reconciliation and fill-based portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .contracts import FillEvent, OrderAck, OrderRequest, OrderStatusEvent


@dataclass(slots=True)
class OrderLedgerEntry:
    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    requested_quantity: float
    accepted: bool
    status: str
    submitted_at: datetime
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class ReconciliationReport:
    """Consistency report between expected and broker-accounted positions."""

    position_breaks: dict[str, float]
    cash_break: float
    unresolved_orders: list[str]
    duplicate_fill_count: int
    total_fees: float

    @property
    def has_breaks(self) -> bool:
        return bool(self.position_breaks) or abs(self.cash_break) > 1e-8 or bool(self.unresolved_orders)


@dataclass(slots=True)
class FillReconciler:
    """Maintain broker-accounted holdings from status/fill events."""

    positions: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    order_ledger: dict[str, OrderLedgerEntry] = field(default_factory=dict)
    processed_execution_ids: set[str] = field(default_factory=set)
    duplicate_fill_count: int = 0
    total_fees: float = 0.0

    def register_submissions(
        self,
        requests: list[OrderRequest],
        acknowledgements: list[OrderAck],
    ) -> None:
        request_map = {req.client_order_id: req for req in requests if req.client_order_id}
        for ack in acknowledgements:
            req = request_map.get(ack.client_order_id)
            if req is None:
                continue
            self.order_ledger[ack.order_id] = OrderLedgerEntry(
                order_id=ack.order_id,
                client_order_id=ack.client_order_id,
                symbol=req.symbol,
                side=req.side,
                requested_quantity=req.quantity,
                accepted=ack.accepted,
                status=ack.status,
                submitted_at=ack.broker_timestamp,
                reason=ack.reason,
            )

    def apply_fill(self, fill: FillEvent) -> bool:
        """Apply one fill and return True if newly processed."""
        if fill.execution_id and fill.execution_id in self.processed_execution_ids:
            self.duplicate_fill_count += 1
            return False
        if fill.execution_id:
            self.processed_execution_ids.add(fill.execution_id)

        sign = 1.0 if fill.side.upper() == "BUY" else -1.0
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + sign * fill.quantity
        self.cash -= sign * fill.quantity * fill.fill_price
        self.cash -= float(fill.fees)
        self.total_fees += float(fill.fees)
        self.positions = {k: v for k, v in self.positions.items() if abs(v) > 1e-12}
        return True

    def apply_status(self, status_event: OrderStatusEvent) -> None:
        entry = self.order_ledger.get(status_event.order_id)
        if entry is not None:
            entry.status = status_event.status
            if status_event.filled_quantity is not None:
                entry.filled_quantity = float(status_event.filled_quantity)
            if status_event.average_fill_price is not None:
                entry.average_fill_price = float(status_event.average_fill_price)
            if status_event.reason:
                entry.reason = status_event.reason
        if status_event.last_fill is not None:
            self.apply_fill(status_event.last_fill)

    def unresolved_orders(self) -> list[str]:
        terminal = {"filled", "cancelled", "rejected", "expired"}
        return [
            order_id
            for order_id, entry in self.order_ledger.items()
            if entry.accepted and entry.status.lower() not in terminal
        ]

    def reconcile_expected(
        self,
        expected_positions: dict[str, float],
        expected_cash: float,
        qty_tolerance: float = 1e-6,
        cash_tolerance: float = 1e-4,
    ) -> ReconciliationReport:
        symbols = sorted(set(expected_positions.keys()).union(self.positions.keys()))
        breaks: dict[str, float] = {}
        for symbol in symbols:
            delta = self.positions.get(symbol, 0.0) - expected_positions.get(symbol, 0.0)
            if abs(delta) > qty_tolerance:
                breaks[symbol] = delta
        cash_break = self.cash - expected_cash
        if abs(cash_break) <= cash_tolerance:
            cash_break = 0.0
        return ReconciliationReport(
            position_breaks=breaks,
            cash_break=cash_break,
            unresolved_orders=self.unresolved_orders(),
            duplicate_fill_count=self.duplicate_fill_count,
            total_fees=self.total_fees,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "positions": self.positions,
            "cash": self.cash,
            "open_orders": len(self.unresolved_orders()),
            "duplicate_fill_count": self.duplicate_fill_count,
            "total_fees": self.total_fees,
        }
