"""Real-time transaction cost attribution for live fills."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from hourly_trading_system.live.contracts import FillEvent, OrderAck, OrderRequest, OrderStatusEvent


@dataclass(slots=True)
class OrderArrivalContext:
    order_id: str
    symbol: str
    side: str
    requested_quantity: float
    arrival_price: float
    decision_time: datetime | None
    client_order_id: str | None


@dataclass(slots=True)
class TCARow:
    order_id: str
    symbol: str
    side: str
    quantity: float
    arrival_price: float
    fill_price: float
    fees: float
    shortfall_bps: float
    fee_bps: float
    total_cost_bps: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "arrival_price": self.arrival_price,
            "fill_price": self.fill_price,
            "fees": self.fees,
            "shortfall_bps": self.shortfall_bps,
            "fee_bps": self.fee_bps,
            "total_cost_bps": self.total_cost_bps,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class LiveTCAAttributor:
    """Compute per-fill and aggregate transaction cost attribution."""

    arrival_by_order: dict[str, OrderArrivalContext] = field(default_factory=dict)
    rows: list[TCARow] = field(default_factory=list)

    def register_submissions(self, requests: list[OrderRequest], acknowledgements: list[OrderAck]) -> None:
        req_map = {r.client_order_id: r for r in requests if r.client_order_id}
        for ack in acknowledgements:
            req = req_map.get(ack.client_order_id)
            if req is None:
                continue
            arrival_price = float(req.limit_price) if req.limit_price is not None else float(req.metadata.get("reference_price", 0.0))
            decision_time = None
            raw_dt = req.metadata.get("decision_time")
            if raw_dt is not None:
                try:
                    decision_time = pd.Timestamp(raw_dt).to_pydatetime()
                except Exception:
                    decision_time = None
            self.arrival_by_order[ack.order_id] = OrderArrivalContext(
                order_id=ack.order_id,
                symbol=req.symbol,
                side=req.side,
                requested_quantity=req.quantity,
                arrival_price=arrival_price,
                decision_time=decision_time,
                client_order_id=req.client_order_id,
            )

    def record_fill(self, fill: FillEvent) -> TCARow | None:
        ctx = self.arrival_by_order.get(fill.order_id)
        if ctx is None or ctx.arrival_price <= 0 or fill.quantity <= 0:
            return None
        side_sign = 1.0 if fill.side.upper() == "BUY" else -1.0
        shortfall_bps = ((fill.fill_price - ctx.arrival_price) * side_sign / ctx.arrival_price) * 1e4
        fee_bps = (fill.fees / max(fill.quantity * fill.fill_price, 1e-12)) * 1e4
        row = TCARow(
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            arrival_price=ctx.arrival_price,
            fill_price=fill.fill_price,
            fees=fill.fees,
            shortfall_bps=float(shortfall_bps),
            fee_bps=float(fee_bps),
            total_cost_bps=float(shortfall_bps + fee_bps),
            timestamp=fill.fill_timestamp,
        )
        self.rows.append(row)
        return row

    def on_status(self, status_event: OrderStatusEvent) -> TCARow | None:
        if status_event.last_fill is None:
            return None
        return self.record_fill(status_event.last_fill)

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(
                columns=[
                    "order_id",
                    "symbol",
                    "side",
                    "quantity",
                    "arrival_price",
                    "fill_price",
                    "fees",
                    "shortfall_bps",
                    "fee_bps",
                    "total_cost_bps",
                    "timestamp",
                ]
            )
        return pd.DataFrame([row.to_dict() for row in self.rows])

    def summary(self, last_n: int | None = None) -> dict[str, float]:
        frame = self.to_frame()
        if frame.empty:
            return {
                "trade_count": 0.0,
                "avg_shortfall_bps": 0.0,
                "avg_fee_bps": 0.0,
                "avg_total_cost_bps": 0.0,
                "p95_total_cost_bps": 0.0,
            }
        if last_n is not None:
            frame = frame.tail(last_n)
        return {
            "trade_count": float(len(frame)),
            "avg_shortfall_bps": float(frame["shortfall_bps"].mean()),
            "avg_fee_bps": float(frame["fee_bps"].mean()),
            "avg_total_cost_bps": float(frame["total_cost_bps"].mean()),
            "p95_total_cost_bps": float(np.quantile(frame["total_cost_bps"], 0.95)),
        }
