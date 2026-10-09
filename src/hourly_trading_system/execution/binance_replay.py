"""Binance-specific execution report parsing and REST replay templates."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hourly_trading_system.live.contracts import FillEvent, OrderStatusEvent

from .broker_signers import BinanceSpotSigner


def _dt_ms(ms: int | float | str | None) -> datetime:
    if ms is None:
        return datetime.now(UTC)
    try:
        value = int(float(ms))
    except Exception:
        return datetime.now(UTC)
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _average_price(quote_qty: Any, base_qty: Any) -> float | None:
    """Average fill price = cumulative quote / cumulative base quantity."""
    base = _safe_float(base_qty)
    if base <= 0.0:
        return None
    return _safe_float(quote_qty) / base


def parse_binance_execution_report(message: dict[str, Any]) -> OrderStatusEvent | None:
    """
    Parse Binance user data stream `executionReport` event.

    Official-like fields:
      e,E,s,c,S,o,f,q,p,X,x,i,l,z,L,n,N,t,T
    """
    if str(message.get("e")) != "executionReport":
        return None
    symbol = str(message.get("s", ""))
    if not symbol:
        return None

    order_id = str(message.get("i", ""))
    if not order_id:
        return None

    last_exec_qty = float(message.get("l", 0.0) or 0.0)
    last_exec_price = float(message.get("L", 0.0) or 0.0)
    fee = float(message.get("n", 0.0) or 0.0)
    fee_asset = message.get("N")
    fill = None
    if last_exec_qty > 0 and last_exec_price > 0:
        fill = FillEvent(
            order_id=order_id,
            symbol=symbol,
            side=str(message.get("S", "")),
            quantity=last_exec_qty,
            fill_price=last_exec_price,
            fees=fee,
            execution_id=str(message.get("t", "")),
            fill_timestamp=_dt_ms(message.get("T")),
            venue="binance",
            metadata={"fee_asset": fee_asset},
        )

    return OrderStatusEvent(
        order_id=order_id,
        status=str(message.get("X", "UNKNOWN")),
        symbol=symbol,
        side=str(message.get("S", "")),
        requested_quantity=float(message.get("q", 0.0) or 0.0),
        filled_quantity=float(message.get("z", 0.0) or 0.0),
        remaining_quantity=max(float(message.get("q", 0.0) or 0.0) - float(message.get("z", 0.0) or 0.0), 0.0),
        average_fill_price=_average_price(message.get("Z"), message.get("z")),
        client_order_id=str(message.get("c", "")),
        reason=str(message.get("r", "")),
        broker_timestamp=_dt_ms(message.get("E")),
        last_fill=fill,
        metadata={
            "execution_type": message.get("x"),
            "order_type": message.get("o"),
            "time_in_force": message.get("f"),
        },
    )


@dataclass(slots=True)
class BinanceExecutionReplayClient:
    """
    Replay template based on official Binance signed REST endpoints.

    Uses:
      - GET /api/v3/allOrders
      - GET /api/v3/myTrades
    and reconstructs `executionReport`-like updates by time interval.
    """

    rest_base_url: str
    signer: BinanceSpotSigner
    timeout_seconds: int = 8

    def _signed_get(self, path: str, params: dict[str, Any]) -> Any:
        signed_params = self.signer.sign_params(params)
        # Must serialise in the exact order that was signed (dict order).
        query = urllib.parse.urlencode(list(signed_params.items()))
        url = f"{self.rest_base_url.rstrip('/')}{path}?{query}"
        request = urllib.request.Request(
            url=url,
            method="GET",
            headers=self.signer.headers(),
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            text = response.read().decode("utf-8")
        return json.loads(text) if text else []

    def replay_symbol_window(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[OrderStatusEvent]:
        orders = self._signed_get(
            "/api/v3/allOrders",
            {
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": min(max(limit, 1), 1000),
            },
        )
        trades = self._signed_get(
            "/api/v3/myTrades",
            {
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": min(max(limit, 1), 1000),
            },
        )
        trades_by_order: dict[str, list[dict[str, Any]]] = {}
        for trade in trades:
            oid = str(trade.get("orderId", ""))
            if not oid:
                continue
            trades_by_order.setdefault(oid, []).append(trade)

        events: list[OrderStatusEvent] = []
        for order in orders:
            oid = str(order.get("orderId", ""))
            if not oid:
                continue
            side = str(order.get("side", ""))
            qty = float(order.get("origQty", 0.0) or 0.0)
            executed = float(order.get("executedQty", 0.0) or 0.0)
            status = str(order.get("status", "UNKNOWN"))
            avg_price = _average_price(order.get("cummulativeQuoteQty"), executed)
            order_trades = sorted(trades_by_order.get(oid, []), key=lambda x: x.get("time", 0))
            # Emit one event per trade so that every execution in the window is
            # replayed; downstream reconciliation dedupes by execution_id.
            fills = [
                FillEvent(
                    order_id=oid,
                    symbol=symbol,
                    side=side,
                    quantity=_safe_float(trade.get("qty")),
                    fill_price=_safe_float(trade.get("price")),
                    fees=_safe_float(trade.get("commission")),
                    execution_id=str(trade.get("id", "")),
                    fill_timestamp=_dt_ms(trade.get("time")),
                    venue="binance",
                    metadata={"fee_asset": trade.get("commissionAsset")},
                )
                for trade in order_trades
            ]
            for fill in fills or [None]:
                events.append(
                    OrderStatusEvent(
                        order_id=oid,
                        status=status,
                        symbol=symbol,
                        side=side,
                        requested_quantity=qty,
                        filled_quantity=executed,
                        remaining_quantity=max(qty - executed, 0.0),
                        average_fill_price=avg_price,
                        client_order_id=str(order.get("clientOrderId", "")),
                        reason="",
                        broker_timestamp=_dt_ms(order.get("updateTime")),
                        last_fill=fill,
                    )
                )
        return events
