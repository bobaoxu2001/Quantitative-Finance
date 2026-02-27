"""Binance-specific execution report parsing and REST replay templates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any
import urllib.parse
import urllib.request

from hourly_trading_system.live.contracts import FillEvent, OrderStatusEvent

from .broker_signers import BinanceSpotSigner


def _dt_ms(ms: int | float | str | None) -> datetime:
    if ms is None:
        return datetime.now(timezone.utc)
    try:
        value = int(float(ms))
    except Exception:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


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
        average_fill_price=float(message.get("Z", 0.0) or 0.0) / max(float(message.get("z", 0.0) or 1.0), 1.0),
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
        query = urllib.parse.urlencode(signed_params, doseq=True)
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
            avg_price = float(order.get("cummulativeQuoteQty", 0.0) or 0.0) / max(executed, 1.0)
            fill = None
            order_trades = sorted(trades_by_order.get(oid, []), key=lambda x: x.get("time", 0))
            if order_trades:
                lt = order_trades[-1]
                fill = FillEvent(
                    order_id=oid,
                    symbol=symbol,
                    side=side,
                    quantity=float(lt.get("qty", 0.0) or 0.0),
                    fill_price=float(lt.get("price", 0.0) or 0.0),
                    fees=float(lt.get("commission", 0.0) or 0.0),
                    execution_id=str(lt.get("id", "")),
                    fill_timestamp=_dt_ms(lt.get("time")),
                    venue="binance",
                    metadata={"fee_asset": lt.get("commissionAsset")},
                )
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
