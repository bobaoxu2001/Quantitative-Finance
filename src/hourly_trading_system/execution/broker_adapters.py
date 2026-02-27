"""Multi-broker OMS adapter templates with unified interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from hourly_trading_system.live.contracts import FillEvent, OrderAck, OrderRequest, OrderStatusEvent, now_utc
from .broker_signers import AlpacaSigner, BinanceSpotSigner, IBKRGatewaySigner

from .broker_gateway import (
    HMACRequestSigner,
    HTTPOMSClient,
    IdempotencyStore,
    OMSClient,
    OrderBatchResult,
    RetryPolicy,
)


class BrokerKind(StrEnum):
    ALPACA = "alpaca"
    BINANCE_SPOT = "binance_spot"
    IBKR_GATEWAY = "ibkr_gateway"


@dataclass(slots=True)
class BrokerConnectionConfig:
    broker: BrokerKind
    submit_url: str
    cancel_url: str
    health_url: str | None = None
    order_updates_url: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None
    timeout_seconds: int = 8
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    extra_headers: dict[str, str] | None = None


class BaseBrokerOMSAdapter(OMSClient):
    """Base template adapter wrapping HTTPOMSClient transport utilities."""

    def __init__(self, config: BrokerConnectionConfig) -> None:
        signer = None
        if config.api_key and config.api_secret:
            signer = HMACRequestSigner(
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.passphrase,
            )
        self.config = config
        self.static_headers = dict(config.extra_headers or {})
        self.http = HTTPOMSClient(
            submit_url=config.submit_url,
            cancel_url=config.cancel_url,
            health_url=config.health_url,
            order_updates_url=config.order_updates_url,
            headers=self.static_headers,
            signer=signer,
            retry_policy=config.retry_policy,
            idempotency_store=IdempotencyStore(),
            timeout_seconds=config.timeout_seconds,
        )

    def _build_order_payload(self, order: OrderRequest) -> dict[str, Any]:
        raise NotImplementedError

    def _prepare_order_payload(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def _extra_order_headers(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, str]:
        return {}

    def _parse_ack(self, response: dict[str, Any], order: OrderRequest) -> OrderAck:
        accepted = bool(response.get("accepted", True))
        status = str(response.get("status", "accepted" if accepted else "rejected"))
        return OrderAck(
            order_id=str(response.get("order_id", f"order-{uuid4().hex}")),
            accepted=accepted,
            status=status,
            client_order_id=order.client_order_id,
            reason=response.get("reason"),
            broker_timestamp=now_utc(),
        )

    def _request_order(self, order: OrderRequest) -> tuple[OrderAck | None, OrderAck | None]:
        idem_key = self.http._idempotency_key(order)
        if self.http.idempotency_store.exists(idem_key):
            reject = OrderAck(
                order_id=f"duplicate-{uuid4().hex[:10]}",
                accepted=False,
                status="duplicate",
                client_order_id=order.client_order_id,
                reason="idempotency_duplicate",
                broker_timestamp=now_utc(),
            )
            return None, reject
        payload = self._prepare_order_payload(order, self._build_order_payload(order))
        extra_headers = {"Idempotency-Key": idem_key}
        extra_headers.update(self._extra_order_headers(order, payload))
        try:
            response = self.http._request_json(
                method="POST",
                url=self.config.submit_url,
                payload=payload,
                extra_headers=extra_headers,
            )
        except Exception as exc:
            reject = OrderAck(
                order_id=f"error-{uuid4().hex[:10]}",
                accepted=False,
                status="error",
                client_order_id=order.client_order_id,
                reason=str(exc),
                broker_timestamp=now_utc(),
            )
            return None, reject
        self.http.idempotency_store.mark(idem_key)
        ack = self._parse_ack(response, order)
        if ack.accepted:
            return ack, None
        return None, ack

    def submit_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        accepted: list[OrderAck] = []
        rejected: list[OrderAck] = []
        for order in orders:
            ok, rej = self._request_order(order)
            if ok is not None:
                accepted.append(ok)
            if rej is not None:
                rejected.append(rej)
        return OrderBatchResult(acks=accepted, rejected=rejected)

    def cancel_all(self, strategy_id: str) -> None:
        self.http.cancel_all(strategy_id)

    def health_check(self) -> bool:
        return self.http.health_check()

    def poll_order_updates(self, strategy_id: str, max_events: int = 200) -> list[OrderStatusEvent]:
        return self.http.poll_order_updates(strategy_id=strategy_id, max_events=max_events)


class AlpacaOMSAdapter(BaseBrokerOMSAdapter):
    """Template adapter for Alpaca-style broker REST schema."""

    def __init__(self, config: BrokerConnectionConfig) -> None:
        super().__init__(config)
        self.alpaca_signer = None
        if config.api_key and config.api_secret:
            self.alpaca_signer = AlpacaSigner(api_key=config.api_key, api_secret=config.api_secret)

    def _build_order_payload(self, order: OrderRequest) -> dict[str, Any]:
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.lower(),
            "type": "limit" if order.limit_price is not None else "market",
            "time_in_force": order.tif.lower(),
            "client_order_id": order.client_order_id,
        }
        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        return payload

    def _extra_order_headers(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, str]:
        if self.alpaca_signer is None:
            return {}
        return {**self.alpaca_signer.headers(), "X-REQUEST-NONCE": self.alpaca_signer.nonce()}

    def _parse_ack(self, response: dict[str, Any], order: OrderRequest) -> OrderAck:
        status = str(response.get("status", "accepted"))
        accepted = status.lower() not in {"rejected", "canceled", "expired"}
        return OrderAck(
            order_id=str(response.get("id", response.get("order_id", f"alp-{uuid4().hex}"))),
            accepted=accepted,
            status=status,
            client_order_id=order.client_order_id,
            reason=response.get("reject_reason"),
            broker_timestamp=now_utc(),
        )


class BinanceSpotOMSAdapter(BaseBrokerOMSAdapter):
    """Template adapter for Binance-spot-like signed endpoints."""

    def __init__(self, config: BrokerConnectionConfig) -> None:
        super().__init__(config)
        self.binance_signer = None
        if config.api_key and config.api_secret:
            self.binance_signer = BinanceSpotSigner(api_key=config.api_key, api_secret=config.api_secret)

    def _build_order_payload(self, order: OrderRequest) -> dict[str, Any]:
        payload = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "type": "LIMIT" if order.limit_price is not None else "MARKET",
            "quantity": f"{order.quantity:.8f}",
            "newClientOrderId": order.client_order_id,
            "newOrderRespType": "RESULT",
        }
        if order.limit_price is not None:
            payload["timeInForce"] = "GTC"
            payload["price"] = f"{order.limit_price:.8f}"
        return payload

    def _prepare_order_payload(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, Any]:
        if self.binance_signer is None:
            return payload
        return self.binance_signer.sign_params(payload)

    def _extra_order_headers(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, str]:
        if self.binance_signer is None:
            return {}
        return self.binance_signer.headers()

    def _parse_ack(self, response: dict[str, Any], order: OrderRequest) -> OrderAck:
        status = str(response.get("status", "NEW"))
        accepted = status.upper() in {"NEW", "PARTIALLY_FILLED", "FILLED"}
        return OrderAck(
            order_id=str(response.get("orderId", f"bn-{uuid4().hex}")),
            accepted=accepted,
            status=status,
            client_order_id=order.client_order_id,
            reason=response.get("msg"),
            broker_timestamp=now_utc(),
        )


class IBKRGatewayOMSAdapter(BaseBrokerOMSAdapter):
    """Template adapter for IBKR Client Portal Gateway style routes."""

    def __init__(self, config: BrokerConnectionConfig) -> None:
        super().__init__(config)
        self.ibkr_signer = IBKRGatewaySigner(session_token=config.api_secret) if config.api_secret else None

    def _build_order_payload(self, order: OrderRequest) -> dict[str, Any]:
        ibkr_order = {
            "acctId": order.metadata.get("account_id", ""),
            "conid": order.metadata.get("conid", ""),
            "ticker": order.symbol,
            "secType": order.metadata.get("secType", "STK"),
            "orderType": "LMT" if order.limit_price is not None else "MKT",
            "side": order.side,
            "quantity": order.quantity,
            "tif": order.tif,
            "cOID": order.client_order_id,
            "outsideRTH": False,
        }
        if order.limit_price is not None:
            ibkr_order["price"] = order.limit_price
        return {"orders": [ibkr_order]}

    def _extra_order_headers(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, str]:
        if self.ibkr_signer is None:
            return {}
        return {**self.ibkr_signer.headers(), "X-IBKR-NONCE": self.ibkr_signer.nonce()}

    def _parse_ack(self, response: dict[str, Any], order: OrderRequest) -> OrderAck:
        # Gateway may return list of statuses.
        status_rows = response.get("orders") or response.get("responses") or []
        if isinstance(status_rows, list) and status_rows:
            first = status_rows[0]
        else:
            first = response
        status = str(first.get("status", "Submitted"))
        accepted = status.lower() not in {"rejected", "cancelled", "inactive"}
        return OrderAck(
            order_id=str(first.get("order_id", first.get("id", f"ibkr-{uuid4().hex}"))),
            accepted=accepted,
            status=status,
            client_order_id=order.client_order_id,
            reason=first.get("message"),
            broker_timestamp=now_utc(),
        )


def build_broker_oms_adapter(config: BrokerConnectionConfig) -> OMSClient:
    """Factory for broker-specific OMS template adapters."""
    if config.broker == BrokerKind.ALPACA:
        return AlpacaOMSAdapter(config)
    if config.broker == BrokerKind.BINANCE_SPOT:
        return BinanceSpotOMSAdapter(config)
    if config.broker == BrokerKind.IBKR_GATEWAY:
        return IBKRGatewayOMSAdapter(config)
    raise ValueError(f"Unsupported broker kind: {config.broker}")


def parse_ws_order_status_message(message: dict[str, Any]) -> OrderStatusEvent | None:
    """
    Parse generic broker WebSocket message to OrderStatusEvent when possible.

    This function is intentionally schema-tolerant for template usage.
    """
    event_type = str(message.get("event") or message.get("e") or "").lower()
    data = message.get("data", message)
    if event_type and "order" not in event_type and "execution" not in event_type and "fill" not in event_type:
        return None

    order_id = str(data.get("order_id") or data.get("id") or data.get("i") or "")
    if not order_id:
        return None
    fill = None
    filled_qty = data.get("filled_qty") or data.get("z")
    avg_price = data.get("avg_fill_price") or data.get("L")
    if filled_qty and avg_price:
        fill = FillEvent(
            order_id=order_id,
            symbol=str(data.get("symbol") or data.get("s") or ""),
            side=str(data.get("side") or data.get("S") or ""),
            quantity=float(filled_qty),
            fill_price=float(avg_price),
            fees=float(data.get("fees") or data.get("n") or 0.0),
            execution_id=str(data.get("execution_id") or data.get("t") or ""),
            fill_timestamp=now_utc(),
            venue=str(data.get("venue") or ""),
        )
    return OrderStatusEvent(
        order_id=order_id,
        status=str(data.get("status") or data.get("X") or "unknown"),
        symbol=str(data.get("symbol") or data.get("s") or ""),
        side=str(data.get("side") or data.get("S") or ""),
        requested_quantity=float(data.get("orig_qty") or data.get("q") or 0.0),
        filled_quantity=float(filled_qty or 0.0),
        remaining_quantity=float(data.get("remaining_qty") or data.get("Q") or 0.0),
        average_fill_price=float(avg_price or 0.0),
        client_order_id=str(data.get("client_order_id") or data.get("c") or ""),
        reason=str(data.get("reason") or ""),
        broker_timestamp=now_utc(),
        last_fill=fill,
    )
