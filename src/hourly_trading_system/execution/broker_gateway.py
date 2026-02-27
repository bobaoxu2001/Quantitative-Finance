"""OMS/EMS integration interfaces for live execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import random
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

from hourly_trading_system.live.contracts import FillEvent, OrderAck, OrderRequest, OrderStatusEvent


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class OrderBatchResult:
    """Batch order send result."""

    acks: list[OrderAck]
    rejected: list[OrderAck]
    status_events: list[OrderStatusEvent] = field(default_factory=list)

    def accepted_count(self) -> int:
        return sum(1 for ack in self.acks if ack.accepted)

    def rejected_count(self) -> int:
        return len(self.rejected)


class OMSClient(ABC):
    """Order management interface."""

    @abstractmethod
    def submit_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        """Submit orders and return acknowledgements."""

    @abstractmethod
    def cancel_all(self, strategy_id: str) -> None:
        """Cancel open orders for strategy."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return OMS availability."""

    @abstractmethod
    def poll_order_updates(self, strategy_id: str, max_events: int = 200) -> list[OrderStatusEvent]:
        """Fetch incremental order status updates and optional fills."""


class EMSClient(ABC):
    """Execution management interface."""

    @abstractmethod
    def route_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        """Route child orders to destination venues."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return EMS availability."""


class PaperOMSClient(OMSClient):
    """Paper-trading OMS implementation with deterministic order IDs."""

    def __init__(self, audit_path: str | Path | None = None) -> None:
        self.audit_path = Path(audit_path) if audit_path else None
        self.open_orders: dict[str, dict[str, Any]] = {}
        self._pending_events: list[OrderStatusEvent] = []
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _audit(self, payload: dict[str, Any]) -> None:
        if not self.audit_path:
            return
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def submit_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        acks: list[OrderAck] = []
        rejected: list[OrderAck] = []
        for order in orders:
            if order.quantity <= 0:
                ack = OrderAck(
                    order_id=f"paper-{uuid4().hex}",
                    client_order_id=order.client_order_id,
                    accepted=False,
                    status="rejected",
                    reason="quantity_must_be_positive",
                    broker_timestamp=_now_utc(),
                )
                rejected.append(ack)
                self._audit({"type": "reject", "order": order.to_dict(), "ack": ack.to_dict()})
                continue
            order_id = f"paper-{uuid4().hex}"
            self.open_orders[order_id] = {
                "order": order.to_dict(),
                "status": "accepted",
                "timestamp": _now_utc().isoformat(),
            }
            ack = OrderAck(
                order_id=order_id,
                client_order_id=order.client_order_id,
                accepted=True,
                status="accepted",
                reason=None,
                broker_timestamp=_now_utc(),
            )
            acks.append(ack)
            self._audit({"type": "accept", "order": order.to_dict(), "ack": ack.to_dict()})

            fill_price = float(order.limit_price) if order.limit_price is not None else float(order.metadata.get("reference_price", 0.0))
            fill = FillEvent(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                fees=0.0,
                execution_id=f"paper-fill-{uuid4().hex}",
                fill_timestamp=_now_utc(),
                venue="paper",
            )
            self._pending_events.append(
                OrderStatusEvent(
                    order_id=order_id,
                    status="filled",
                    symbol=order.symbol,
                    side=order.side,
                    requested_quantity=order.quantity,
                    filled_quantity=order.quantity,
                    remaining_quantity=0.0,
                    average_fill_price=fill_price,
                    client_order_id=order.client_order_id,
                    broker_timestamp=_now_utc(),
                    last_fill=fill,
                )
            )
        return OrderBatchResult(acks=acks, rejected=rejected, status_events=[])

    def cancel_all(self, strategy_id: str) -> None:
        to_cancel = [
            order_id
            for order_id, payload in self.open_orders.items()
            if payload.get("order", {}).get("strategy_id") == strategy_id
        ]
        for order_id in to_cancel:
            self.open_orders[order_id]["status"] = "cancelled"
            self._audit({"type": "cancel", "order_id": order_id, "strategy_id": strategy_id})

    def health_check(self) -> bool:
        return True

    def poll_order_updates(self, strategy_id: str, max_events: int = 200) -> list[OrderStatusEvent]:
        n = min(max_events, len(self._pending_events))
        out = self._pending_events[:n]
        self._pending_events = self._pending_events[n:]
        return out


class HMACRequestSigner:
    """Generic HMAC-SHA256 signer for REST and WebSocket broker auth."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str | None = None) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.passphrase = passphrase

    def sign(self, method: str, path: str, timestamp: str, body: str = "") -> str:
        message = f"{timestamp}{method.upper()}{path}{body}".encode("utf-8")
        return hmac.new(self.api_secret, message, hashlib.sha256).hexdigest()

    def auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        headers = {
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": ts,
            "X-SIGNATURE": self.sign(method=method, path=path, timestamp=ts, body=body),
        }
        if self.passphrase:
            headers["X-PASSPHRASE"] = self.passphrase
        return headers


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 0.3
    max_delay_seconds: float = 8.0
    backoff_multiplier: float = 2.0
    jitter_seconds: float = 0.1
    retriable_status_codes: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)


class IdempotencyStore:
    """Simple in-memory idempotency key store for order submissions."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def exists(self, key: str) -> bool:
        return key in self._seen

    def mark(self, key: str) -> None:
        self._seen.add(key)


class HTTPOMSClient(OMSClient):
    """HTTP-based OMS adapter for broker/execution platform integration."""

    def __init__(
        self,
        submit_url: str,
        cancel_url: str,
        health_url: str | None = None,
        order_updates_url: str | None = None,
        headers: dict[str, str] | None = None,
        signer: HMACRequestSigner | None = None,
        retry_policy: RetryPolicy | None = None,
        idempotency_store: IdempotencyStore | None = None,
        timeout_seconds: int = 8,
    ) -> None:
        self.submit_url = submit_url
        self.cancel_url = cancel_url
        self.health_url = health_url
        self.order_updates_url = order_updates_url
        self.headers = headers or {}
        self.signer = signer
        self.retry_policy = retry_policy or RetryPolicy()
        self.idempotency_store = idempotency_store or IdempotencyStore()
        self.timeout_seconds = timeout_seconds

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload) if payload is not None else ""
        data = body.encode("utf-8") if payload is not None else None
        parsed = urllib.parse.urlparse(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        headers = {"Content-Type": "application/json", **self.headers}
        if self.signer is not None:
            headers.update(self.signer.auth_headers(method=method, path=path, body=body))
        if extra_headers:
            headers.update(extra_headers)

        attempt = 0
        while True:
            attempt += 1
            request = urllib.request.Request(
                url=url,
                data=data,
                method=method.upper(),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    status = int(getattr(response, "status", 200))
                    text = response.read().decode("utf-8")
                    if status in self.retry_policy.retriable_status_codes and attempt < self.retry_policy.max_attempts:
                        raise urllib.error.HTTPError(url, status, f"retryable {status}", hdrs=None, fp=None)
                    return json.loads(text) if text else {}
            except urllib.error.HTTPError as exc:
                if exc.code not in self.retry_policy.retriable_status_codes or attempt >= self.retry_policy.max_attempts:
                    raise
            except urllib.error.URLError:
                if attempt >= self.retry_policy.max_attempts:
                    raise

            delay = min(
                self.retry_policy.base_delay_seconds * (self.retry_policy.backoff_multiplier ** (attempt - 1)),
                self.retry_policy.max_delay_seconds,
            )
            delay += random.uniform(0.0, self.retry_policy.jitter_seconds)
            time.sleep(delay)

    @staticmethod
    def _idempotency_key(order: OrderRequest) -> str:
        if order.client_order_id:
            return f"client:{order.client_order_id}"
        digest = hashlib.sha256(json.dumps(order.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
        return f"ord:{digest[:24]}"

    def _submit_one(self, order: OrderRequest) -> tuple[OrderAck | None, OrderAck | None]:
        idem_key = self._idempotency_key(order)
        if self.idempotency_store.exists(idem_key):
            reject = OrderAck(
                order_id=f"local-duplicate-{uuid4().hex[:10]}",
                client_order_id=order.client_order_id,
                accepted=False,
                status="duplicate",
                reason="idempotency_duplicate",
                broker_timestamp=_now_utc(),
            )
            return None, reject

        response = self._request_json(
            method="POST",
            url=self.submit_url,
            payload={"order": order.to_dict()},
            extra_headers={"Idempotency-Key": idem_key},
        )
        self.idempotency_store.mark(idem_key)
        ack_payload = response.get("ack")
        if ack_payload is None:
            accepted = bool(response.get("accepted", False))
            ack_payload = {
                "order_id": str(response.get("order_id", f"order-{uuid4().hex}")),
                "client_order_id": order.client_order_id,
                "accepted": accepted,
                "status": str(response.get("status", "accepted" if accepted else "rejected")),
                "reason": response.get("reason"),
                "broker_timestamp": _now_utc(),
            }
        if isinstance(ack_payload.get("broker_timestamp"), str):
            ack_payload["broker_timestamp"] = datetime.fromisoformat(ack_payload["broker_timestamp"])
        ack = OrderAck(**ack_payload)
        if ack.accepted:
            return ack, None
        return None, ack

    def submit_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        accepted: list[OrderAck] = []
        rejected: list[OrderAck] = []
        for order in orders:
            try:
                ok, rej = self._submit_one(order)
            except Exception as exc:
                ok, rej = None, OrderAck(
                    order_id=f"order-error-{uuid4().hex[:10]}",
                    client_order_id=order.client_order_id,
                    accepted=False,
                    status="error",
                    reason=str(exc),
                    broker_timestamp=_now_utc(),
                )
            if ok is not None:
                accepted.append(ok)
            if rej is not None:
                rejected.append(rej)
        return OrderBatchResult(acks=accepted, rejected=rejected, status_events=[])

    def cancel_all(self, strategy_id: str) -> None:
        self._request_json(method="POST", url=self.cancel_url, payload={"strategy_id": strategy_id})

    def health_check(self) -> bool:
        if self.health_url is None:
            return True
        try:
            self._request_json(method="GET", url=self.health_url, payload=None)
            return True
        except Exception:
            return False

    def poll_order_updates(self, strategy_id: str, max_events: int = 200) -> list[OrderStatusEvent]:
        if self.order_updates_url is None:
            return []
        try:
            response = self._request_json(
                method="POST",
                url=self.order_updates_url,
                payload={"strategy_id": strategy_id, "limit": max_events},
            )
        except Exception:
            return []
        events: list[OrderStatusEvent] = []
        for row in response.get("events", []):
            payload = dict(row)
            if isinstance(payload.get("broker_timestamp"), str):
                payload["broker_timestamp"] = datetime.fromisoformat(payload["broker_timestamp"])
            if isinstance(payload.get("last_fill"), dict):
                last_fill = dict(payload["last_fill"])
                if isinstance(last_fill.get("fill_timestamp"), str):
                    last_fill["fill_timestamp"] = datetime.fromisoformat(last_fill["fill_timestamp"])
                payload["last_fill"] = FillEvent(**last_fill)
            events.append(OrderStatusEvent(**payload))
        return events


class BrokerWebSocketClient:
    """WebSocket callback template for broker order update streams."""

    def __init__(
        self,
        ws_url: str,
        signer: HMACRequestSigner,
        on_message: Callable[[dict[str, Any]], None],
        on_error: Callable[[Exception], None] | None = None,
        subscribe_message_factory: Callable[[], dict[str, Any]] | None = None,
        reconnect_max_attempts: int = 8,
    ) -> None:
        self.ws_url = ws_url
        self.signer = signer
        self.on_message = on_message
        self.on_error = on_error
        self.subscribe_message_factory = subscribe_message_factory
        self.reconnect_max_attempts = reconnect_max_attempts
        self._running = False

    def _auth_message(self) -> dict[str, Any]:
        ts = str(int(time.time() * 1000))
        return {
            "op": "auth",
            "args": {
                "api_key": self.signer.api_key,
                "timestamp": ts,
                "signature": self.signer.sign("GET", "/ws/auth", ts, ""),
                "passphrase": self.signer.passphrase,
            },
        }

    def run_forever(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "BrokerWebSocketClient requires websocket-client dependency."
            ) from exc

        self._running = True
        attempts = 0
        while self._running:
            ws = None
            try:
                ws = websocket.create_connection(self.ws_url, timeout=12)
                ws.send(json.dumps(self._auth_message()))
                if self.subscribe_message_factory:
                    ws.send(json.dumps(self.subscribe_message_factory()))
                attempts = 0
                while self._running:
                    raw = ws.recv()
                    if raw is None:
                        break
                    msg = json.loads(raw)
                    self.on_message(msg)
            except Exception as exc:  # pragma: no cover
                attempts += 1
                if self.on_error:
                    self.on_error(exc)
                if attempts >= self.reconnect_max_attempts:
                    break
                time.sleep(min(2 ** attempts, 30))
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

    def stop(self) -> None:
        self._running = False


class SimpleEMSClient(EMSClient):
    """Minimal EMS stub that forwards orders to one OMS endpoint."""

    def __init__(self, oms_client: OMSClient) -> None:
        self.oms_client = oms_client

    def route_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        return self.oms_client.submit_orders(orders)

    def health_check(self) -> bool:
        return self.oms_client.health_check()
