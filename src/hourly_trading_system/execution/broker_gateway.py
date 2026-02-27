"""OMS/EMS integration interfaces for live execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import urllib.request

from hourly_trading_system.live.contracts import OrderAck, OrderRequest


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class OrderBatchResult:
    """Batch order send result."""

    acks: list[OrderAck]
    rejected: list[OrderAck]

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
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _audit(self, payload: dict[str, Any]) -> None:
        if not self.audit_path:
            return
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def submit_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        acks: list[OrderAck] = []
        rejected: list[OrderAck] = []
        for order in orders:
            if order.quantity <= 0:
                ack = OrderAck(
                    order_id=f"paper-{uuid4().hex}",
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
                accepted=True,
                status="accepted",
                reason=None,
                broker_timestamp=_now_utc(),
            )
            acks.append(ack)
            self._audit({"type": "accept", "order": order.to_dict(), "ack": ack.to_dict()})
        return OrderBatchResult(acks=acks, rejected=rejected)

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


class HTTPOMSClient(OMSClient):
    """HTTP-based OMS adapter for broker/execution platform integration."""

    def __init__(
        self,
        submit_url: str,
        cancel_url: str,
        health_url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: int = 8,
    ) -> None:
        self.submit_url = submit_url
        self.cancel_url = cancel_url
        self.health_url = health_url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", **self.headers},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}

    def submit_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        response = self._post(
            self.submit_url,
            payload={"orders": [order.to_dict() for order in orders]},
        )
        accepted = response.get("accepted", [])
        rejected = response.get("rejected", [])
        acks = [OrderAck(**ack) for ack in accepted]
        rejects = [OrderAck(**ack) for ack in rejected]
        return OrderBatchResult(acks=acks, rejected=rejects)

    def cancel_all(self, strategy_id: str) -> None:
        self._post(self.cancel_url, payload={"strategy_id": strategy_id})

    def health_check(self) -> bool:
        if self.health_url is None:
            return True
        try:
            request = urllib.request.Request(url=self.health_url, method="GET", headers=self.headers)
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):
                return True
        except Exception:
            return False


class SimpleEMSClient(EMSClient):
    """Minimal EMS stub that forwards orders to one OMS endpoint."""

    def __init__(self, oms_client: OMSClient) -> None:
        self.oms_client = oms_client

    def route_orders(self, orders: list[OrderRequest]) -> OrderBatchResult:
        return self.oms_client.submit_orders(orders)

    def health_check(self) -> bool:
        return self.oms_client.health_check()
