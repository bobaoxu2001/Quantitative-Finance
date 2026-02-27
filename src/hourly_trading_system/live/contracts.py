"""Contracts for live trading infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def now_utc() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ModelApprovalStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED_CANARY = "deployed_canary"
    DEPLOYED_CHAMPION = "deployed_champion"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"


@dataclass(slots=True)
class QueueMessage:
    """Generic queue payload for real-time routing."""

    topic: str
    payload: dict[str, Any]
    key: str | None = None
    timestamp: datetime = field(default_factory=now_utc)
    message_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "QueueMessage":
        out = payload.copy()
        out["timestamp"] = datetime.fromisoformat(out["timestamp"])
        return QueueMessage(**out)


@dataclass(slots=True)
class AlertEvent:
    """Structured alert event for routing to channels."""

    severity: AlertSeverity
    source: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": str(self.severity),
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class OrderRequest:
    """Order request emitted by strategy for OMS/EMS."""

    symbol: str
    side: str  # BUY / SELL
    quantity: float
    order_type: str = "MKT"
    tif: str = "DAY"
    limit_price: float | None = None
    client_order_id: str | None = None
    strategy_id: str = "hourly_system"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrderAck:
    """OMS/EMS acknowledgement."""

    order_id: str
    client_order_id: str | None = None
    accepted: bool
    status: str
    reason: str | None = None
    broker_timestamp: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "broker_timestamp": self.broker_timestamp.isoformat(),
        }


@dataclass(slots=True)
class FillEvent:
    """Execution fill event from broker callbacks or polling."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fees: float = 0.0
    execution_id: str | None = None
    fill_timestamp: datetime = field(default_factory=now_utc)
    venue: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "fees": self.fees,
            "execution_id": self.execution_id,
            "fill_timestamp": self.fill_timestamp.isoformat(),
            "venue": self.venue,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "FillEvent":
        data = payload.copy()
        data["fill_timestamp"] = datetime.fromisoformat(str(data["fill_timestamp"]))
        return FillEvent(**data)


@dataclass(slots=True)
class OrderStatusEvent:
    """Order lifecycle status update event."""

    order_id: str
    status: str
    symbol: str | None = None
    side: str | None = None
    requested_quantity: float | None = None
    filled_quantity: float | None = None
    remaining_quantity: float | None = None
    average_fill_price: float | None = None
    client_order_id: str | None = None
    reason: str | None = None
    broker_timestamp: datetime = field(default_factory=now_utc)
    last_fill: FillEvent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "status": self.status,
            "symbol": self.symbol,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "client_order_id": self.client_order_id,
            "reason": self.reason,
            "broker_timestamp": self.broker_timestamp.isoformat(),
            "last_fill": self.last_fill.to_dict() if self.last_fill else None,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "OrderStatusEvent":
        data = payload.copy()
        data["broker_timestamp"] = datetime.fromisoformat(str(data["broker_timestamp"]))
        if data.get("last_fill"):
            data["last_fill"] = FillEvent.from_dict(data["last_fill"])
        return OrderStatusEvent(**data)


@dataclass(slots=True)
class ModelVersionRecord:
    """Model registry record with deployment status."""

    version_id: str
    model_name: str
    route: str
    artifact_path: str
    feature_schema: list[str]
    created_at: datetime = field(default_factory=now_utc)
    status: ModelApprovalStatus = ModelApprovalStatus.DRAFT
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    submitted_by: str | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    rolled_from_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = str(self.status)
        out["created_at"] = self.created_at.isoformat()
        return out

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ModelVersionRecord":
        data = payload.copy()
        data["status"] = ModelApprovalStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return ModelVersionRecord(**data)


@dataclass(slots=True)
class DeploymentState:
    """Current deployment slots for champion/canary strategy versions."""

    champion_version: str | None = None
    canary_version: str | None = None
    previous_champion_version: str | None = None
    canary_allocation: float = 0.0
    rollout_started_at: datetime | None = None
    rollback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_version": self.champion_version,
            "canary_version": self.canary_version,
            "previous_champion_version": self.previous_champion_version,
            "canary_allocation": self.canary_allocation,
            "rollout_started_at": self.rollout_started_at.isoformat() if self.rollout_started_at else None,
            "rollback_reason": self.rollback_reason,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "DeploymentState":
        data = payload.copy()
        if data.get("rollout_started_at"):
            data["rollout_started_at"] = datetime.fromisoformat(data["rollout_started_at"])
        return DeploymentState(**data)
