"""Persistent control plane for live kill-switch and unlock approvals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from .contracts import now_utc


@dataclass(slots=True)
class UnlockRequest:
    requestor: str
    reason: str
    requested_at: str
    approvals: list[str] = field(default_factory=list)
    finalized: bool = False
    finalized_by: str | None = None
    finalized_at: str | None = None


@dataclass(slots=True)
class LiveControlState:
    force_kill_switch: bool = False
    kill_reason: str | None = None
    kill_actor: str | None = None
    kill_timestamp: str | None = None
    unlock_request: UnlockRequest | None = None
    last_updated_at: str = field(default_factory=lambda: now_utc().isoformat())

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "LiveControlState":
        data = payload.copy()
        if data.get("unlock_request"):
            data["unlock_request"] = UnlockRequest(**data["unlock_request"])
        return LiveControlState(**data)


class LiveControlPlane:
    """File-backed control plane for operator actions in production."""

    def __init__(self, state_path: str | Path, required_unlock_approvals: int = 2) -> None:
        self.state_path = Path(state_path)
        self.required_unlock_approvals = required_unlock_approvals
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._save(LiveControlState())

    def _save(self, state: LiveControlState) -> None:
        state.last_updated_at = now_utc().isoformat()
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)

    def get_state(self) -> LiveControlState:
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return LiveControlState.from_dict(payload)

    def force_kill_switch(self, actor: str, reason: str) -> LiveControlState:
        state = self.get_state()
        state.force_kill_switch = True
        state.kill_reason = reason
        state.kill_actor = actor
        state.kill_timestamp = now_utc().isoformat()
        self._save(state)
        return state

    def request_unlock(self, requestor: str, reason: str) -> LiveControlState:
        state = self.get_state()
        state.unlock_request = UnlockRequest(
            requestor=requestor,
            reason=reason,
            requested_at=now_utc().isoformat(),
            approvals=[],
            finalized=False,
        )
        self._save(state)
        return state

    def approve_unlock(self, approver: str) -> LiveControlState:
        state = self.get_state()
        if state.unlock_request is None:
            raise ValueError("No unlock request found.")
        if approver not in state.unlock_request.approvals:
            state.unlock_request.approvals.append(approver)
        self._save(state)
        return state

    def can_finalize_unlock(self, state: LiveControlState | None = None) -> bool:
        current = state or self.get_state()
        if current.unlock_request is None:
            return False
        return len(current.unlock_request.approvals) >= self.required_unlock_approvals

    def finalize_unlock(self, actor: str) -> LiveControlState:
        state = self.get_state()
        if state.unlock_request is None:
            raise ValueError("No unlock request found.")
        if len(state.unlock_request.approvals) < self.required_unlock_approvals:
            raise ValueError(
                f"Insufficient approvals ({len(state.unlock_request.approvals)}/{self.required_unlock_approvals})."
            )
        state.unlock_request.finalized = True
        state.unlock_request.finalized_by = actor
        state.unlock_request.finalized_at = now_utc().isoformat()
        state.force_kill_switch = False
        state.kill_reason = None
        state.kill_actor = None
        state.kill_timestamp = None
        self._save(state)
        return state
