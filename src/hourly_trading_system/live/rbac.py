"""RBAC policy objects for live control plane actions."""

from __future__ import annotations

from dataclasses import dataclass, field


class AccessDeniedError(PermissionError):
    """Raised when actor role is not allowed for action."""


@dataclass(slots=True)
class RBACPolicy:
    """
    Role-based access policy for control-plane actions.

    Default policy:
      - force_kill_switch: ops, risk_manager, admin
      - request_unlock: ops, admin
      - approve_unlock: risk_manager, head_of_trading, admin
      - finalize_unlock: admin, head_of_trading
    """

    permissions: dict[str, set[str]] = field(
        default_factory=lambda: {
            "force_kill_switch": {"ops", "risk_manager", "admin"},
            "request_unlock": {"ops", "admin"},
            "approve_unlock": {"risk_manager", "head_of_trading", "admin"},
            "finalize_unlock": {"admin", "head_of_trading"},
        }
    )

    def allow(self, action: str, role: str | None) -> bool:
        if role is None:
            return False
        allowed = self.permissions.get(action, set())
        return role in allowed

    def assert_allowed(self, action: str, role: str | None) -> None:
        if not self.allow(action, role):
            raise AccessDeniedError(f"Role '{role}' not permitted for action '{action}'.")
