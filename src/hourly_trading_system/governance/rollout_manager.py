"""Operational manager for canary evaluation and automated actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .deployment_guard import CanaryPolicy
from .model_registry import ModelRegistry


@dataclass(slots=True)
class RolloutDecision:
    action: str
    stats: dict[str, float]
    deployment_state: dict[str, Any]


class RolloutManager:
    """Evaluate canary performance and trigger promote/rollback/hold actions."""

    def __init__(self, registry: ModelRegistry, canary_policy: CanaryPolicy | None = None) -> None:
        self.registry = registry
        self.canary_policy = canary_policy or CanaryPolicy()

    def evaluate(
        self,
        champion_returns: pd.Series,
        canary_returns: pd.Series,
        champion_turnover: pd.Series | None = None,
        canary_turnover: pd.Series | None = None,
    ) -> RolloutDecision:
        action, stats = self.canary_policy.evaluate(
            champion_returns=champion_returns,
            canary_returns=canary_returns,
            champion_turnover=champion_turnover,
            canary_turnover=canary_turnover,
        )
        if action == "promote":
            state = self.registry.promote_canary().to_dict()
            return RolloutDecision(action=action, stats=stats, deployment_state=state)
        if action == "rollback":
            state = self.registry.rollback(reason=f"auto_rollback: {stats}").to_dict()
            return RolloutDecision(action=action, stats=stats, deployment_state=state)
        return RolloutDecision(action="hold", stats=stats, deployment_state=self.registry.deployment_state().to_dict())
