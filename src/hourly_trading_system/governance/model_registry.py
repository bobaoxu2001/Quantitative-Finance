"""Model registry with approval workflow, canary rollout, and rollback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import pickle

from hourly_trading_system.live.contracts import (
    DeploymentState,
    ModelApprovalStatus,
    ModelVersionRecord,
    now_utc,
)


@dataclass(slots=True)
class ApprovalPolicy:
    """Policy gates used for approval and promotion decisions."""

    required_metrics: dict[str, float] = field(
        default_factory=lambda: {"sharpe": 0.8, "calmar": 0.4}
    )
    max_drawdown_metric_name: str = "max_drawdown"
    max_drawdown_limit: float = 0.20

    def evaluate(self, metrics: dict[str, float]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for metric_name, minimum in self.required_metrics.items():
            value = metrics.get(metric_name)
            if value is None or value < minimum:
                reasons.append(f"{metric_name}<{minimum}")
        dd = metrics.get(self.max_drawdown_metric_name)
        if dd is not None and dd > self.max_drawdown_limit:
            reasons.append(
                f"{self.max_drawdown_metric_name}>{self.max_drawdown_limit}"
            )
        return len(reasons) == 0, reasons


class ModelRegistry:
    """Filesystem-backed model registry for live deployment governance."""

    def __init__(self, root_dir: str | Path, policy: ApprovalPolicy | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.policy = policy or ApprovalPolicy()
        self.versions_dir = self.root_dir / "versions"
        self.artifacts_dir = self.root_dir / "artifacts"
        self.deployment_file = self.root_dir / "deployment_state.json"
        self.audit_file = self.root_dir / "audit_log.jsonl"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not self.deployment_file.exists():
            self._save_deployment(DeploymentState())

    def _version_file(self, version_id: str) -> Path:
        return self.versions_dir / f"{version_id}.json"

    def _artifact_file(self, version_id: str) -> Path:
        return self.artifacts_dir / f"{version_id}.pkl"

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        row = {"timestamp": now_utc().isoformat(), "event": event, **payload}
        with self.audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _save_record(self, record: ModelVersionRecord) -> None:
        with self._version_file(record.version_id).open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2)

    def _load_record(self, version_id: str) -> ModelVersionRecord:
        path = self._version_file(version_id)
        if not path.exists():
            raise KeyError(f"Unknown version_id '{version_id}'")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ModelVersionRecord.from_dict(payload)

    def _save_deployment(self, state: DeploymentState) -> None:
        with self.deployment_file.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)

    def _load_deployment(self) -> DeploymentState:
        with self.deployment_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return DeploymentState.from_dict(payload)

    def register_model(
        self,
        model_object: object,
        model_name: str,
        route: str,
        feature_schema: list[str],
        metrics: dict[str, float],
        notes: str = "",
    ) -> ModelVersionRecord:
        """Register a new model artifact in draft status."""
        version_id = f"{model_name}-{uuid4().hex[:12]}"
        artifact_path = self._artifact_file(version_id)
        with artifact_path.open("wb") as handle:
            pickle.dump(model_object, handle)

        record = ModelVersionRecord(
            version_id=version_id,
            model_name=model_name,
            route=route,
            artifact_path=str(artifact_path),
            feature_schema=feature_schema,
            metrics=metrics,
            status=ModelApprovalStatus.DRAFT,
            notes=notes,
        )
        self._save_record(record)
        self._audit("register_model", {"version_id": version_id, "model_name": model_name})
        return record

    def list_versions(self) -> list[ModelVersionRecord]:
        out: list[ModelVersionRecord] = []
        for path in sorted(self.versions_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                out.append(ModelVersionRecord.from_dict(json.load(handle)))
        return out

    def get_version(self, version_id: str) -> ModelVersionRecord:
        return self._load_record(version_id)

    def submit_for_approval(self, version_id: str, submitted_by: str) -> ModelVersionRecord:
        record = self._load_record(version_id)
        if record.status != ModelApprovalStatus.DRAFT:
            raise ValueError(f"Cannot submit version in status {record.status}")
        record.status = ModelApprovalStatus.PENDING_APPROVAL
        record.submitted_by = submitted_by
        self._save_record(record)
        self._audit("submit_for_approval", {"version_id": version_id, "submitted_by": submitted_by})
        return record

    def approve(self, version_id: str, approved_by: str, notes: str = "") -> ModelVersionRecord:
        record = self._load_record(version_id)
        if record.status not in {ModelApprovalStatus.PENDING_APPROVAL, ModelApprovalStatus.DRAFT}:
            raise ValueError(f"Cannot approve version in status {record.status}")
        passes, reasons = self.policy.evaluate(record.metrics)
        if not passes:
            raise ValueError(f"Approval policy failed: {', '.join(reasons)}")
        record.status = ModelApprovalStatus.APPROVED
        record.approved_by = approved_by
        if notes:
            record.notes = f"{record.notes}\n{notes}".strip()
        self._save_record(record)
        self._audit("approve_model", {"version_id": version_id, "approved_by": approved_by})
        return record

    def reject(self, version_id: str, approved_by: str, reason: str) -> ModelVersionRecord:
        record = self._load_record(version_id)
        if record.status not in {ModelApprovalStatus.PENDING_APPROVAL, ModelApprovalStatus.DRAFT}:
            raise ValueError(f"Cannot reject version in status {record.status}")
        record.status = ModelApprovalStatus.REJECTED
        record.approved_by = approved_by
        record.rejection_reason = reason
        self._save_record(record)
        self._audit("reject_model", {"version_id": version_id, "approved_by": approved_by, "reason": reason})
        return record

    def deploy_champion(self, version_id: str) -> DeploymentState:
        record = self._load_record(version_id)
        if record.status != ModelApprovalStatus.APPROVED:
            raise ValueError("Only approved models can be promoted to champion.")
        state = self._load_deployment()
        previous = state.champion_version
        state.previous_champion_version = previous
        state.champion_version = version_id
        state.canary_version = None
        state.canary_allocation = 0.0
        state.rollout_started_at = now_utc()
        self._save_deployment(state)

        record.status = ModelApprovalStatus.DEPLOYED_CHAMPION
        self._save_record(record)
        if previous:
            prev = self._load_record(previous)
            if prev.status == ModelApprovalStatus.DEPLOYED_CHAMPION:
                prev.status = ModelApprovalStatus.RETIRED
                self._save_record(prev)
        self._audit("deploy_champion", {"version_id": version_id, "previous_champion": previous})
        return state

    def deploy_canary(self, version_id: str, allocation: float = 0.10) -> DeploymentState:
        if allocation <= 0 or allocation >= 1:
            raise ValueError("Canary allocation must be between 0 and 1.")
        record = self._load_record(version_id)
        if record.status != ModelApprovalStatus.APPROVED:
            raise ValueError("Only approved models can be deployed as canary.")
        state = self._load_deployment()
        if state.champion_version is None:
            raise ValueError("Champion model must be deployed before canary rollout.")
        state.canary_version = version_id
        state.canary_allocation = allocation
        state.rollout_started_at = now_utc()
        self._save_deployment(state)
        record.status = ModelApprovalStatus.DEPLOYED_CANARY
        self._save_record(record)
        self._audit(
            "deploy_canary",
            {"version_id": version_id, "allocation": allocation, "champion": state.champion_version},
        )
        return state

    def promote_canary(self) -> DeploymentState:
        state = self._load_deployment()
        if not state.canary_version:
            raise ValueError("No canary model deployed.")
        canary = self._load_record(state.canary_version)
        if canary.status != ModelApprovalStatus.DEPLOYED_CANARY:
            raise ValueError("Canary model is not in deployed canary state.")
        old_champion = state.champion_version
        state.previous_champion_version = old_champion
        state.champion_version = state.canary_version
        state.canary_version = None
        state.canary_allocation = 0.0
        state.rollout_started_at = now_utc()
        self._save_deployment(state)

        canary.status = ModelApprovalStatus.DEPLOYED_CHAMPION
        self._save_record(canary)
        if old_champion:
            old = self._load_record(old_champion)
            old.status = ModelApprovalStatus.RETIRED
            self._save_record(old)
        self._audit("promote_canary", {"new_champion": state.champion_version, "old_champion": old_champion})
        return state

    def rollback(self, reason: str) -> DeploymentState:
        state = self._load_deployment()
        fallback = state.previous_champion_version
        if not fallback:
            raise ValueError("No previous champion available for rollback.")
        current = state.champion_version
        state.champion_version = fallback
        state.canary_version = None
        state.canary_allocation = 0.0
        state.rollback_reason = reason
        state.rollout_started_at = now_utc()
        self._save_deployment(state)

        fallback_record = self._load_record(fallback)
        fallback_record.status = ModelApprovalStatus.DEPLOYED_CHAMPION
        self._save_record(fallback_record)
        if current:
            current_record = self._load_record(current)
            current_record.status = ModelApprovalStatus.ROLLED_BACK
            current_record.rolled_from_version = fallback
            self._save_record(current_record)
        self._audit("rollback", {"from_version": current, "to_version": fallback, "reason": reason})
        return state

    def load_model(self, version_id: str) -> object:
        record = self._load_record(version_id)
        artifact = Path(record.artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"Missing model artifact: {artifact}")
        with artifact.open("rb") as handle:
            return pickle.load(handle)

    def deployment_state(self) -> DeploymentState:
        return self._load_deployment()

    def get_champion_and_canary_models(self) -> dict[str, object | None]:
        state = self._load_deployment()
        champion = self.load_model(state.champion_version) if state.champion_version else None
        canary = self.load_model(state.canary_version) if state.canary_version else None
        return {"champion": champion, "canary": canary}
