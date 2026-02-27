"""One-click production preflight checks for live deployment readiness."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import yaml

from hourly_trading_system.governance import ModelRegistry
from hourly_trading_system.live import LiveControlPlane


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str  # pass / warn / fail
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _check_python() -> CheckResult:
    ok = sys.version_info >= (3, 11)
    status = "pass" if ok else "fail"
    return CheckResult(
        name="python_version",
        status=status,
        message=f"Python {platform.python_version()}",
        details={"required": ">=3.11"},
    )


def _check_paths_writable(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        token = path.parent / ".write_test.tmp"
        token.write_text("ok", encoding="utf-8")
        token.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _check_live_config(live_cfg: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    queue_path = Path(live_cfg.get("queue", {}).get("path", "outputs/live_queue"))
    queue_ok = _check_paths_writable(queue_path / ".touch")
    results.append(
        CheckResult(
            name="queue_path_writable",
            status="pass" if queue_ok else "fail",
            message=str(queue_path),
            details={},
        )
    )

    replay_cfg = live_cfg.get("queue", {}).get("replay", {})
    seq_state = Path(replay_cfg.get("sequence_state_path", "outputs/ws_sequence.state"))
    seq_ok = _check_paths_writable(seq_state)
    results.append(
        CheckResult(
            name="ws_sequence_state_writable",
            status="pass" if seq_ok else "fail",
            message=str(seq_state),
            details={"replay_enabled": bool(replay_cfg.get("enabled", False))},
        )
    )

    broker = live_cfg.get("broker", {})
    mode = str(broker.get("mode", "paper")).lower()
    if mode == "paper":
        results.append(
            CheckResult(
                name="broker_mode",
                status="warn",
                message="paper mode enabled",
                details={},
            )
        )
    else:
        oms = broker.get("oms", {})
        auth = oms.get("auth", {})
        critical_missing = []
        for key in ("submit_url", "cancel_url"):
            if not oms.get(key):
                critical_missing.append(f"oms.{key}")
        for key in ("api_key", "api_secret"):
            if not auth.get(key):
                critical_missing.append(f"oms.auth.{key}")
        results.append(
            CheckResult(
                name="broker_credentials",
                status="fail" if critical_missing else "pass",
                message="broker live credential check",
                details={"missing": critical_missing},
            )
        )
    return results


def _check_registry_and_control(live_cfg: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    registry_dir = live_cfg.get("registry", {}).get("root_dir", "outputs/model_registry")
    try:
        registry = ModelRegistry(registry_dir)
        state = registry.deployment_state()
        champion = state.champion_version
        results.append(
            CheckResult(
                name="registry_champion_deployed",
                status="pass" if champion else "fail",
                message=str(champion),
                details={"canary": state.canary_version},
            )
        )
    except Exception as exc:
        results.append(
            CheckResult(
                name="registry_access",
                status="fail",
                message="registry unavailable",
                details={"error": str(exc)},
            )
        )

    ctrl_cfg = live_cfg.get("control_plane", {})
    state_path = Path(ctrl_cfg.get("state_path", "outputs/live_controls.json"))
    try:
        control = LiveControlPlane(
            state_path=state_path,
            required_unlock_approvals=int(ctrl_cfg.get("required_unlock_approvals", 2)),
            enforce_rbac=bool(ctrl_cfg.get("enforce_rbac", False)),
        )
        state = control.get_state()
        results.append(
            CheckResult(
                name="control_plane_ready",
                status="pass",
                message="control plane accessible",
                details={
                    "force_kill_switch": state.force_kill_switch,
                    "enforce_rbac": bool(ctrl_cfg.get("enforce_rbac", False)),
                },
            )
        )
    except Exception as exc:
        results.append(
            CheckResult(
                name="control_plane_ready",
                status="fail",
                message="control plane unavailable",
                details={"error": str(exc)},
            )
        )
    return results


def _check_startup_scripts() -> list[CheckResult]:
    startup = Path("cloud/startup.sh")
    preflight = Path("scripts/run_preflight.sh")
    return [
        CheckResult(
            name="cloud_startup_script",
            status="pass" if startup.exists() else "fail",
            message=str(startup),
            details={"executable": os.access(startup, os.X_OK) if startup.exists() else False},
        ),
        CheckResult(
            name="preflight_wrapper_script",
            status="pass" if preflight.exists() else "warn",
            message=str(preflight),
            details={"executable": os.access(preflight, os.X_OK) if preflight.exists() else False},
        ),
    ]


def run_preflight(system_path: Path, live_path: Path) -> list[CheckResult]:
    live_cfg = _load_yaml(live_path)
    results: list[CheckResult] = []
    results.append(_check_python())
    results.extend(_check_live_config(live_cfg))
    results.extend(_check_registry_and_control(live_cfg))
    results.extend(_check_startup_scripts())
    results.append(
        CheckResult(
            name="checklist_doc",
            status="pass" if Path("docs/PRODUCTION_PRECHECK.md").exists() else "warn",
            message="docs/PRODUCTION_PRECHECK.md",
            details={},
        )
    )
    return results


def summarize(results: list[CheckResult]) -> dict[str, Any]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for row in results:
        counts[row.status] += 1
    overall = "PASS"
    if counts["fail"] > 0:
        overall = "FAIL"
    elif counts["warn"] > 0:
        overall = "PASS_WITH_WARNINGS"
    return {"overall": overall, "counts": counts, "checks": [r.to_dict() for r in results]}


def main() -> None:
    system_cfg = Path("config/system.yaml")
    live_cfg = Path("config/live/live_system.yaml")
    result = summarize(run_preflight(system_cfg, live_cfg))
    print(json.dumps(result, indent=2))
    if result["overall"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
