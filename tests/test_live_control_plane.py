from __future__ import annotations

import pytest

from hourly_trading_system.live import AccessDeniedError, LiveControlPlane, RBACPolicy


def test_control_plane_kill_and_unlock_approval_flow(tmp_path) -> None:
    control = LiveControlPlane(state_path=tmp_path / "controls.json", required_unlock_approvals=2)
    state = control.force_kill_switch(actor="ops_a", reason="manual_incident")
    assert state.force_kill_switch
    assert state.kill_actor == "ops_a"

    state = control.request_unlock(requestor="ops_b", reason="incident_mitigated")
    assert state.unlock_request is not None
    assert state.unlock_request.requestor == "ops_b"
    assert len(state.unlock_request.approvals) == 0

    state = control.approve_unlock(approver="risk_mgr")
    assert state.unlock_request is not None
    assert len(state.unlock_request.approvals) == 1
    with pytest.raises(ValueError):
        control.finalize_unlock(actor="ops_b")

    state = control.approve_unlock(approver="head_of_trading")
    assert state.unlock_request is not None
    assert len(state.unlock_request.approvals) == 2
    final = control.finalize_unlock(actor="ops_b")
    assert not final.force_kill_switch
    assert final.unlock_request is not None
    assert final.unlock_request.finalized


def test_control_plane_rbac_enforcement(tmp_path) -> None:
    policy = RBACPolicy(
        permissions={
            "force_kill_switch": {"admin"},
            "request_unlock": {"ops"},
            "approve_unlock": {"risk_manager"},
            "finalize_unlock": {"admin"},
        }
    )
    control = LiveControlPlane(
        state_path=tmp_path / "controls_rbac.json",
        required_unlock_approvals=1,
        rbac_policy=policy,
        enforce_rbac=True,
    )
    with pytest.raises(AccessDeniedError):
        control.force_kill_switch(actor="ops_user", reason="x", actor_role="ops")

    control.force_kill_switch(actor="admin_user", reason="x", actor_role="admin")
    control.request_unlock(requestor="ops_user", reason="recover", actor_role="ops")
    with pytest.raises(AccessDeniedError):
        control.approve_unlock(approver="ops_user", actor_role="ops")
    control.approve_unlock(approver="risk_user", actor_role="risk_manager")
    with pytest.raises(AccessDeniedError):
        control.finalize_unlock(actor="risk_user", actor_role="risk_manager")
    final = control.finalize_unlock(actor="admin_user", actor_role="admin")
    assert final.unlock_request is not None
    assert final.unlock_request.finalized
