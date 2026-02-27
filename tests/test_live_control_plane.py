from __future__ import annotations

import pytest

from hourly_trading_system.live import LiveControlPlane


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
