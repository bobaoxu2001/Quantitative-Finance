"""Template: build live runner from config/live/live_system.yaml."""

from __future__ import annotations

from pathlib import Path
import yaml

from hourly_trading_system.analytics import LiveTCAAttributor
from hourly_trading_system.config import SystemConfig, load_config
from hourly_trading_system.execution import (
    BrokerConnectionConfig,
    BrokerKind,
    PaperOMSClient,
    RetryPolicy,
    build_broker_oms_adapter,
)
from hourly_trading_system.governance import ModelRegistry
from hourly_trading_system.live import (
    AlertRouter,
    FileBackedRealtimeQueue,
    FillReconciler,
    LiveControlPlane,
    RBACPolicy,
)
from hourly_trading_system.orchestration import LiveTradingRunner
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import LiveGuardConfig, LiveSafetyGuard, RiskController


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_oms_from_live_config(live_cfg: dict):
    broker_cfg = live_cfg.get("broker", {})
    mode = str(broker_cfg.get("mode", "paper")).lower()
    if mode == "paper":
        return PaperOMSClient(audit_path="outputs/paper_oms_audit.jsonl")

    retry_raw = broker_cfg.get("oms", {}).get("retry", {})
    retry = RetryPolicy(
        max_attempts=int(retry_raw.get("max_attempts", 4)),
        base_delay_seconds=float(retry_raw.get("base_delay_seconds", 0.3)),
        max_delay_seconds=float(retry_raw.get("max_delay_seconds", 8.0)),
        backoff_multiplier=float(retry_raw.get("backoff_multiplier", 2.0)),
        jitter_seconds=float(retry_raw.get("jitter_seconds", 0.1)),
        retriable_status_codes=tuple(retry_raw.get("retriable_status_codes", [408, 409, 425, 429, 500, 502, 503, 504])),
    )
    oms_raw = broker_cfg.get("oms", {})
    conn = BrokerConnectionConfig(
        broker=BrokerKind(str(broker_cfg.get("kind", "alpaca"))),
        submit_url=str(oms_raw["submit_url"]),
        cancel_url=str(oms_raw["cancel_url"]),
        health_url=oms_raw.get("health_url"),
        order_updates_url=oms_raw.get("order_updates_url"),
        api_key=oms_raw.get("auth", {}).get("api_key"),
        api_secret=oms_raw.get("auth", {}).get("api_secret"),
        passphrase=oms_raw.get("auth", {}).get("passphrase"),
        timeout_seconds=8,
        retry_policy=retry,
    )
    return build_broker_oms_adapter(conn)


def build_live_runner(
    system_config_path: str = "config/system.yaml",
    live_config_path: str = "config/live/live_system.yaml",
) -> LiveTradingRunner:
    system_cfg: SystemConfig = load_config(system_config_path)
    live_cfg = _load_yaml(live_config_path)
    queue_cfg = live_cfg.get("queue", {})
    queue = FileBackedRealtimeQueue(queue_cfg.get("path", "outputs/live_queue"))
    router = AlertRouter.with_console_and_file(live_cfg.get("alerts", {}).get("file_path", "outputs/live_alerts.jsonl"))

    guard_raw = live_cfg.get("risk_guards", {})
    safety_guard = LiveSafetyGuard(
        config=LiveGuardConfig(
            max_orders_per_cycle=int(guard_raw.get("max_orders_per_cycle", 80)),
            max_order_notional=float(guard_raw.get("max_order_notional", 50_000.0)),
            max_total_cycle_notional=float(guard_raw.get("max_total_cycle_notional", 200_000.0)),
            max_rejection_streak=int(guard_raw.get("max_rejection_streak", 5)),
            max_consecutive_failures=int(guard_raw.get("max_consecutive_failures", 3)),
            max_unresolved_orders=int(guard_raw.get("max_unresolved_orders", 40)),
            max_intraday_loss_pct=float(guard_raw.get("max_intraday_loss_pct", 0.05)),
            limit_protection_bps=float(guard_raw.get("limit_protection_bps", 30.0)),
            hard_price_floor=float(guard_raw.get("hard_price_floor", 0.01)),
        )
    )

    registry = ModelRegistry(root_dir=live_cfg.get("registry", {}).get("root_dir", "outputs/model_registry"))
    ctrl_cfg = live_cfg.get("control_plane", {})
    control_plane = LiveControlPlane(
        state_path=ctrl_cfg.get("state_path", "outputs/live_controls.json"),
        required_unlock_approvals=int(ctrl_cfg.get("required_unlock_approvals", 2)),
        rbac_policy=RBACPolicy(
            permissions={
                action: set(roles)
                for action, roles in ctrl_cfg.get("permissions", {}).items()
            }
            if ctrl_cfg.get("permissions")
            else RBACPolicy().permissions
        ),
        enforce_rbac=bool(ctrl_cfg.get("enforce_rbac", False)),
    )
    allocator = HourlyPortfolioAllocator(config=system_cfg.portfolio)
    risk_ctrl = RiskController(
        max_drawdown=system_cfg.risk.max_drawdown_hard_stop,
        target_vol_low=system_cfg.portfolio.target_vol_lower,
        target_vol_high=system_cfg.portfolio.target_vol_upper,
    )
    oms = _build_oms_from_live_config(live_cfg)
    return LiveTradingRunner(
        config=system_cfg,
        registry=registry,
        allocator=allocator,
        risk_controller=risk_ctrl,
        alert_router=router,
        realtime_queue=queue,
        oms_client=oms,
        safety_guard=safety_guard,
        reconciler=FillReconciler(cash=system_cfg.initial_capital),
        tca_attributor=LiveTCAAttributor(),
        control_plane=control_plane,
        strategy_id=live_cfg.get("strategy_id", "hourly_system_live"),
    )


if __name__ == "__main__":
    runner = build_live_runner()
    print("Live runner built successfully:", runner.strategy_id)
