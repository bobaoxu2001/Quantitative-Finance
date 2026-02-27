"""Live trading safety guards: kill-switch, circuit-breaker, and limit protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from hourly_trading_system.live.contracts import OrderRequest


@dataclass(slots=True)
class LiveGuardConfig:
    max_orders_per_cycle: int = 80
    max_order_notional: float = 50_000.0
    max_total_cycle_notional: float = 200_000.0
    max_rejection_streak: int = 5
    max_consecutive_failures: int = 3
    max_unresolved_orders: int = 40
    max_intraday_loss_pct: float = 0.05
    limit_protection_bps: float = 30.0
    hard_price_floor: float = 0.01


@dataclass(slots=True)
class GuardDecision:
    allowed: bool
    reason: str
    adjusted_orders: list[OrderRequest]


@dataclass(slots=True)
class LiveSafetyGuard:
    """Stateful guard rail set for live order flow."""

    config: LiveGuardConfig = field(default_factory=LiveGuardConfig)
    kill_switch_engaged: bool = False
    kill_switch_reason: str | None = None
    rejection_streak: int = 0
    failure_streak: int = 0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0

    def engage_kill_switch(self, reason: str) -> None:
        self.kill_switch_engaged = True
        self.kill_switch_reason = reason

    def reset_kill_switch(self) -> None:
        self.kill_switch_engaged = False
        self.kill_switch_reason = None

    def update_equity_state(self, equity: float, is_new_day: bool = False) -> None:
        if self.peak_equity == 0:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if is_new_day or self.day_start_equity == 0:
            self.day_start_equity = equity
        if self.day_start_equity > 0:
            day_loss = 1.0 - equity / self.day_start_equity
            if day_loss >= self.config.max_intraday_loss_pct:
                self.engage_kill_switch(f"intraday_loss>{self.config.max_intraday_loss_pct:.2%}")

    def apply_limit_protection(
        self,
        orders: Iterable[OrderRequest],
        reference_prices: dict[str, float],
    ) -> list[OrderRequest]:
        """Set protective limit price bands on market orders."""
        adjusted: list[OrderRequest] = []
        for order in orders:
            price = float(reference_prices.get(order.symbol, 0.0))
            if price <= self.config.hard_price_floor:
                continue
            updated = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                tif=order.tif,
                limit_price=order.limit_price,
                client_order_id=order.client_order_id,
                strategy_id=order.strategy_id,
                metadata=dict(order.metadata),
            )
            if updated.limit_price is None:
                bps = self.config.limit_protection_bps * 1e-4
                if updated.side.upper() == "BUY":
                    updated.limit_price = price * (1.0 + bps)
                else:
                    updated.limit_price = price * (1.0 - bps)
                updated.order_type = "LMT"
                updated.metadata["limit_protected"] = True
            adjusted.append(updated)
        return adjusted

    def pre_trade_check(
        self,
        orders: list[OrderRequest],
        reference_prices: dict[str, float],
    ) -> GuardDecision:
        if self.kill_switch_engaged:
            return GuardDecision(
                allowed=False,
                reason=f"kill_switch:{self.kill_switch_reason}",
                adjusted_orders=[],
            )
        if len(orders) > self.config.max_orders_per_cycle:
            return GuardDecision(
                allowed=False,
                reason=f"too_many_orders>{self.config.max_orders_per_cycle}",
                adjusted_orders=[],
            )
        total_notional = 0.0
        for order in orders:
            px = float(reference_prices.get(order.symbol, 0.0))
            notional = abs(order.quantity * px)
            total_notional += notional
            if notional > self.config.max_order_notional:
                return GuardDecision(
                    allowed=False,
                    reason=f"order_notional>{self.config.max_order_notional}",
                    adjusted_orders=[],
                )
        if total_notional > self.config.max_total_cycle_notional:
            return GuardDecision(
                allowed=False,
                reason=f"cycle_notional>{self.config.max_total_cycle_notional}",
                adjusted_orders=[],
            )
        adjusted = self.apply_limit_protection(orders, reference_prices=reference_prices)
        return GuardDecision(allowed=True, reason="ok", adjusted_orders=adjusted)

    def register_batch_outcome(self, accepted_count: int, rejected_count: int, had_exception: bool = False) -> None:
        if had_exception:
            self.failure_streak += 1
            if self.failure_streak >= self.config.max_consecutive_failures:
                self.engage_kill_switch("consecutive_submission_failures")
            return
        self.failure_streak = 0
        if rejected_count > 0 and accepted_count == 0:
            self.rejection_streak += 1
        else:
            self.rejection_streak = 0
        if self.rejection_streak >= self.config.max_rejection_streak:
            self.engage_kill_switch("rejection_streak")

    def evaluate_reconciliation(self, unresolved_orders: int, duplicate_fills: int) -> None:
        if unresolved_orders > self.config.max_unresolved_orders:
            self.engage_kill_switch("unresolved_orders_exceeded")
        if duplicate_fills > 10:
            self.engage_kill_switch("duplicate_fill_storm")
