"""Hourly execution engine with partial fills and cost-aware accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hourly_trading_system.config import CostModelConfig
from hourly_trading_system.execution.costs import (
    apply_cost_to_price,
    impact_eta_from_dollar_volume,
    transaction_cost_bps,
)
from hourly_trading_system.types import Fill


@dataclass(slots=True)
class ExecutionResult:
    timestamp: pd.Timestamp
    fills: list[Fill]
    positions: dict[str, float]
    cash: float
    total_cost: float
    unfilled_symbols: list[str]

    def to_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["symbol", "timestamp", "quantity", "fill_price", "notional", "trading_cost", "partial"])
        return pd.DataFrame([fill.__dict__ for fill in self.fills])


class ExecutionEngine:
    """Simulate execution at next hour open with realistic constraints."""

    def __init__(self, cost_config: CostModelConfig) -> None:
        self.cost_config = cost_config

    def execute_target_weights(
        self,
        timestamp: pd.Timestamp,
        equity: float,
        current_positions: dict[str, float],
        cash: float,
        target_weights: pd.Series,
        market_next_open: pd.DataFrame,
    ) -> ExecutionResult:
        """
        Execute delta from current holdings to target weights at next open.

        market_next_open required columns:
          symbol, open, volume
        """
        if target_weights.empty:
            return ExecutionResult(
                timestamp=timestamp,
                fills=[],
                positions=current_positions.copy(),
                cash=cash,
                total_cost=0.0,
                unfilled_symbols=[],
            )

        prices = market_next_open.set_index("symbol")["open"].to_dict()
        volumes = market_next_open.set_index("symbol")["volume"].to_dict()
        fills: list[Fill] = []
        positions = current_positions.copy()
        total_cost = 0.0
        unfilled_symbols: list[str] = []

        symbols = sorted(set(target_weights.index).union(current_positions.keys()))
        for symbol in symbols:
            price = float(prices.get(symbol, 0.0))
            if price <= 0.0:
                unfilled_symbols.append(symbol)
                continue

            current_qty = float(positions.get(symbol, 0.0))
            current_notional = current_qty * price
            target_notional = float(target_weights.get(symbol, 0.0) * equity)
            delta_notional = target_notional - current_notional
            if abs(delta_notional) < 1e-6:
                continue

            desired_qty = delta_notional / price
            avail_volume = float(volumes.get(symbol, 0.0))
            fill_cap = self.cost_config.max_participation_rate * max(avail_volume, 0.0)
            if fill_cap <= 0:
                unfilled_symbols.append(symbol)
                continue

            fill_qty = max(min(desired_qty, fill_cap), -fill_cap)
            partial = abs(fill_qty - desired_qty) > 1e-9
            side = "BUY" if fill_qty > 0 else "SELL"

            dollar_volume = avail_volume * price
            eta = impact_eta_from_dollar_volume(
                dollar_volume=dollar_volume,
                base_eta=self.cost_config.base_impact_eta,
            )
            cost = transaction_cost_bps(
                quantity=fill_qty,
                exec_volume=max(avail_volume, 1.0),
                slippage_bps=self.cost_config.slippage_bps,
                half_spread_bps=self.cost_config.half_spread_bps,
                impact_eta=eta,
            )
            fill_price = apply_cost_to_price(price=price, side=side, total_bps=cost.total_bps)
            notional = fill_qty * fill_price
            trading_cost = abs(fill_qty * price) * cost.total_bps * 1e-4

            positions[symbol] = current_qty + fill_qty
            cash -= notional
            total_cost += trading_cost
            fills.append(
                Fill(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    quantity=fill_qty,
                    fill_price=fill_price,
                    notional=notional,
                    trading_cost=trading_cost,
                    partial=partial,
                )
            )
            if partial:
                unfilled_symbols.append(symbol)

        # Drop near-zero positions to keep book clean.
        positions = {k: v for k, v in positions.items() if abs(v) > 1e-10}
        return ExecutionResult(
            timestamp=timestamp,
            fills=fills,
            positions=positions,
            cash=cash,
            total_cost=total_cost,
            unfilled_symbols=sorted(set(unfilled_symbols)),
        )

    @staticmethod
    def mark_to_market(
        positions: dict[str, float],
        cash: float,
        market_snapshot: pd.DataFrame,
        price_col: str = "close",
    ) -> dict[str, Any]:
        """Compute current equity, invested value, and symbol notional."""
        prices = market_snapshot.set_index("symbol")[price_col].to_dict()
        notionals = {symbol: qty * float(prices.get(symbol, 0.0)) for symbol, qty in positions.items()}
        invested = float(sum(notionals.values()))
        equity = cash + invested
        return {"equity": equity, "invested": invested, "cash": cash, "notionals": notionals}
