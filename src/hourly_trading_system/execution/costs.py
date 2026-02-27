"""Transaction cost and market impact models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CostBreakdown:
    slippage_bps: float
    spread_bps: float
    impact_bps: float

    @property
    def total_bps(self) -> float:
        return self.slippage_bps + self.spread_bps + self.impact_bps


def impact_eta_from_dollar_volume(dollar_volume: float, base_eta: float = 8.0) -> float:
    """Calibrate impact eta by liquidity bucket."""
    if dollar_volume >= 100_000_000:
        return base_eta * 0.5
    if dollar_volume >= 30_000_000:
        return base_eta * 0.7
    if dollar_volume >= 10_000_000:
        return base_eta * 1.0
    if dollar_volume >= 3_000_000:
        return base_eta * 1.4
    return base_eta * 1.9


def transaction_cost_bps(
    quantity: float,
    exec_volume: float,
    slippage_bps: float = 1.0,
    half_spread_bps: float = 0.5,
    impact_eta: float = 8.0,
) -> CostBreakdown:
    """Compute per-trade cost in bps with square-root impact."""
    if exec_volume <= 0:
        return CostBreakdown(slippage_bps, half_spread_bps, impact_bps=25.0)
    participation = min(abs(quantity) / exec_volume, 1.0)
    impact = float(impact_eta * np.sqrt(participation))
    return CostBreakdown(
        slippage_bps=float(slippage_bps),
        spread_bps=float(half_spread_bps),
        impact_bps=impact,
    )


def apply_cost_to_price(price: float, side: str, total_bps: float) -> float:
    """Apply transaction cost to execution price."""
    multiplier = 1.0 + total_bps * 1e-4 if side.upper() == "BUY" else 1.0 - total_bps * 1e-4
    return float(price * multiplier)
