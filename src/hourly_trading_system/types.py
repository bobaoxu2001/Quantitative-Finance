"""Core domain datatypes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Signal:
    symbol: str
    timestamp: datetime
    expected_excess_return: float
    downside_probability: float
    expected_cost_bps: float


@dataclass(slots=True)
class Order:
    symbol: str
    timestamp: datetime
    target_weight: float
    delta_weight: float
    quantity: float
    side: str


@dataclass(slots=True)
class Fill:
    symbol: str
    timestamp: datetime
    quantity: float
    fill_price: float
    notional: float
    trading_cost: float
    partial: bool


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float
