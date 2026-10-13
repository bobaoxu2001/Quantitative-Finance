"""Kalshi fee, expected-value and Kelly sizing helpers for binary event contracts.

Prices are in dollars per contract (0-1). A winning contract pays $1.
For a three-way football market (home / draw / away) each outcome is its own
Yes/No contract, so a "No" on one outcome is a bet on the other two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

TAKER_FEE_RATE = 0.07
MAKER_FEE_RATE = 0.0175
OUTCOMES = ("home", "draw", "away")


@dataclass(frozen=True, slots=True)
class ContractEvaluation:
    outcome: str
    side: str
    price: float
    fee_per_contract: float
    cost: float
    p_win: float
    ev: float
    roi: float
    kelly: float


@dataclass(frozen=True, slots=True)
class Recommendation:
    evaluation: ContractEvaluation
    stake: float
    contracts: int


def _check_price(price: float) -> None:
    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be strictly between 0 and 1 dollars, got {price}")


def kalshi_fee(price: float, contracts: int, rate: float = TAKER_FEE_RATE) -> float:
    """Fee in dollars for one order: rate * C * P * (1 - P), rounded up to the cent.

    Use ``TAKER_FEE_RATE`` for orders that take liquidity and ``MAKER_FEE_RATE``
    for resting limit orders on markets that charge maker fees. Check the
    current Kalshi fee schedule: some series carry a different multiplier.
    """
    _check_price(price)
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    raw_cents = rate * contracts * price * (1.0 - price) * 100.0
    return math.ceil(round(raw_cents, 6)) / 100.0


def american_to_prob(odds: float) -> float:
    """Convert American odds (e.g. +228 or -150) to an implied probability."""
    if -100 < odds < 100:
        raise ValueError(f"American odds must be <= -100 or >= +100, got {odds}")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def evaluate_contract(
    p_model: float,
    price: float,
    side: str = "yes",
    contracts: int = 100,
    fee_rate: float = TAKER_FEE_RATE,
    outcome: str = "",
) -> ContractEvaluation:
    """Expected value of buying one Yes or No contract at ``price``.

    ``p_model`` is our probability that the outcome happens. For ``side="no"``
    the contract wins when the outcome does not happen, and ``price`` is the
    No price. The fee is spread over ``contracts`` because Kalshi rounds per order.
    """
    if side not in ("yes", "no"):
        raise ValueError("side must be 'yes' or 'no'")
    if not 0.0 <= p_model <= 1.0:
        raise ValueError(f"p_model must be in [0, 1], got {p_model}")
    p_win = p_model if side == "yes" else 1.0 - p_model
    fee = kalshi_fee(price, contracts, fee_rate) / contracts
    cost = price + fee
    ev = p_win - cost
    kelly = max(0.0, ev / (1.0 - cost)) if cost < 1.0 else 0.0
    return ContractEvaluation(
        outcome=outcome,
        side=side,
        price=price,
        fee_per_contract=fee,
        cost=cost,
        p_win=p_win,
        ev=ev,
        roi=ev / cost,
        kelly=kelly,
    )


def evaluate_match(
    probs: Mapping[str, float],
    yes_prices: Mapping[str, float],
    no_prices: Mapping[str, float] | None = None,
    contracts: int = 100,
    fee_rate: float = TAKER_FEE_RATE,
) -> list[ContractEvaluation]:
    """Evaluate Yes and No on every outcome of a home/draw/away market.

    ``no_prices`` defaults to ``1 - yes_price``; pass the live No ask instead
    when the book has a spread.
    """
    total = sum(probs[o] for o in OUTCOMES)
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"home/draw/away probabilities must sum to 1, got {total:.3f}")
    rows: list[ContractEvaluation] = []
    for outcome in OUTCOMES:
        yes_price = yes_prices[outcome]
        no_price = no_prices[outcome] if no_prices is not None else 1.0 - yes_price
        rows.append(evaluate_contract(probs[outcome], yes_price, "yes", contracts, fee_rate, outcome))
        rows.append(evaluate_contract(probs[outcome], no_price, "no", contracts, fee_rate, outcome))
    return rows


def recommend(
    probs: Mapping[str, float],
    yes_prices: Mapping[str, float],
    bankroll: float,
    no_prices: Mapping[str, float] | None = None,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.02,
    contracts: int = 100,
    fee_rate: float = TAKER_FEE_RATE,
    best_only: bool = True,
) -> list[Recommendation]:
    """Directions whose EV per contract after fees is at least ``min_edge`` dollars.

    Stake is fractional Kelly capped at ``max_stake`` of bankroll. Contracts on
    the same match are correlated (e.g. home Yes and away No both win on a home
    win), so by default only the single best direction is returned.
    """
    candidates = [
        e for e in evaluate_match(probs, yes_prices, no_prices, contracts, fee_rate) if e.ev >= min_edge
    ]
    candidates.sort(key=lambda e: e.ev, reverse=True)
    if best_only:
        candidates = candidates[:1]
    picks: list[Recommendation] = []
    for e in candidates:
        stake = bankroll * min(e.kelly * kelly_fraction, max_stake)
        picks.append(Recommendation(evaluation=e, stake=stake, contracts=math.floor(stake / e.cost)))
    return picks
