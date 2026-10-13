"""Kalshi fee, expected-value and Kelly sizing helpers for binary event contracts.

Prices are in dollars per contract (0-1). A winning contract pays $1.
For a three-way football market (home / draw / away) each outcome is its own
Yes/No contract, so a "No" on one outcome is a bet on the other two.

Use ask prices (what you pay). On Kalshi the No ask is 1 - Yes bid, not
1 - Yes ask, so No contracts are only evaluated when No asks are supplied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

BASE_FEE_RATE = 0.07
TAKER_FEE_RATE = BASE_FEE_RATE
MAKER_FEE_RATE = 0.25 * BASE_FEE_RATE
# Balance precision: kalshi.com accounts are direct members; FCM-cleared accounts align to the cent.
DIRECT_MEMBER_PRECISION = 0.0001
FCM_PRECISION = 0.01
OUTCOMES = ("home", "draw", "away")

_MAKER_MULTIPLIERS = {
    "quadratic": 0.0,
    "quadratic_with_maker_fees": 0.25,
    "quadratic_with_combo_maker_fees": 0.5,
}


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
    log_growth: float


def _check_price(price: float) -> None:
    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be strictly between 0 and 1 dollars, got {price}")


def fee_rate(fee_type: str = "quadratic_with_maker_fees", fee_multiplier: float = 1.0, maker: bool = False) -> float:
    """Fee rate for a series, from the ``fee_type`` and ``fee_multiplier`` of GET /series/{ticker}.

    Takers pay 0.07 on every quadratic series. Makers pay 0 on ``quadratic``
    series and a quarter (or half, for combo) of the taker rate otherwise.
    """
    if fee_type not in _MAKER_MULTIPLIERS:
        raise ValueError(f"unsupported fee_type {fee_type!r}; expected one of {sorted(_MAKER_MULTIPLIERS)}")
    multiplier = _MAKER_MULTIPLIERS[fee_type] if maker else 1.0
    return BASE_FEE_RATE * multiplier * fee_multiplier


def kalshi_fee(
    price: float,
    contracts: int,
    rate: float = TAKER_FEE_RATE,
    precision: float = DIRECT_MEMBER_PRECISION,
) -> float:
    """Fee in dollars for buying ``contracts`` at ``price`` in one order.

    Model fee is rate * C * P * (1 - P), rounded up to $0.000001; the balance
    change is then rounded so it lands on the account's precision grid
    ($0.0001 for direct members, $0.01 for FCM-cleared accounts). Kalshi's
    per-order accumulator makes split fills net out to this single-fill fee.
    """
    _check_price(price)
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    notional = price * contracts
    trade_fee = math.ceil(round(rate * contracts * price * (1.0 - price) * 1e6, 6)) / 1e6
    aligned_cost = math.ceil(round((notional + trade_fee) / precision, 6)) * precision
    return round(aligned_cost - notional, 6)


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
    precision: float = DIRECT_MEMBER_PRECISION,
) -> ContractEvaluation:
    """Expected value of buying one Yes or No contract at ask ``price``.

    ``p_model`` is our probability that the outcome happens. For ``side="no"``
    the contract wins when the outcome does not happen, and ``price`` is the
    No ask. The fee is spread over ``contracts`` because it is charged per order.
    """
    if side not in ("yes", "no"):
        raise ValueError("side must be 'yes' or 'no'")
    if not 0.0 <= p_model <= 1.0:
        raise ValueError(f"p_model must be in [0, 1], got {p_model}")
    p_win = p_model if side == "yes" else 1.0 - p_model
    fee = kalshi_fee(price, contracts, fee_rate, precision) / contracts
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
    yes_asks: Mapping[str, float],
    no_asks: Mapping[str, float] | None = None,
    contracts: int = 100,
    fee_rate: float = TAKER_FEE_RATE,
    precision: float = DIRECT_MEMBER_PRECISION,
) -> list[ContractEvaluation]:
    """Evaluate every outcome of a home/draw/away market at the given asks.

    No contracts are included only when ``no_asks`` is given (No ask = 1 - Yes bid).
    """
    total = sum(probs[o] for o in OUTCOMES)
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"home/draw/away probabilities must sum to 1, got {total:.3f}")
    rows: list[ContractEvaluation] = []
    for outcome in OUTCOMES:
        rows.append(
            evaluate_contract(probs[outcome], yes_asks[outcome], "yes", contracts, fee_rate, outcome, precision)
        )
        if no_asks is not None:
            rows.append(
                evaluate_contract(probs[outcome], no_asks[outcome], "no", contracts, fee_rate, outcome, precision)
            )
    return rows


def _log_growth(p_win: float, cost: float, fraction: float) -> float:
    """Expected log growth of bankroll when staking ``fraction`` on a contract costing ``cost``."""
    return p_win * math.log1p(fraction * (1.0 - cost) / cost) + (1.0 - p_win) * math.log1p(-fraction)


def _size(
    e: ContractEvaluation,
    p_model: float,
    bankroll: float,
    kelly_fraction: float,
    max_stake: float,
    fee_rate: float,
    precision: float,
) -> tuple[ContractEvaluation, int]:
    """Contract count for a fractional-Kelly stake, re-pricing the fee at the real order size."""
    target = bankroll * min(e.kelly * kelly_fraction, max_stake)
    n = math.floor(target / e.cost)
    for _ in range(3):
        if n <= 0:
            return e, 0
        e = evaluate_contract(p_model, e.price, e.side, n, fee_rate, e.outcome, precision)
        fitted = math.floor(target / e.cost)
        if fitted >= n:
            break
        n = fitted
    return e, max(n, 0)


def recommend(
    probs: Mapping[str, float],
    yes_asks: Mapping[str, float],
    bankroll: float,
    no_asks: Mapping[str, float] | None = None,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.02,
    fee_rate: float = TAKER_FEE_RATE,
    precision: float = DIRECT_MEMBER_PRECISION,
) -> Recommendation | None:
    """The single best contract on a match, or None if nothing clears ``min_edge``.

    Stake is fractional Kelly capped at ``max_stake`` of bankroll, and the fee
    and edge are re-checked at the actual order size. Candidates are ranked by
    expected log growth of bankroll at that stake. Only one contract is picked
    because contracts on the same match are correlated (home Yes and away No
    both win on a home win).
    """
    best: Recommendation | None = None
    for e in evaluate_match(probs, yes_asks, no_asks, 100, fee_rate, precision):
        if e.ev < min_edge:
            continue
        sized, n = _size(e, probs[e.outcome], bankroll, kelly_fraction, max_stake, fee_rate, precision)
        if n == 0 or sized.ev < min_edge:
            continue
        stake = n * sized.cost
        growth = _log_growth(sized.p_win, sized.cost, stake / bankroll)
        if best is None or growth > best.log_growth:
            best = Recommendation(evaluation=sized, stake=stake, contracts=n, log_growth=growth)
    return best
