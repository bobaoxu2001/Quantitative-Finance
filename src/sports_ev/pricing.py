"""Score-grid pricing of Kalshi football contracts from a forecast's expected goals."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

RHO = -0.06
GOALS_MULTIPLIER = 1.15
FAVOURITE_BOOST = 0.09

Grid = Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class Contract:
    """What a Kalshi market pays on: ``kind`` is win, tie, spread, total, teamtotal or btts."""

    kind: str
    team: str = ""
    line: float = 0.0
    first_half: bool = False


def grid(home_xg: float, away_xg: float, rho: float = RHO, max_goals: int = 11) -> dict[tuple[int, int], float]:
    """Dixon-Coles score probabilities for scores up to ``max_goals`` each, normalised to 1."""
    if home_xg <= 0 or away_xg <= 0:
        raise ValueError("expected goals must be positive")
    out = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = math.exp(-home_xg - away_xg) * home_xg**i * away_xg**j / (math.factorial(i) * math.factorial(j))
            if (i, j) == (0, 0):
                p *= 1 - home_xg * away_xg * rho
            elif (i, j) == (0, 1):
                p *= 1 + home_xg * rho
            elif (i, j) == (1, 0):
                p *= 1 + away_xg * rho
            elif (i, j) == (1, 1):
                p *= 1 - rho
            out[(i, j)] = p
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def grid_prob(g: Grid, event: Callable[[int, int], bool]) -> float:
    return sum(p for (i, j), p in g.items() if event(i, j))


def one_x_two(g: Grid) -> list[float]:
    """Home, draw and away probabilities of a score grid."""
    return [grid_prob(g, lambda i, j: i > j), grid_prob(g, lambda i, j: i == j), grid_prob(g, lambda i, j: i < j)]


def fit_xg(target: Sequence[float], total: float) -> tuple[float, float]:
    """Split ``total`` goals between the sides so the grid's 1X2 is closest to ``target``."""
    best = None
    for share in range(5, 96):
        home = total * share / 100
        probs = one_x_two(grid(home, total - home))
        err = sum((p - t) ** 2 for p, t in zip(probs, target))
        if best is None or err < best[0]:
            best = (err, home)
    return round(best[1], 3), round(total - best[1], 3)


def calibrated_xg(probs: Sequence[float], home_xg: float, away_xg: float, goals: float = GOALS_MULTIPLIER) -> tuple[float, float]:
    """Expected goals with the total scaled by ``goals`` and the split refitted to our 1X2."""
    norm = sum(probs)
    return fit_xg([p / norm for p in probs], goals * (home_xg + away_xg))


def boost_favourite(probs: Sequence[float], favourite: int, boost: float = FAVOURITE_BOOST) -> list[float]:
    """Move ``boost`` of probability to outcome ``favourite``, taken pro rata from the other two."""
    rest = 1 - probs[favourite]
    boost = min(boost, rest)
    return [p + boost if k == favourite else p - boost * p / rest for k, p in enumerate(probs)]


def parse_contract(series: str, suffix: str, home_code: str, away_code: str) -> Contract | None:
    """Contract for a market from its series suffix (e.g. ``1HSPREAD``) and ticker suffix (e.g. ``FRA2``).

    Returns None when the ticker cannot be matched to a side, so that a market
    is never priced against the wrong team.
    """
    half = series.startswith("1H")
    base = series[2:] if half else series
    if base in ("GAME", ""):
        if suffix == "TIE":
            return Contract("tie", first_half=half)
        if suffix in (home_code, away_code):
            return Contract("win", "home" if suffix == home_code else "away", first_half=half)
        return None
    if base == "BTTS":
        return Contract("btts", first_half=half) if suffix == "BTTS" else None
    if base == "TOTAL":
        return Contract("total", line=int(suffix) - 0.5, first_half=half) if suffix.isdigit() else None
    if base in ("SPREAD", "TEAMTOTAL"):
        for side, code in sorted((("home", home_code), ("away", away_code)), key=lambda x: -len(x[1])):
            m = re.fullmatch(re.escape(code) + r"(\d+)", suffix)
            if m:
                return Contract(base.lower(), side, int(m.group(1)) - 0.5, half)
    return None


def contract_prob(contract: Contract, g: Grid) -> float:
    """Probability that ``contract`` settles Yes under score grid ``g`` (home goals first)."""
    line, home = contract.line, contract.team == "home"
    events: dict[str, Callable[[int, int], bool]] = {
        "tie": lambda i, j: i == j,
        "win": lambda i, j: (i > j) if home else (j > i),
        "spread": lambda i, j: (i - j > line) if home else (j - i > line),
        "total": lambda i, j: i + j > line,
        "teamtotal": lambda i, j: (i if home else j) > line,
        "btts": lambda i, j: i > 0 and j > 0,
    }
    return grid_prob(g, events[contract.kind])
