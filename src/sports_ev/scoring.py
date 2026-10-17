"""Accuracy scoring for home/draw/away match forecasts."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import pandas as pd

OUTCOMES = ("home", "draw", "away")
_SCORE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)")


def _index(outcome: str) -> int:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    return OUTCOMES.index(outcome)


def multiclass_brier(probs: Sequence[float], outcome: str) -> float:
    """Sum of squared errors over home/draw/away. Uniform 1/3 scores 0.667."""
    hit = _index(outcome)
    return float(sum((p - (1.0 if i == hit else 0.0)) ** 2 for i, p in enumerate(probs)))


def log_loss(probs: Sequence[float], outcome: str) -> float:
    """Negative log probability given to the actual outcome. Uniform scores ln 3."""
    p = probs[_index(outcome)]
    return float(-math.log(max(p, 1e-12)))


def _parse_score(score: object) -> tuple[int, int] | None:
    """(home, away) goals from "2-1" or "1-1 (4-3 pens)"; None if missing or unparseable."""
    match = _SCORE.match(score) if isinstance(score, str) else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def summarize(log: pd.DataFrame) -> dict[str, float]:
    """Aggregate scores over settled rows of a predictions log.

    Expects columns p_home, p_draw, p_away, outcome, result and top_score.
    Rows whose outcome is not home/draw/away (unplayed or pending) are skipped;
    rows without a parseable result are left out of the score-based counts.
    """
    settled = log[log["outcome"].isin(OUTCOMES)]
    if settled.empty:
        return {"matches": 0}
    probs = settled[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    outcomes = settled["outcome"].tolist()
    favourites = [OUTCOMES[int(row.argmax())] for row in probs]
    results = settled["result"].map(_parse_score)
    tops = settled["top_score"].map(_parse_score)
    scored = results.notna()
    return {
        "matches": len(settled),
        "brier": float(sum(multiclass_brier(p, o) for p, o in zip(probs, outcomes)) / len(settled)),
        "brier_uniform": 2.0 / 3.0,
        "log_loss": float(sum(log_loss(p, o) for p, o in zip(probs, outcomes)) / len(settled)),
        "log_loss_uniform": math.log(3.0),
        "favourite_hits": sum(f == o for f, o in zip(favourites, outcomes)),
        "favourite_hits_expected": float(probs.max(axis=1).sum()),
        "top_score_hits": int((scored & (tops == results)).sum()),
        "goals_per_match": float(results[scored].map(sum).mean()) if scored.any() else math.nan,
    }
