"""Tests for Kalshi EV helpers and forecast scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sports_ev import (
    american_to_prob,
    evaluate_contract,
    evaluate_match,
    kalshi_fee,
    multiclass_brier,
    recommend,
    summarize,
)

LOG_PATH = Path(__file__).resolve().parents[1] / "football" / "predictions_log.csv"

# Coventry v Newcastle, 12 Oct 2026: our probabilities and Kalshi Yes prices.
COV_PROBS = {"home": 0.35, "draw": 0.31, "away": 0.34}
COV_PRICES = {"home": 0.305, "draw": 0.285, "away": 0.478}


def _find(rows, outcome, side):
    return next(e for e in rows if e.outcome == outcome and e.side == side)


def test_fee_rounds_up_per_order() -> None:
    assert kalshi_fee(0.305, 100) == pytest.approx(1.49)
    assert kalshi_fee(0.50, 100) == pytest.approx(1.75)
    assert kalshi_fee(0.50, 1) == pytest.approx(0.02)


def test_fee_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        kalshi_fee(1.0, 10)
    with pytest.raises(ValueError):
        kalshi_fee(0.5, 0)


def test_american_odds_conversion() -> None:
    assert american_to_prob(228) == pytest.approx(100 / 328)
    assert american_to_prob(-150) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        american_to_prob(50)


def test_coventry_match_ev() -> None:
    rows = evaluate_match(COV_PROBS, COV_PRICES)
    assert len(rows) == 6
    assert _find(rows, "away", "no").ev == pytest.approx(0.1205)
    assert _find(rows, "home", "yes").ev == pytest.approx(0.0301)
    assert _find(rows, "draw", "yes").ev == pytest.approx(0.0107)
    assert _find(rows, "away", "yes").ev == pytest.approx(-0.1555)


def test_kelly_fraction_for_binary_contract() -> None:
    e = evaluate_contract(0.34, 0.522, side="no")  # No on a 34% outcome wins 66% of the time
    assert e.p_win == pytest.approx(0.66)
    assert e.cost == pytest.approx(0.5395)
    assert e.kelly == pytest.approx((0.66 - 0.5395) / (1 - 0.5395))
    assert evaluate_contract(0.34, 0.478, side="yes").kelly == 0.0


def test_recommend_picks_best_direction_and_caps_stake() -> None:
    picks = recommend(COV_PROBS, COV_PRICES, bankroll=1000.0)
    assert len(picks) == 1
    pick = picks[0]
    assert (pick.evaluation.outcome, pick.evaluation.side) == ("away", "no")
    assert pick.stake == pytest.approx(20.0)  # quarter Kelly (6.5%) capped at 2%
    assert pick.contracts == 37


def test_recommend_respects_min_edge() -> None:
    picks = recommend(COV_PROBS, COV_PRICES, bankroll=1000.0, best_only=False)
    chosen = {(p.evaluation.outcome, p.evaluation.side) for p in picks}
    assert ("draw", "yes") not in chosen  # +1.1c is below the 3c threshold
    assert ("away", "yes") not in chosen
    assert recommend(COV_PROBS, COV_PRICES, bankroll=1000.0, min_edge=0.20) == []


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        evaluate_match({"home": 0.5, "draw": 0.5, "away": 0.5}, COV_PRICES)


def test_brier_scores() -> None:
    assert multiclass_brier([0.35, 0.31, 0.34], "home") == pytest.approx(0.6342)
    assert multiclass_brier([1 / 3, 1 / 3, 1 / 3], "draw") == pytest.approx(2 / 3)


def test_summarize_seeded_log() -> None:
    s = summarize(pd.read_csv(LOG_PATH))
    assert s["matches"] == 9
    assert s["brier"] == pytest.approx(0.588, abs=5e-4)
    assert s["favourite_hits"] == 6
    assert s["top_score_hits"] == 2
    assert s["goals_per_match"] == pytest.approx(33 / 9)


def test_summarize_skips_unsettled_rows() -> None:
    log = pd.DataFrame(
        {
            "p_home": [0.5, 0.4],
            "p_draw": [0.3, 0.3],
            "p_away": [0.2, 0.3],
            "outcome": ["home", None],
            "result": ["1-0", None],
            "top_score": ["1-0", "1-1"],
        }
    )
    s = summarize(log)
    assert s["matches"] == 1
    assert s["top_score_hits"] == 1
