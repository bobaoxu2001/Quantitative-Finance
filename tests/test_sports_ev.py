"""Tests for Kalshi EV helpers and forecast scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sports_ev import (
    FCM_PRECISION,
    american_to_prob,
    evaluate_contract,
    evaluate_match,
    fee_rate,
    fit_sharpness,
    kalshi_fee,
    multiclass_brier,
    recommend,
    sharpen,
    summarize,
)

LOG_PATH = Path(__file__).resolve().parents[1] / "football" / "predictions_log.csv"

# Coventry v Newcastle, 12 Oct 2026: our probabilities and the Kalshi asks
# (KXEPLGAME-26OCT12COVNEW) one hour before kickoff.
COV_PROBS = {"home": 0.35, "draw": 0.31, "away": 0.34}
COV_YES_ASKS = {"home": 0.27, "draw": 0.27, "away": 0.48}
COV_NO_ASKS = {"home": 0.74, "draw": 0.74, "away": 0.53}


def _find(rows, outcome, side):
    return next(e for e in rows if e.outcome == outcome and e.side == side)


def test_fee_direct_member_aligns_to_hundredth_of_a_cent() -> None:
    assert kalshi_fee(0.305, 100) == pytest.approx(1.4839)
    assert kalshi_fee(0.30, 100) == pytest.approx(1.47)
    assert kalshi_fee(0.50, 1) == pytest.approx(0.0175)


def test_fee_fcm_account_aligns_to_the_cent() -> None:
    assert kalshi_fee(0.305, 100, precision=FCM_PRECISION) == pytest.approx(1.49)
    assert kalshi_fee(0.50, 1, precision=FCM_PRECISION) == pytest.approx(0.02)


def test_fee_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        kalshi_fee(1.0, 10)
    with pytest.raises(ValueError):
        kalshi_fee(0.5, 0)


def test_fee_rate_by_series_type() -> None:
    assert fee_rate("quadratic") == pytest.approx(0.07)
    assert fee_rate("quadratic", maker=True) == 0.0
    assert fee_rate("quadratic_with_maker_fees", maker=True) == pytest.approx(0.0175)
    assert fee_rate("quadratic_with_maker_fees", fee_multiplier=0.5) == pytest.approx(0.035)
    with pytest.raises(ValueError):
        fee_rate("flat")


def test_american_odds_conversion() -> None:
    assert american_to_prob(228) == pytest.approx(100 / 328)
    assert american_to_prob(-150) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        american_to_prob(50)


def test_coventry_match_ev() -> None:
    rows = evaluate_match(COV_PROBS, COV_YES_ASKS, COV_NO_ASKS)
    assert len(rows) == 6
    assert _find(rows, "home", "yes").ev == pytest.approx(0.35 - 0.27 - 0.013797)
    assert _find(rows, "draw", "yes").ev == pytest.approx(0.31 - 0.27 - 0.013797)
    assert _find(rows, "away", "no").ev == pytest.approx(0.66 - 0.53 - 0.017437)
    assert _find(rows, "away", "yes").ev < 0


def test_no_side_needs_no_asks() -> None:
    rows = evaluate_match(COV_PROBS, COV_YES_ASKS)
    assert {e.side for e in rows} == {"yes"}


def test_kelly_fraction_for_binary_contract() -> None:
    e = evaluate_contract(0.34, 0.53, side="no")  # No on a 34% outcome wins 66% of the time
    assert e.p_win == pytest.approx(0.66)
    assert e.kelly == pytest.approx((0.66 - e.cost) / (1 - e.cost))
    assert evaluate_contract(0.34, 0.48, side="yes").kelly == 0.0


def test_recommend_coventry_picks_one_capped_contract() -> None:
    pick = recommend(COV_PROBS, COV_YES_ASKS, bankroll=1000.0, no_asks=COV_NO_ASKS)
    assert pick is not None
    assert (pick.evaluation.outcome, pick.evaluation.side) == ("home", "yes")
    assert pick.contracts == 70
    assert pick.stake <= 20.0  # quarter Kelly (2.3%) capped at 2% of bankroll


def test_recommend_ranks_by_log_growth_not_ev_per_contract() -> None:
    # away-No has the larger EV per contract (3.5c vs 3.4c) but a much lower return per dollar.
    pick = recommend(
        {"home": 0.245, "draw": 0.505, "away": 0.25},
        {"home": 0.20, "draw": 0.50, "away": 0.32},
        bankroll=1000.0,
        no_asks={"home": 0.82, "draw": 0.52, "away": 0.70},
    )
    assert pick is not None
    assert (pick.evaluation.outcome, pick.evaluation.side) == ("home", "yes")


def test_recommend_respects_min_edge_and_order_size() -> None:
    assert recommend(COV_PROBS, COV_YES_ASKS, bankroll=1000.0, no_asks=COV_NO_ASKS, min_edge=0.20) is None
    # A stake smaller than one contract is not a bet.
    assert recommend(COV_PROBS, COV_YES_ASKS, bankroll=10.0, no_asks=COV_NO_ASKS) is None


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        evaluate_match({"home": 0.5, "draw": 0.5, "away": 0.5}, COV_YES_ASKS)


def test_brier_scores() -> None:
    assert multiclass_brier([0.35, 0.31, 0.34], "home") == pytest.approx(0.6342)
    assert multiclass_brier([1 / 3, 1 / 3, 1 / 3], "draw") == pytest.approx(2 / 3)


def test_summarize_seeded_log() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-12"])
    assert s["matches"] == 9
    assert s["brier"] == pytest.approx(0.588, abs=5e-4)
    assert s["favourite_hits"] == 6
    assert s["top_score_hits"] == 2
    assert s["goals_per_match"] == pytest.approx(33 / 9)


def test_summarize_champions_league_night() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-13"])
    assert s["matches"] == 9
    assert s["favourite_hits"] == 5
    assert s["goals_per_match"] == pytest.approx(32 / 9)


def test_summarize_second_champions_league_night() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-14"])
    assert s["matches"] == 9
    assert s["brier"] == pytest.approx(0.488, abs=5e-4)
    assert s["favourite_hits"] == 6
    assert s["top_score_hits"] == 0
    assert s["goals_per_match"] == pytest.approx(46 / 9)


def test_summarize_europa_conference_night() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-15"])
    assert s["matches"] == 12
    assert s["brier"] == pytest.approx(0.444, abs=5e-4)
    assert s["favourite_hits"] == 8
    assert s["goals_per_match"] == pytest.approx(42 / 12)


def test_summarize_saturday_league_round() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-17"])
    assert s["matches"] == 11
    assert s["favourite_hits"] == 8
    assert s["goals_per_match"] == pytest.approx(43 / 11)


def test_summarize_first_calibrated_round() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-18"])  # probabilities logged after sharpening with a=1.5
    assert s["matches"] == 15
    assert s["brier"] == pytest.approx(0.509, abs=5e-4)
    assert s["favourite_hits"] == 10
    assert s["goals_per_match"] == pytest.approx(53 / 15)


def test_summarize_monday_round_with_unknown_score() -> None:
    log = pd.read_csv(LOG_PATH)
    day = log[log["date"] == "2026-10-19"]
    assert sorted(day["outcome"]) == ["away", "draw", "home"]
    s = summarize(day)
    assert s["matches"] == 3
    assert s["goals_per_match"] == pytest.approx(3.0)  # Braga v Gil Vicente has no recorded score


def test_summarize_champions_league_matchday_three() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-20"])
    assert s["matches"] == 9
    assert s["goals_per_match"] == pytest.approx(40 / 9)


def test_summarize_champions_league_matchday_three_second_night() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-21"])
    assert s["matches"] == 9
    assert s["goals_per_match"] == pytest.approx(34 / 9)


def test_summarize_europa_and_conference_league_matchday_three() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-22"])
    assert s["matches"] == 12
    assert s["goals_per_match"] == pytest.approx(28 / 12)


def test_summarize_friday_league_round() -> None:
    log = pd.read_csv(LOG_PATH)
    s = summarize(log[log["date"] == "2026-10-23"])
    assert s["matches"] == 5
    assert s["goals_per_match"] == pytest.approx(17 / 5)


def test_sharpen_moves_weight_to_the_favourite() -> None:
    assert sharpen([0.5, 0.3, 0.2], 1.0) == pytest.approx([0.5, 0.3, 0.2])
    out = sharpen([0.5, 0.3, 0.2], 1.5)
    assert sum(out) == pytest.approx(1.0)
    assert out[0] > 0.5 and out[2] < 0.2
    with pytest.raises(ValueError):
        sharpen([0.5, 0.3, 0.2], 0)


def test_fit_sharpness_finds_our_favourite_bias() -> None:
    # Favourites won 33 of the first 50 settled matches against 25.75 expected, so the fit stretches.
    # Rows from 18 Oct are logged already sharpened, so only the earlier, raw forecasts are fitted.
    log = pd.read_csv(LOG_PATH)
    assert fit_sharpness(log[log["date"] <= "2026-10-17"]) > 1.0


def test_bets_log_pnl_matches_fee_model() -> None:
    bets = pd.read_csv(LOG_PATH.parent / "bets_log.csv")
    bets = bets[bets["result"].notna()]  # unsettled bets have no result yet
    won = bets["result"] == "win"
    expected = bets["contracts"] * (won.astype(float) - bets["cost_per_contract_cents"] / 100)
    assert bets["pnl_usd"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-3)
    assert bets["pnl_usd"].sum() == pytest.approx(-21.76, abs=0.01)


def test_bets_log_costs_match_kalshi_fills() -> None:
    # The 14 Oct rows are real 50-100 contract fills, so this checks kalshi_fee against Kalshi's own charges.
    bets = pd.read_csv(LOG_PATH.parent / "bets_log.csv")
    for row in bets.itertuples():
        fee = kalshi_fee(row.ask_cents / 100, row.contracts)
        expected = row.ask_cents + 100 * fee / row.contracts
        assert row.cost_per_contract_cents == pytest.approx(expected, abs=0.01)


def test_summarize_tolerates_unsettled_and_unparseable_rows() -> None:
    log = pd.DataFrame(
        {
            "p_home": [0.5, 0.4, 0.3, 0.3],
            "p_draw": [0.3, 0.3, 0.4, 0.4],
            "p_away": [0.2, 0.3, 0.3, 0.3],
            "outcome": ["home", None, "home", "draw"],
            "result": ["1-0", None, None, "1-1 (4-3 pens)"],
            "top_score": ["1-0", "1-1", "1-0", "1-1"],
        }
    )
    s = summarize(log)
    assert s["matches"] == 3
    assert s["top_score_hits"] == 2
    assert s["goals_per_match"] == pytest.approx(1.5)
