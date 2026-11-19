"""Log an odds-blind forecast, then price it against every Kalshi market for the match.

The forecast is one JSON object with home_xg, away_xg, first_half_share, p_home,
p_draw, p_away and top_score. Log it (and commit) before running ``price``.

Examples:
    python scripts/football_price.py log --date 2026-11-12 --comp "UEFA Nations League" \
        --home Italy --away France --forecast '{"home_xg": 1.1, ...}'
    python scripts/football_price.py price --prefix KXUEFANL --kdate 26NOV12 --code ITAFRA \
        --forecast '{"home_xg": 1.1, ...}'
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from sports_ev import (
    boost_favourite,
    calibrated_xg,
    contract_prob,
    evaluate_contract,
    fit_xg,
    grid,
    one_x_two,
    parse_contract,
)

API = "https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker="
LOG_PATH = Path(__file__).resolve().parents[1] / "football" / "predictions_log.csv"
SERIES = ("GAME", "SPREAD", "TOTAL", "TEAMTOTAL", "BTTS", "1H", "1HTOTAL", "1HBTTS", "1HSPREAD")
OUTCOMES = ("home", "draw", "away")
MIN_EV_CENTS = 3.0
SIZE = 4


def _get(url: str) -> dict:
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(url, timeout=20))
        except Exception:
            time.sleep(3 + 2 * attempt)
    return {"markets": []}


def log_forecast(args: argparse.Namespace) -> None:
    f = json.loads(args.forecast)
    rows = list(csv.reader(LOG_PATH.open(newline="")))
    if any(r[:4] == [args.date, args.comp, args.home, args.away] for r in rows[1:]):
        print(f"already logged {args.home} v {args.away}")
        return
    probs = [f["p_home"], f["p_draw"], f["p_away"]]
    notes = (
        f"odds-blind researcher + reviewer subagents with {args.elo} opponent adjustment; "
        f"probabilities sharpened a=1.0 from raw {probs[0]:.3f}/{probs[1]:.3f}/{probs[2]:.3f}; "
        f"raw xG {f['home_xg']}-{f['away_xg']} (priced x1.15), 1H share {f['first_half_share']}"
    )
    row = [args.date, args.comp, args.home, args.away, *probs, f["top_score"], "pending"] + [""] * 9 + [notes]
    rows.append([str(x) for x in row])
    with LOG_PATH.open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\r\n").writerows(rows)
    print(f"logged {args.home} v {args.away}")


def _market_xg(game: dict, markets: list[dict]) -> tuple[float, float]:
    """Expected goals implied by the de-vigged 1X2 asks and the Over 2.5 mid price."""
    asks = [game[o][0] for o in OUTCOMES]
    target = [a / sum(asks) for a in asks]
    over = next((m for m in markets if m["series"] == "TOTAL" and m["suffix"] == "3"), None)
    best = None
    for tenths in range(15, 56):
        total = tenths / 10
        home, away = fit_xg(target, total)
        err = 0.0
        if over:
            p_over = contract_prob(parse_contract("TOTAL", "3", "", ""), grid(home, away))
            err = (p_over - (over["ask"] + over["bid"]) / 2) ** 2
        if best is None or err < best[0]:
            best = (err, home, away)
        if not over:
            break
    return (best[1], best[2]) if over else fit_xg(target, 2.7)


def price(args: argparse.Namespace) -> None:
    f = json.loads(args.forecast)
    markets = []
    for series in SERIES:
        for m in _get(f"{API}{args.prefix}{series}-{args.kdate}{args.code}").get("markets", []):
            markets.append(
                {
                    "series": series,
                    "suffix": m["ticker"].split("-")[-1],
                    "name": m.get("yes_sub_title") or m.get("title") or "",
                    "ask": float(m.get("yes_ask_dollars") or 0),
                    "bid": float(m.get("yes_bid_dollars") or 0),
                    "oi": float(m.get("open_interest_fp") or 0),
                }
            )
        time.sleep(0.3)
    teams = [m["suffix"] for m in markets if m["series"] == "GAME" and m["suffix"] != "TIE"]
    home_code = next(t for t in teams if args.code.startswith(t))
    away_code = next(t for t in teams if t != home_code)
    game = {}
    for m in markets:
        if m["series"] == "GAME":
            key = "draw" if m["suffix"] == "TIE" else "home" if m["suffix"] == home_code else "away"
            game[key] = (m["ask"], 1 - m["bid"])

    ours = [f["p_home"], f["p_draw"], f["p_away"]]
    ours = [p / sum(ours) for p in ours]
    ours_xg = calibrated_xg(ours, f["home_xg"], f["away_xg"])
    mkt_xg = _market_xg(game, markets)
    favourite = max(range(3), key=lambda k: game[OUTCOMES[k]][0])
    scenarios = {
        "ours": ours_xg,
        "mkt_total": fit_xg(ours, sum(mkt_xg)),
        "fav_boost": fit_xg(boost_favourite(ours, favourite), sum(ours_xg)),
    }
    share = f["first_half_share"]
    grids = {k: (grid(*xg), grid(xg[0] * share, xg[1] * share)) for k, xg in scenarios.items()}

    print(
        f"== {args.code} ours {ours[0]:.2f}/{ours[1]:.2f}/{ours[2]:.2f} xg {list(ours_xg)} | Kalshi yes "
        + "/".join(f"{game[o][0] * 100:.0f}" for o in OUTCOMES)
        + " no "
        + "/".join(f"{game[o][1] * 100:.0f}" for o in OUTCOMES)
        + f" mkt xg {list(mkt_xg)}"
    )
    fav_side = OUTCOMES[favourite]
    fav_price = game[fav_side][0] / sum(game[o][0] for o in OUTCOMES)
    heavy = fav_price >= 0.70 and ours[favourite] < fav_price - 0.05
    picks = []
    for m in markets:
        contract = parse_contract(m["series"], m["suffix"], home_code, away_code)
        if contract is None:
            continue
        p = {k: contract_prob(contract, g[1] if contract.first_half else g[0]) for k, g in grids.items()}
        if not 0.001 < p["ours"] < 0.999:
            continue
        for side, cost in (("yes", m["ask"]), ("no", 1 - m["bid"])):
            if not 0.01 <= cost <= 0.99:
                continue
            ev = {k: round(evaluate_contract(v, cost, side, SIZE).ev * 100, 1) for k, v in p.items()}
            if contract.kind in ("total", "teamtotal", "btts"):
                direction = "over" if side == "yes" else "under"
            else:
                backs_fav = contract.team == fav_side if contract.kind != "tie" else False
                direction = "fav" if backs_fav == (side == "yes") else "dog"
            if min(ev["ours"], ev["mkt_total"]) < MIN_EV_CENTS:
                continue
            flags = []
            if direction in ("dog", "under"):
                if ev["fav_boost"] < MIN_EV_CENTS:
                    continue
                if sum(ours_xg) < sum(mkt_xg):
                    flags.append("FAIL total<market")
                if heavy:
                    flags.append("FAIL heavy-fav gap")
            p_win = p["ours"] if side == "yes" else 1 - p["ours"]
            picks.append((min(ev["ours"], ev["mkt_total"]), m, side, cost, direction, p_win, ev, flags))
    for _, m, side, cost, direction, p_win, ev, flags in sorted(picks, key=lambda x: -x[0]):
        print(
            f"  {m['series']:9} {m['name'][:42]:42} {side:3} {cost * 100:3.0f} {direction:5} "
            f"p {p_win:.3f} EV {ev} oi {m['oi']:.0f} {' '.join(flags)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    log = sub.add_parser("log", help="append a forecast to football/predictions_log.csv")
    log.add_argument("--date", required=True)
    log.add_argument("--comp", required=True)
    log.add_argument("--home", required=True)
    log.add_argument("--away", required=True)
    log.add_argument("--forecast", required=True)
    log.add_argument("--elo", default="ClubElo", help="rating system named in the notes")
    log.set_defaults(func=log_forecast)
    pr = sub.add_parser("price", help="price the forecast against Kalshi markets")
    pr.add_argument("--prefix", required=True, help="series prefix, e.g. KXEPL")
    pr.add_argument("--kdate", required=True, help="date part of the event ticker, e.g. 26NOV12")
    pr.add_argument("--code", required=True, help="match code, e.g. ITAFRA")
    pr.add_argument("--forecast", required=True)
    pr.set_defaults(func=price)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
