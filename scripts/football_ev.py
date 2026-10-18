"""Kalshi EV check for a football match and accuracy summary of the predictions log.

Prices are asks in cents: --prices are Yes asks, --no-prices are No asks
(No ask = 100 - Yes bid). Without --no-prices only Yes contracts are evaluated.

Examples:
    python scripts/football_ev.py ev --probs 0.35 0.31 0.34 --prices 27 27 48 --no-prices 74 74 53
    python scripts/football_ev.py score football/predictions_log.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from sports_ev import (
    DIRECT_MEMBER_PRECISION,
    FCM_PRECISION,
    evaluate_match,
    fee_rate,
    fit_sharpness,
    recommend,
    sharpen,
    summarize,
)

OUTCOMES = ("home", "draw", "away")


def _cents_to_dollars(values: list[float]) -> dict[str, float]:
    return {o: v / 100.0 for o, v in zip(OUTCOMES, values)}


def run_ev(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    yes_total = sum(args.prices)
    if not 90.0 <= yes_total <= 130.0:
        parser.error(
            f"Yes asks sum to {yes_total:g}c; a home/draw/away market should sum to roughly 100-110c. "
            "Prices are in cents (e.g. --prices 27 27 48), not dollars."
        )
    probs = dict(zip(OUTCOMES, args.probs))
    yes_asks = _cents_to_dollars(args.prices)
    no_asks = _cents_to_dollars(args.no_prices) if args.no_prices else None
    rate = fee_rate(args.fee_type, args.fee_multiplier, maker=args.maker)
    precision = FCM_PRECISION if args.fcm else DIRECT_MEMBER_PRECISION
    rows = evaluate_match(probs, yes_asks, no_asks, 100, rate, precision)
    pick = recommend(
        probs,
        yes_asks,
        args.bankroll,
        no_asks=no_asks,
        min_edge=args.min_edge / 100.0,
        kelly_fraction=args.kelly_fraction,
        max_stake=args.max_stake,
        fee_rate=rate,
        precision=precision,
    )
    picked = (pick.evaluation.outcome, pick.evaluation.side) if pick else None

    print(f"{'contract':<10}{'ask':>8}{'fee':>8}{'cost':>8}{'p_win':>7}{'EV':>8}{'ROI':>8}{'Kelly':>7}")
    for e in rows:
        mark = "  <- pick" if (e.outcome, e.side) == picked else ""
        print(
            f"{e.outcome + '-' + e.side:<10}{e.price * 100:>7.1f}c{e.fee_per_contract * 100:>7.3f}c"
            f"{e.cost * 100:>7.2f}c{e.p_win:>7.2f}{e.ev * 100:>+7.2f}c{e.roi:>+8.1%}{e.kelly:>7.1%}{mark}"
        )
    print("(fee and EV per contract at a 100-contract order)")
    if no_asks is None:
        print("No contracts skipped: pass --no-prices with the No asks (100 - Yes bid) to include them.")
    if pick is None:
        print(f"\nNo contract clears the {args.min_edge:g}c minimum edge after fees at a sizeable stake: no bet.")
        return
    e = pick.evaluation
    print(
        f"\nPick {e.outcome}-{e.side}: {pick.contracts} contracts, ${pick.stake:.2f} incl. fee "
        f"({e.cost * 100:.2f}c each, EV {e.ev * 100:+.2f}c), {args.kelly_fraction:g} Kelly capped at "
        f"{args.max_stake:.0%} of ${args.bankroll:,.0f}."
    )


def run_score(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    log = pd.read_csv(args.log)
    s = summarize(log)
    if s["matches"] == 0:
        print("No settled matches in the log.")
        return
    n = s["matches"]
    print(f"Settled matches:      {n}")
    print(f"Brier (mean):         {s['brier']:.3f}   uniform baseline {s['brier_uniform']:.3f}")
    print(f"Log loss (mean):      {s['log_loss']:.3f}   uniform baseline {s['log_loss_uniform']:.3f}")
    print(f"Favourite hits:       {s['favourite_hits']}/{n}   expected from our probs {s['favourite_hits_expected']:.2f}")
    print(f"Top-score hits:       {s['top_score_hits']}/{n}")
    print(f"Goals per match:      {s['goals_per_match']:.2f}")
    a = fit_sharpness(log)
    sharpened = log.copy()
    probs = sharpened[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    sharpened[["p_home", "p_draw", "p_away"]] = [sharpen(p, a) for p in probs]
    print(f"Best sharpening a:    {a:.2f}   Brier if applied {summarize(sharpened)['brier']:.3f} (in-sample)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("ev", help="EV of the Yes/No contracts on a home/draw/away market")
    ev.add_argument("--probs", nargs=3, type=float, required=True, metavar=("HOME", "DRAW", "AWAY"))
    ev.add_argument("--prices", nargs=3, type=float, required=True, metavar=("HOME", "DRAW", "AWAY"),
                    help="Kalshi Yes asks in cents (what you pay, not the last price)")
    ev.add_argument("--no-prices", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                    help="Kalshi No asks in cents (= 100 - Yes bid); No contracts are skipped without them")
    ev.add_argument("--bankroll", type=float, default=1000.0)
    ev.add_argument("--min-edge", type=float, default=3.0, help="minimum EV per contract after fees, in cents")
    ev.add_argument("--kelly-fraction", type=float, default=0.25)
    ev.add_argument("--max-stake", type=float, default=0.02, help="cap as a fraction of bankroll")
    ev.add_argument("--fee-type", default="quadratic_with_maker_fees",
                    choices=["quadratic", "quadratic_with_maker_fees", "quadratic_with_combo_maker_fees"],
                    help="series fee_type from GET /series/{ticker}; only affects --maker")
    ev.add_argument("--fee-multiplier", type=float, default=1.0, help="series fee_multiplier from GET /series/{ticker}")
    ev.add_argument("--maker", action="store_true",
                    help="price as a resting limit order; --prices must then be your own limit prices")
    ev.add_argument("--fcm", action="store_true", help="account cleared by an FCM/broker (fees align to the cent)")
    ev.set_defaults(func=run_ev)

    score = sub.add_parser("score", help="accuracy summary of a predictions log CSV")
    score.add_argument("log", help="path to predictions_log.csv")
    score.set_defaults(func=run_score)

    args = parser.parse_args()
    args.func(args, parser)


if __name__ == "__main__":
    main()
