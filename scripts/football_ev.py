"""Kalshi EV check for a football match and accuracy summary of the predictions log.

Examples:
    python scripts/football_ev.py ev --probs 0.35 0.31 0.34 --prices 30.5 28.5 47.8
    python scripts/football_ev.py score football/predictions_log.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from sports_ev import (
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    evaluate_match,
    recommend,
    summarize,
)

OUTCOMES = ("home", "draw", "away")


def _cents_to_dollars(values: list[float]) -> dict[str, float]:
    return {o: v / 100.0 for o, v in zip(OUTCOMES, values)}


def run_ev(args: argparse.Namespace) -> None:
    probs = dict(zip(OUTCOMES, args.probs))
    yes_prices = _cents_to_dollars(args.prices)
    no_prices = _cents_to_dollars(args.no_prices) if args.no_prices else None
    fee_rate = MAKER_FEE_RATE if args.maker else TAKER_FEE_RATE
    rows = evaluate_match(probs, yes_prices, no_prices, args.contracts, fee_rate)
    picks = recommend(
        probs,
        yes_prices,
        args.bankroll,
        no_prices=no_prices,
        min_edge=args.min_edge / 100.0,
        kelly_fraction=args.kelly_fraction,
        max_stake=args.max_stake,
        contracts=args.contracts,
        fee_rate=fee_rate,
        best_only=not args.all,
    )
    picked = {(p.evaluation.outcome, p.evaluation.side) for p in picks}

    print(f"{'contract':<10}{'price':>8}{'fee':>7}{'cost':>8}{'p_win':>7}{'EV':>8}{'ROI':>8}{'Kelly':>7}")
    for e in rows:
        mark = "  <- pick" if (e.outcome, e.side) in picked else ""
        print(
            f"{e.outcome + '-' + e.side:<10}{e.price * 100:>7.1f}c{e.fee_per_contract * 100:>6.2f}c"
            f"{e.cost * 100:>7.1f}c{e.p_win:>7.2f}{e.ev * 100:>+7.1f}c{e.roi:>+8.1%}{e.kelly:>7.1%}{mark}"
        )
    if not picks:
        print(f"\nNo contract clears the {args.min_edge:.1f}c minimum edge after fees: no bet.")
    for p in picks:
        e = p.evaluation
        print(
            f"\nPick {e.outcome}-{e.side}: stake ${p.stake:.2f} ({p.contracts} contracts at {e.cost * 100:.1f}c "
            f"incl. fee), {args.kelly_fraction:g} Kelly capped at {args.max_stake:.0%} of ${args.bankroll:,.0f}."
        )


def run_score(args: argparse.Namespace) -> None:
    s = summarize(pd.read_csv(args.log))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("ev", help="EV of every Yes/No contract on a home/draw/away market")
    ev.add_argument("--probs", nargs=3, type=float, required=True, metavar=("HOME", "DRAW", "AWAY"))
    ev.add_argument("--prices", nargs=3, type=float, required=True, metavar=("HOME", "DRAW", "AWAY"),
                    help="Kalshi Yes prices in cents")
    ev.add_argument("--no-prices", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                    help="Kalshi No prices in cents (default: 100 - Yes price)")
    ev.add_argument("--bankroll", type=float, default=1000.0)
    ev.add_argument("--min-edge", type=float, default=3.0, help="minimum EV per contract after fees, in cents")
    ev.add_argument("--kelly-fraction", type=float, default=0.25)
    ev.add_argument("--max-stake", type=float, default=0.02, help="cap as a fraction of bankroll")
    ev.add_argument("--contracts", type=int, default=100, help="order size used to round the fee")
    ev.add_argument("--maker", action="store_true", help="use the maker fee rate (resting limit order)")
    ev.add_argument("--all", action="store_true", help="list every qualifying contract, not just the best")
    ev.set_defaults(func=run_ev)

    score = sub.add_parser("score", help="accuracy summary of a predictions log CSV")
    score.add_argument("log", help="path to predictions_log.csv")
    score.set_defaults(func=run_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
