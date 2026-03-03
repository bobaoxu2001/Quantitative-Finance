"""Generate detailed test-set trading report for the ensemble model (Route C).

Produces performance metrics, trading activity breakdown, and per-symbol
statistics in markdown table format for inclusion in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hourly_trading_system.backtest import HourlyBacktestEngine
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import ExecutionEngine
from hourly_trading_system.features.library import required_features
from hourly_trading_system.models import (
    AdaptiveEnsembleModel,
    DeepCrossSectionalModel,
    FactorLightGBMBaseline,
)
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController

# Reuse synthetic data generator from demo backtest.
from run_demo_backtest import make_synthetic_market_data


def main() -> None:
    market, membership, fundamentals, macro, sentiment, analyst = make_synthetic_market_data()

    config = SystemConfig()
    feature_cols = required_features()
    model_a = DeepCrossSectionalModel(feature_columns=feature_cols)
    model_b = FactorLightGBMBaseline(feature_columns=feature_cols)
    model = AdaptiveEnsembleModel(
        sub_models=[model_a, model_b],
        feature_columns=feature_cols,
    )
    allocator = HourlyPortfolioAllocator(config=config.portfolio)
    executor = ExecutionEngine(cost_config=config.costs)
    risk = RiskController(
        max_drawdown=config.risk.max_drawdown_hard_stop,
        target_vol_low=config.portfolio.target_vol_lower,
        target_vol_high=config.portfolio.target_vol_upper,
    )
    engine = HourlyBacktestEngine(
        config=config,
        model=model,
        allocator=allocator,
        executor=executor,
        risk_controller=risk,
    )

    artifacts = engine.run(
        market=market,
        membership=membership,
        fundamentals=fundamentals,
        macro=macro,
        sentiment=sentiment,
        analyst=analyst,
        start="2026-01-01",
        end="2026-02-26",
        train_end="2025-10-31",
    )

    # --- Extract data ---
    summary = artifacts.performance.summary
    fills = artifacts.fills
    equity = artifacts.equity_curve
    weights = artifacts.weight_history
    meta = artifacts.metadata

    # Performance metrics
    total_return = artifacts.performance.extras.get("total_return", 0.0)
    period_start = str(equity["event_time"].iloc[0]) if not equity.empty else "N/A"
    period_end = str(equity["event_time"].iloc[-1]) if not equity.empty else "N/A"

    # Trading activity
    if not fills.empty:
        buys = fills[fills["quantity"] > 0]
        sells = fills[fills["quantity"] < 0]
        total_buys = len(buys)
        total_sells = len(sells)
        total_fills = len(fills)
        unique_symbols = int(fills["symbol"].nunique())
        total_buy_notional = float(buys["notional"].abs().sum())
        total_sell_notional = float(sells["notional"].abs().sum())
        total_notional = float(fills["notional"].abs().sum())
        total_costs = float(fills["trading_cost"].sum())
        avg_cost_per_trade = total_costs / total_fills if total_fills > 0 else 0.0
        partial_fills = int(fills["partial"].sum())

        # Per-symbol breakdown
        sym_stats = []
        for sym in sorted(fills["symbol"].unique()):
            sf = fills[fills["symbol"] == sym]
            sb = sf[sf["quantity"] > 0]
            ss = sf[sf["quantity"] < 0]
            sym_stats.append({
                "symbol": sym,
                "buys": len(sb),
                "sells": len(ss),
                "buy_notional": float(sb["notional"].abs().sum()),
                "sell_notional": float(ss["notional"].abs().sum()),
                "total_cost": float(sf["trading_cost"].sum()),
            })
    else:
        total_buys = total_sells = total_fills = unique_symbols = 0
        total_buy_notional = total_sell_notional = total_notional = 0.0
        total_costs = avg_cost_per_trade = 0.0
        partial_fills = 0
        sym_stats = []

    # --- Print markdown tables ---
    print("=" * 70)
    print("ENSEMBLE MODEL (ROUTE C) TEST-SET REPORT")
    print("=" * 70)
    print()

    print("### Performance metrics")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Model | `{meta.get('model_name', 'N/A')}` |")
    print(f"| Period start | `{period_start}` |")
    print(f"| Period end | `{period_end}` |")
    print(f"| Total return | `{total_return * 100:.4f}%` |")
    print(f"| CAGR | `{summary['cagr'] * 100:.4f}%` |")
    print(f"| Annualized volatility | `{summary['annualized_volatility'] * 100:.4f}%` |")
    print(f"| Sharpe | `{summary['sharpe']:.4f}` |")
    print(f"| Sortino | `{summary['sortino']:.4f}` |")
    print(f"| Calmar | `{summary['calmar']:.4f}` |")
    print(f"| Max drawdown | `{summary['max_drawdown'] * 100:.4f}%` |")
    print(f"| Beta to SPY | `{summary['beta_to_spy']:.4f}` |")
    print(f"| Information ratio | `{summary['information_ratio']:.4f}` |")
    print(f"| Tail ratio | `{summary['tail_ratio']:.4f}` |")
    print(f"| Turnover (avg) | `{summary['turnover']:.4f}` |")
    print(f"| Exposure (avg) | `{summary['exposure_pct'] * 100:.4f}%` |")
    print()

    print("### Trading activity summary")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Total fills | {total_fills} |")
    print(f"| Buy orders | {total_buys} |")
    print(f"| Sell orders | {total_sells} |")
    print(f"| Unique symbols traded | {unique_symbols} |")
    print(f"| Total buy notional | ${total_buy_notional:,.2f} |")
    print(f"| Total sell notional | ${total_sell_notional:,.2f} |")
    print(f"| Total notional traded | ${total_notional:,.2f} |")
    print(f"| Total trading costs | ${total_costs:,.2f} |")
    print(f"| Avg cost per trade | ${avg_cost_per_trade:,.4f} |")
    print(f"| Partial fills | {partial_fills} |")
    print(f"| Backtest periods | {meta.get('n_periods', 0)} |")
    print()

    if sym_stats:
        print("### Per-symbol trading breakdown")
        print()
        print("| Symbol | Buys | Sells | Buy notional | Sell notional | Trading cost |")
        print("|---|---:|---:|---:|---:|---:|")
        for s in sym_stats:
            print(
                f"| {s['symbol']} | {s['buys']} | {s['sells']} "
                f"| ${s['buy_notional']:,.2f} | ${s['sell_notional']:,.2f} "
                f"| ${s['total_cost']:,.2f} |"
            )
        print()

    # --- Save JSON summary ---
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    report = {
        "model_name": meta.get("model_name", ""),
        "period_start": period_start,
        "period_end": period_end,
        "total_return_pct": round(total_return * 100, 4),
        "cagr_pct": round(summary["cagr"] * 100, 4),
        "annualized_vol_pct": round(summary["annualized_volatility"] * 100, 4),
        "sharpe": round(summary["sharpe"], 4),
        "sortino": round(summary["sortino"], 4),
        "calmar": round(summary["calmar"], 4),
        "max_drawdown_pct": round(summary["max_drawdown"] * 100, 4),
        "beta_to_spy": round(summary["beta_to_spy"], 4),
        "information_ratio": round(summary["information_ratio"], 4),
        "turnover": round(summary["turnover"], 4),
        "exposure_pct": round(summary["exposure_pct"] * 100, 4),
        "total_fills": total_fills,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "unique_symbols": unique_symbols,
        "total_buy_notional": round(total_buy_notional, 2),
        "total_sell_notional": round(total_sell_notional, 2),
        "total_notional": round(total_notional, 2),
        "total_trading_costs": round(total_costs, 2),
        "partial_fills": partial_fills,
        "n_periods": meta.get("n_periods", 0),
        "per_symbol": sym_stats,
    }
    with open(out_dir / "ensemble_report_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved JSON summary to {out_dir / 'ensemble_report_summary.json'}")


if __name__ == "__main__":
    main()
