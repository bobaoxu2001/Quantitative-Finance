"""Run an end-to-end demo backtest with synthetic point-in-time data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hourly_trading_system.backtest import HourlyBacktestEngine
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import ExecutionEngine
from hourly_trading_system.features.library import required_features
from hourly_trading_system.models import FactorLightGBMBaseline
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController


def _hourly_market_calendar(start: str, end: str) -> pd.DatetimeIndex:
    idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    # Keep common US cash-market active hours approximation: 14:00-20:00 UTC.
    return idx[(idx.hour >= 14) & (idx.hour <= 20) & (idx.dayofweek < 5)]


def make_synthetic_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG", "UNH"]
    sectors = {
        "SPY": "ETF",
        "AAPL": "TECH",
        "MSFT": "TECH",
        "NVDA": "TECH",
        "AMZN": "CONS_DISC",
        "GOOGL": "COMM",
        "JPM": "FIN",
        "XOM": "ENERGY",
        "JNJ": "HEALTH",
        "PG": "CONS_STPL",
        "UNH": "HEALTH",
    }
    times = _hourly_market_calendar("2023-01-01", "2026-02-26")

    records: list[dict[str, object]] = []
    for sym in symbols:
        base_price = 400.0 if sym == "SPY" else float(rng.uniform(40, 350))
        vol_scale = 0.004 if sym == "SPY" else float(rng.uniform(0.004, 0.012))
        prices = [base_price]
        for _ in range(1, len(times) + 1):
            drift = 0.00005 if sym == "SPY" else 0.00012
            shock = rng.normal(0, vol_scale)
            prices.append(prices[-1] * np.exp(drift + shock))
        prices = np.asarray(prices[1:])

        opens = prices * (1.0 + rng.normal(0.0, vol_scale / 3.0, len(prices)))
        closes = prices
        highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0, vol_scale / 4.0, len(prices))))
        lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0, vol_scale / 4.0, len(prices))))
        volume = rng.integers(300_000, 2_500_000, size=len(prices))
        shares_out = rng.integers(800_000_000, 8_000_000_000)
        for i, t in enumerate(times):
            records.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "public_release_time": t,
                    "ingest_time": t + pd.Timedelta(minutes=1),
                    "known_time": t + pd.Timedelta(minutes=1),
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": float(volume[i]),
                    "sector": sectors[sym],
                    "shares_outstanding": float(shares_out),
                }
            )
    market = pd.DataFrame(records)

    membership_rows: list[dict[str, object]] = []
    for sym in symbols:
        for t in pd.date_range("2023-01-01", "2026-02-26", freq="7D", tz="UTC"):
            is_member = sym != "SPY"
            # Simulate rare constituent exit/re-entry.
            if sym in {"PG", "UNH"} and pd.Timestamp("2025-06-01", tz="UTC") <= t <= pd.Timestamp("2025-08-31", tz="UTC"):
                is_member = False
            membership_rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "public_release_time": t,
                    "ingest_time": t + pd.Timedelta(minutes=5),
                    "known_time": t + pd.Timedelta(minutes=5),
                    "is_member": bool(is_member),
                }
            )
    membership = pd.DataFrame(membership_rows)

    fundamentals_rows: list[dict[str, object]] = []
    for sym in symbols:
        if sym == "SPY":
            continue
        for t in pd.date_range("2023-01-01", "2026-02-26", freq="30D", tz="UTC"):
            fundamentals_rows.append(
                {
                    "symbol": sym,
                    "event_time": t - pd.Timedelta(days=1),
                    "public_release_time": t,
                    "ingest_time": t + pd.Timedelta(minutes=3),
                    "known_time": t + pd.Timedelta(minutes=3),
                    "eps_surprise_std": float(rng.normal(0, 1)),
                    "revenue_surprise_std": float(rng.normal(0, 1)),
                    "guidance_sentiment": float(rng.uniform(-1, 1)),
                    "est_revision_rate": float(rng.normal(0, 0.15)),
                    "days_since_filing": float(rng.integers(1, 90)),
                    "gross_margin_ttm": float(rng.uniform(0.15, 0.75)),
                    "roe_ttm": float(rng.uniform(0.05, 0.45)),
                    "net_debt_ebitda": float(rng.uniform(0.2, 5.0)),
                }
            )
    fundamentals = pd.DataFrame(fundamentals_rows)

    macro_rows: list[dict[str, object]] = []
    vix = 18.0
    y2 = 4.2
    y10 = 4.0
    dxy = 103.0
    wti = 75.0
    gold = 1900.0
    for t in times:
        vix = max(8.0, vix + rng.normal(0, 0.25))
        y2 = y2 + rng.normal(0, 0.01)
        y10 = y10 + rng.normal(0, 0.009)
        dxy = dxy * np.exp(rng.normal(0, 0.0007))
        wti = wti * np.exp(rng.normal(0, 0.002))
        gold = gold * np.exp(rng.normal(0, 0.0015))
        macro_rows.append(
            {
                "event_time": t,
                "public_release_time": t,
                "ingest_time": t + pd.Timedelta(minutes=2),
                "known_time": t + pd.Timedelta(minutes=2),
                "vix": float(vix),
                "yield_2y": float(y2),
                "yield_10y": float(y10),
                "dxy": float(dxy),
                "wti": float(wti),
                "gold": float(gold),
                "cpi_surprise_state": float(rng.normal(0, 0.2)),
                "liquidity_stress": float(abs(rng.normal(0, 1))),
                "regime_prob": float(rng.uniform(0, 1)),
            }
        )
    macro = pd.DataFrame(macro_rows)

    sentiment_rows: list[dict[str, object]] = []
    analyst_rows: list[dict[str, object]] = []
    for sym in symbols:
        if sym == "SPY":
            continue
        for t in times:
            if rng.uniform() < 0.35:
                update_t = t + pd.Timedelta(minutes=4)
                sentiment_rows.append(
                    {
                        "symbol": sym,
                        "event_time": t,
                        "public_release_time": t,
                        "ingest_time": update_t,
                        "known_time": update_t,
                        "sentiment_score": float(rng.normal(0, 1)),
                        "sentiment_dispersion": float(abs(rng.normal(0, 1))),
                        "mention_abnormal": float(rng.normal(0, 1)),
                        "hours_since_update": 0.0,
                        "sentiment_update_time": update_t,
                    }
                )
            if rng.uniform() < 0.05:
                update_t = t + pd.Timedelta(minutes=6)
                analyst_rows.append(
                    {
                        "symbol": sym,
                        "event_time": t,
                        "public_release_time": t,
                        "ingest_time": update_t,
                        "known_time": update_t,
                        "net_upgrades_downgrades": float(rng.normal(0, 1)),
                        "target_price_revision": float(rng.normal(0, 0.02)),
                        "hours_since_update": 0.0,
                        "analyst_update_time": update_t,
                    }
                )
    sentiment = pd.DataFrame(sentiment_rows)
    analyst = pd.DataFrame(analyst_rows)
    return market, membership, fundamentals, macro, sentiment, analyst


def main() -> None:
    market, membership, fundamentals, macro, sentiment, analyst = make_synthetic_market_data()

    config = SystemConfig()
    model = FactorLightGBMBaseline(feature_columns=required_features())
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
        start="2025-11-01",
        end="2026-02-26",
        train_end="2025-10-31",
    )
    print("=== SUMMARY METRICS ===")
    for k, v in artifacts.performance.summary.items():
        print(f"{k:32s}: {v}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    artifacts.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    artifacts.fills.to_csv(out_dir / "fills.csv", index=False)
    artifacts.weight_history.to_csv(out_dir / "weights.csv", index=False)
    artifacts.risk_history.to_csv(out_dir / "risk_history.csv", index=False)
    artifacts.performance.monthly_returns.to_csv(out_dir / "monthly_returns.csv")
    print("Saved artifacts in outputs/")


if __name__ == "__main__":
    main()
