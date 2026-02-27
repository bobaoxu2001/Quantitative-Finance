from __future__ import annotations

import numpy as np
import pandas as pd

from hourly_trading_system.backtest import HourlyBacktestEngine
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import ExecutionEngine
from hourly_trading_system.features import required_features
from hourly_trading_system.models import FactorLightGBMBaseline
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController


def _make_minimal_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    symbols = ["SPY", "AAPL", "MSFT", "XOM"]
    sectors = {"SPY": "ETF", "AAPL": "TECH", "MSFT": "TECH", "XOM": "ENERGY"}
    times = pd.date_range("2025-09-01", periods=180, freq="h", tz="UTC")
    times = times[(times.hour >= 14) & (times.hour <= 20) & (times.dayofweek < 5)]
    rows = []
    for sym in symbols:
        p = 400.0 if sym == "SPY" else float(rng.uniform(80, 250))
        for t in times:
            p *= np.exp(rng.normal(0.00005, 0.006))
            open_ = p * (1 + rng.normal(0, 0.001))
            close = p
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.0015)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.0015)))
            volume = float(rng.integers(450_000, 2_000_000))
            rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "sector": sectors[sym],
                    "shares_outstanding": float(rng.integers(1_000_000_000, 6_000_000_000)),
                }
            )
    market = pd.DataFrame(rows)

    membership_rows = []
    for sym in symbols:
        for t in pd.date_range("2025-08-01", "2026-01-01", freq="10D", tz="UTC"):
            membership_rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "public_release_time": t,
                    "ingest_time": t + pd.Timedelta(minutes=2),
                    "known_time": t + pd.Timedelta(minutes=2),
                    "is_member": sym != "SPY",
                }
            )
    membership = pd.DataFrame(membership_rows)
    return market, membership


def test_backtest_engine_runs_end_to_end() -> None:
    market, membership = _make_minimal_market()
    config = SystemConfig(initial_capital=100_000.0)
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
    result = engine.run(
        market=market,
        membership=membership,
        start="2025-10-01",
        end="2025-12-15",
        train_end="2025-09-30",
    )
    assert not result.equity_curve.empty
    assert "max_drawdown" in result.performance.summary
    assert result.metadata["n_periods"] > 0
