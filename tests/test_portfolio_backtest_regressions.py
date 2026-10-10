"""Regression tests for portfolio, execution, risk and backtest accounting fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hourly_trading_system.backtest import HourlyBacktestEngine
from hourly_trading_system.config import CostModelConfig, PortfolioConfig, SystemConfig
from hourly_trading_system.execution import ExecutionEngine
from hourly_trading_system.features import required_features
from hourly_trading_system.models import FactorLightGBMBaseline
from hourly_trading_system.portfolio import HourlyPortfolioAllocator, estimate_portfolio_volatility
from hourly_trading_system.risk import RiskController

# ------------------------------------------------------------------------ allocator


def _snapshot(symbols: list[str], rv_20h: float = 0.02) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "close": 100.0,
            "volume": 1_000_000.0,
            "sector": [f"S{i}" for i in range(len(symbols))],
            "rv_20h": rv_20h,
        }
    )


def test_turnover_cap_applies_to_liquidating_names_outside_target() -> None:
    config = PortfolioConfig(max_turnover_per_rebalance=0.10, max_sector_weight=1.0)
    allocator = HourlyPortfolioAllocator(config=config)
    preds = pd.DataFrame(
        {"symbol": ["NEW1", "NEW2"], "expected_excess_return": [0.4, 0.3], "downside_probability": [0.0, 0.0]}
    )
    current = pd.Series({"OLD": 0.10})  # held, but no longer predicted
    decision = allocator.allocate(pd.Timestamp("2026-01-05", tz="UTC"), preds, _snapshot(["NEW1", "NEW2"]), current)
    target = decision.weights.reindex(["OLD", "NEW1", "NEW2"]).fillna(0.0)
    true_turnover = float((target - current.reindex(target.index).fillna(0.0)).abs().sum())
    assert decision.turnover == pytest.approx(true_turnover)
    assert true_turnover <= config.max_turnover_per_rebalance + 1e-9
    assert target["OLD"] > 0.0  # liquidation is spread out under the budget


def test_portfolio_vol_uses_hourly_units_and_correlation() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    hourly = pd.Series({"A": 0.01, "B": 0.01})
    diag = estimate_portfolio_volatility(weights, hourly, assumed_correlation=0.0, periods_per_year=1600)
    full = estimate_portfolio_volatility(weights, hourly, assumed_correlation=1.0, periods_per_year=1600)
    assert diag == pytest.approx(np.sqrt(0.5) * 0.01 * 40)
    assert full == pytest.approx(0.01 * 40)

    # rv_20h is a 20-bar horizon vol: the allocator must convert it to hourly
    # (divide by sqrt(20)) before annualising, otherwise risk is overstated ~4.5x.
    config = PortfolioConfig(max_position_weight=0.5, max_sector_weight=1.0, max_turnover_per_rebalance=2.0)
    allocator = HourlyPortfolioAllocator(config=config, assumed_correlation=0.0)
    preds = pd.DataFrame({"symbol": ["A", "B"], "expected_excess_return": [0.3, 0.3], "downside_probability": 0.0})
    hourly_vol = 0.004
    snap = _snapshot(["A", "B"], rv_20h=hourly_vol * np.sqrt(20))
    decision = allocator.allocate(pd.Timestamp("2026-01-05", tz="UTC"), preds, snap, pd.Series(dtype=float))
    expected = np.sqrt(2 * (0.5 * hourly_vol) ** 2) * np.sqrt(252 * 6.5)
    assert decision.estimated_portfolio_vol == pytest.approx(expected)


# ------------------------------------------------------------------------ execution


def test_execution_sells_before_buys_and_never_borrows_cash() -> None:
    engine = ExecutionEngine(CostModelConfig(max_participation_rate=0.05))
    market_next = pd.DataFrame({"symbol": ["OLD", "NEW"], "open": [100.0, 100.0], "volume": [200.0, 1_000_000.0]})
    # OLD can only sell 10 shares (5% of 200) but the target wants to buy NEW with the full equity.
    result = engine.execute_target_weights(
        timestamp=pd.Timestamp("2026-01-05 15:00", tz="UTC"),
        equity=10_000.0,
        current_positions={"OLD": 100.0},
        cash=0.0,
        target_weights=pd.Series({"NEW": 1.0}),
        market_next_open=market_next,
    )
    assert result.cash >= -1e-9
    assert result.fills[0].symbol == "OLD"  # sells first
    assert result.positions["OLD"] == pytest.approx(90.0)
    assert "NEW" in result.unfilled_symbols  # buy scaled to the cash actually available


def test_empty_target_liquidates_existing_positions() -> None:
    engine = ExecutionEngine(CostModelConfig(max_participation_rate=1.0))
    result = engine.execute_target_weights(
        timestamp=pd.Timestamp("2026-01-05 15:00", tz="UTC"),
        equity=10_000.0,
        current_positions={"AAA": 50.0},
        cash=5_000.0,
        target_weights=pd.Series(dtype=float),
        market_next_open=pd.DataFrame({"symbol": ["AAA"], "open": [100.0], "volume": [1_000_000.0]}),
    )
    assert "AAA" not in result.positions
    assert result.cash > 5_000.0


def test_mark_to_market_uses_fallback_price_for_missing_bars() -> None:
    marked = ExecutionEngine.mark_to_market(
        positions={"AAA": 10.0, "HALTED": 5.0},
        cash=100.0,
        market_snapshot=pd.DataFrame({"symbol": ["AAA"], "close": [10.0]}),
        fallback_prices={"HALTED": 20.0},
    )
    assert marked["equity"] == pytest.approx(100.0 + 100.0 + 100.0)


# ----------------------------------------------------------------------------- risk


def test_risk_off_rearms_after_cooldown_instead_of_deadlocking() -> None:
    risk = RiskController(max_drawdown=0.20, risk_off_cooldown_periods=5, min_conservative_periods=2)
    risk.update(100.0)
    assert risk.update(89.0).regime == "conservative"  # 11% drawdown
    assert risk.update(79.0).regime == "risk_off"
    assert risk.capital_scaler() == 0.0
    # Flat book: equity cannot recover while risk-off.
    regimes = [risk.update(79.0).regime for _ in range(5)]
    assert regimes[-1] == "conservative"
    assert risk.capital_scaler() == 0.5
    regimes = [risk.update(79.5).regime for _ in range(3)]
    assert regimes[-1] == "normal"


def test_risk_off_without_cooldown_keeps_legacy_behaviour() -> None:
    risk = RiskController(max_drawdown=0.20, risk_off_cooldown_periods=None)
    risk.update(100.0)
    risk.update(79.0)
    assert all(risk.update(79.0).regime == "risk_off" for _ in range(50))


def test_register_return_keeps_histories_aligned() -> None:
    risk = RiskController()
    risk.register_return(0.01, benchmark_return=None)
    risk.register_return(0.02, benchmark_return=0.005)
    risk.update(100.0, benchmark_return=0.3)
    assert risk.returns_history == [0.01, 0.02]
    assert risk.benchmark_history == [0.0, 0.005]


# ------------------------------------------------------------------------- backtest


def _market_with_exit(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    rng = np.random.default_rng(seed)
    symbols = ["SPY", "AAA", "BBB", "CCC", "DDD"]
    times = pd.date_range("2025-09-01", "2025-10-20", freq="h", tz="UTC")
    times = times[(times.hour >= 14) & (times.hour <= 20) & (times.dayofweek < 5)]
    rows = []
    for sym in symbols:
        price = 100.0
        for t in times:
            price *= float(np.exp(rng.normal(0.0002, 0.006)))
            rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "open": price * (1 + rng.normal(0, 0.001)),
                    "high": price * 1.003,
                    "low": price * 0.997,
                    "close": price,
                    "volume": 2_000_000.0,
                    "sector": "ETF" if sym == "SPY" else f"SEC_{sym}",
                }
            )
    market = pd.DataFrame(rows)
    exit_time = pd.Timestamp("2025-10-08", tz="UTC")
    membership_rows = []
    for sym in symbols:
        for t in pd.date_range("2025-08-25", "2025-10-20", freq="D", tz="UTC"):
            member = sym != "SPY" and not (sym == "AAA" and t >= exit_time)
            membership_rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "public_release_time": t,
                    "ingest_time": t,
                    "known_time": t + pd.Timedelta(minutes=1),
                    "is_member": member,
                }
            )
    return market, pd.DataFrame(membership_rows), exit_time


def _engine(config: SystemConfig) -> HourlyBacktestEngine:
    return HourlyBacktestEngine(
        config=config,
        model=FactorLightGBMBaseline(feature_columns=required_features()),
        allocator=HourlyPortfolioAllocator(
            config=PortfolioConfig(max_position_weight=0.3, max_sector_weight=1.0, max_turnover_per_rebalance=2.0),
            require_positive_scores=False,
            min_total_exposure=0.9,
        ),
        executor=ExecutionEngine(cost_config=config.costs),
        risk_controller=RiskController(),
    )


def test_backtest_accounting_benchmark_and_universe_exit() -> None:
    market, membership, exit_time = _market_with_exit()
    config = SystemConfig(initial_capital=100_000.0)
    result = _engine(config).run(
        market=market, membership=membership, train_end="2025-09-30", start="2025-10-01", end="2025-10-20"
    )
    curve = result.equity_curve
    # Initial capital is the first point of the equity curve.
    assert curve["equity"].iloc[0] == pytest.approx(config.initial_capital)
    assert curve["event_time"].iloc[0] == pd.Timestamp("2025-10-01 14:00", tz="UTC")
    # SPY is not investable, but the benchmark must still be measured.
    assert np.isfinite(result.performance.summary["beta_to_spy"])
    assert result.performance.summary["information_ratio"] != pytest.approx(result.performance.summary["sharpe"])
    # AAA leaves the universe: it is never newly bought and is fully liquidated
    # (never marked at zero / stranded).
    weights = result.weight_history
    after_exit = weights.loc[(weights["symbol"] == "AAA") & (weights["event_time"] > exit_time + pd.Timedelta(days=1))]
    assert after_exit.empty
    fills = result.fills
    aaa_net = fills.loc[fills["symbol"] == "AAA", "quantity"].sum()
    assert aaa_net == pytest.approx(0.0, abs=1e-6)
    # Realised exposure is reported from holdings and equity never jumps down
    # by a whole position when a name exits.
    assert (curve["exposure"].iloc[1:] > 0.5).all()
    assert curve["equity"].pct_change().abs().max() < 0.05
    # Training was purged and never overlaps the evaluation window.
    assert result.metadata["in_sample_overlap"] is False
    assert pd.Timestamp(result.metadata["train_end"]) < curve["event_time"].iloc[0]


def test_backtest_refuses_to_train_on_future_data() -> None:
    market, membership, _ = _market_with_exit()
    with pytest.raises(ValueError, match="No valid training rows"):
        _engine(SystemConfig()).run(
            market=market, membership=membership, train_end="2025-08-01", start="2025-10-01", end="2025-10-20"
        )
