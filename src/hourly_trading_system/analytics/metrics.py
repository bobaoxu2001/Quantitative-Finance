"""Performance and risk metric calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 252 * 6.5


@dataclass(slots=True)
class PerformanceReport:
    summary: dict[str, float]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.Series
    monthly_returns: pd.DataFrame
    extras: dict[str, Any]


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _tail_ratio(returns: pd.Series) -> float:
    q95 = returns.quantile(0.95)
    q05 = returns.quantile(0.05)
    if q05 == 0:
        return np.nan
    return float(abs(q95 / q05))


def _annualize_return(total_return: float, n_periods: int) -> float:
    years = n_periods / HOURS_PER_YEAR
    if years <= 0:
        return np.nan
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _annualize_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(HOURS_PER_YEAR))


def _information_ratio(active_returns: pd.Series) -> float:
    tracking_error = active_returns.std(ddof=0)
    if tracking_error == 0:
        return np.nan
    return float(active_returns.mean() / tracking_error * np.sqrt(HOURS_PER_YEAR))


def _beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 3:
        return np.nan
    var_b = benchmark_returns.var(ddof=0)
    if var_b == 0:
        return np.nan
    cov = np.cov(portfolio_returns, benchmark_returns, ddof=0)[0, 1]
    return float(cov / var_b)


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float(returns.mean() / downside_std * np.sqrt(HOURS_PER_YEAR))


def _monthly_returns(returns: pd.Series) -> pd.DataFrame:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    table = monthly.to_frame(name="monthly_return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="monthly_return").sort_index()


def _trade_stats(trades: pd.DataFrame | None) -> dict[str, float]:
    if trades is None or trades.empty:
        return {
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "average_holding_period_hours": np.nan,
            "trade_count": 0.0,
        }
    if "pnl" not in trades.columns:
        return {
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "average_holding_period_hours": float(
                trades["holding_hours"].mean() if "holding_hours" in trades.columns else np.nan
            ),
            "trade_count": float(len(trades)),
        }
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    profit_factor = float(wins / abs(losses)) if losses != 0 else np.nan
    win_rate = float((pnl > 0).mean()) if len(pnl) else np.nan
    avg_hold = float(trades["holding_hours"].mean()) if "holding_hours" in trades.columns else np.nan
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_holding_period_hours": avg_hold,
        "trade_count": float(len(trades)),
    }


def compute_performance_report(
    equity_curve: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    turnover_series: pd.Series | None = None,
    exposure_series: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> PerformanceReport:
    """
    Compute institutional performance metrics.

    equity_curve required columns:
      event_time, equity
    Optional:
      cash, invested
    """
    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty")
    curve = equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    curve = curve.sort_values("event_time").set_index("event_time")
    curve["returns"] = curve["equity"].pct_change().fillna(0.0)
    returns = curve["returns"]

    total_return = float(curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0)
    cagr = _annualize_return(total_return, len(curve))
    annual_vol = _annualize_vol(returns)
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(HOURS_PER_YEAR)) if returns.std(ddof=0) > 0 else np.nan
    sortino = _sortino(returns)
    max_dd = _max_drawdown(curve["equity"])
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    drawdown_curve = curve["equity"] / curve["equity"].cummax() - 1.0

    benchmark_returns = benchmark_returns.reindex(curve.index).fillna(0.0) if benchmark_returns is not None else None
    active_returns = returns - benchmark_returns if benchmark_returns is not None else pd.Series(dtype=float)
    beta_spy = _beta(returns, benchmark_returns) if benchmark_returns is not None else np.nan
    information_ratio = _information_ratio(active_returns) if benchmark_returns is not None else np.nan

    trade_stats = _trade_stats(trades)
    turnover = float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) else np.nan
    exposure = float(exposure_series.mean()) if exposure_series is not None and len(exposure_series) else np.nan

    summary = {
        "cagr": cagr,
        "annualized_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": float(abs(max_dd)),
        "win_rate": trade_stats["win_rate"],
        "profit_factor": trade_stats["profit_factor"],
        "turnover": turnover,
        "average_holding_period_hours": trade_stats["average_holding_period_hours"],
        "exposure_pct": exposure,
        "beta_to_spy": beta_spy,
        "information_ratio": information_ratio,
        "tail_ratio": _tail_ratio(returns),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurt()),
    }

    monthly_returns = _monthly_returns(returns)
    extras = {
        "total_return": total_return,
        "trade_count": trade_stats["trade_count"],
        "returns_series": returns,
    }
    return PerformanceReport(
        summary=summary,
        equity_curve=curve.reset_index(),
        drawdown_curve=drawdown_curve,
        monthly_returns=monthly_returns,
        extras=extras,
    )
