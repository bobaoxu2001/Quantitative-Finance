"""Performance and risk metric calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 252 * 6.5
SECONDS_PER_YEAR = 365.25 * 24 * 3600
_MIN_SPAN_DAYS_FOR_INFERENCE = 5.0


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


def infer_periods_per_year(index: pd.DatetimeIndex, default: float = HOURS_PER_YEAR) -> float:
    """
    Infer the number of return periods per year from observation timestamps.

    Uses observed periods per elapsed calendar year, so session calendars
    (7 bars per weekday, 24/7 crypto, holidays) are handled without a hard-coded
    constant. Falls back to ``default`` for spans shorter than a few days.
    """
    if len(index) < 3:
        return float(default)
    span_seconds = (index.max() - index.min()).total_seconds()
    if span_seconds < _MIN_SPAN_DAYS_FOR_INFERENCE * 86400:
        return float(default)
    return float((len(index) - 1) / (span_seconds / SECONDS_PER_YEAR))


def _annualize_return(total_return: float, years: float) -> float:
    if years <= 0 or not np.isfinite(years) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _annualize_vol(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def _information_ratio(active_returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    tracking_error = active_returns.std(ddof=0)
    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan
    return float(active_returns.mean() / tracking_error * np.sqrt(periods_per_year))


def _beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 3:
        return np.nan
    var_b = benchmark_returns.var(ddof=0)
    if var_b == 0:
        return np.nan
    cov = np.cov(portfolio_returns, benchmark_returns, ddof=0)[0, 1]
    return float(cov / var_b)


def _sortino(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """
    Sortino ratio with the standard downside deviation (target return 0).

    Downside deviation is ``sqrt(mean(min(r, 0)^2))`` over *all* periods -- not
    the standard deviation of the negative returns only, which ignores how often
    losses occur and is centred on the wrong mean.
    """
    if returns.empty:
        return np.nan
    downside_dev = float(np.sqrt(np.mean(np.minimum(returns.to_numpy(dtype=float), 0.0) ** 2)))
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return float(returns.mean() / downside_dev * np.sqrt(periods_per_year))


def _monthly_returns(returns: pd.Series) -> pd.DataFrame:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    table = monthly.to_frame(name="monthly_return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="monthly_return").sort_index()


def closed_trades_from_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """
    Derive realised trades from signed fills using average-cost accounting.

    Every fill that reduces a long position closes part of it and produces one
    record with realised ``pnl`` (``qty * (fill_price - avg_cost)``) and
    ``holding_hours`` measured from the quantity-weighted average entry time.
    Fill prices are assumed to already include transaction costs.

    Required columns: symbol, event_time, quantity (signed), fill_price.
    """
    required = {"symbol", "event_time", "quantity", "fill_price"}
    if fills is None or fills.empty or not required.issubset(fills.columns):
        return pd.DataFrame(columns=["symbol", "event_time", "quantity", "pnl", "return", "holding_hours"])

    ordered = fills.copy()
    ordered["event_time"] = pd.to_datetime(ordered["event_time"], utc=True)
    ordered = ordered.sort_values(["event_time", "symbol"], kind="stable")
    book: dict[str, tuple[float, float, float]] = {}  # symbol -> (qty, avg_cost, avg_entry_seconds)
    records: list[dict[str, Any]] = []
    for symbol, ts, qty, price in ordered[["symbol", "event_time", "quantity", "fill_price"]].itertuples(
        index=False, name=None
    ):
        qty = float(qty)
        price = float(price)
        t_sec = ts.timestamp()
        held, avg_cost, entry = book.get(symbol, (0.0, 0.0, t_sec))
        if qty > 0:
            new_qty = held + qty
            avg_cost = (held * avg_cost + qty * price) / new_qty
            entry = (held * entry + qty * t_sec) / new_qty
            book[symbol] = (new_qty, avg_cost, entry)
        elif qty < 0 and held > 0:
            closed = min(-qty, held)
            pnl = closed * (price - avg_cost)
            records.append(
                {
                    "symbol": symbol,
                    "event_time": ts,
                    "quantity": closed,
                    "pnl": pnl,
                    "return": price / avg_cost - 1.0 if avg_cost > 0 else np.nan,
                    "holding_hours": (t_sec - entry) / 3600.0,
                }
            )
            remaining = held - closed
            book[symbol] = (remaining, avg_cost, entry) if remaining > 1e-12 else (0.0, 0.0, t_sec)
    return pd.DataFrame(
        records, columns=["symbol", "event_time", "quantity", "pnl", "return", "holding_hours"]
    )


def _trade_stats(trades: pd.DataFrame | None) -> dict[str, float]:
    if trades is not None and not trades.empty and "pnl" not in trades.columns:
        closed = closed_trades_from_fills(trades)
        if not closed.empty:
            stats = _trade_stats(closed)
            stats["trade_count"] = float(len(trades))
            stats["closed_trade_count"] = float(len(closed))
            return stats
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
        "closed_trade_count": float(len(trades)),
    }


def compute_performance_report(
    equity_curve: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    turnover_series: pd.Series | None = None,
    exposure_series: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    periods_per_year: float | None = None,
) -> PerformanceReport:
    """
    Compute institutional performance metrics.

    equity_curve required columns:
      event_time, equity
    Optional:
      cash, invested

    ``periods_per_year`` controls annualisation of volatility-based ratios; by
    default it is inferred from the equity-curve timestamps. CAGR always uses
    elapsed calendar time. ``trades`` may be realised trades with a ``pnl``
    column or raw signed fills (symbol, event_time, quantity, fill_price), in
    which case realised trades are derived with average-cost accounting.
    """
    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty")
    curve = equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    curve = curve.sort_values("event_time").set_index("event_time")
    curve["returns"] = curve["equity"].pct_change().fillna(0.0)
    returns = curve["returns"]

    ppy = float(periods_per_year) if periods_per_year else infer_periods_per_year(curve.index)
    total_return = float(curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0)
    elapsed_years = (curve.index[-1] - curve.index[0]).total_seconds() / SECONDS_PER_YEAR
    if elapsed_years <= 0:
        elapsed_years = max(len(curve) - 1, 0) / ppy
    cagr = _annualize_return(total_return, elapsed_years)
    annual_vol = _annualize_vol(returns, ppy)
    ret_std = returns.std(ddof=0)
    sharpe = float(returns.mean() / ret_std * np.sqrt(ppy)) if ret_std > 0 else np.nan
    sortino = _sortino(returns, ppy)
    max_dd = _max_drawdown(curve["equity"])
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    drawdown_curve = curve["equity"] / curve["equity"].cummax() - 1.0

    benchmark_returns = benchmark_returns.reindex(curve.index).fillna(0.0) if benchmark_returns is not None else None
    active_returns = returns - benchmark_returns if benchmark_returns is not None else pd.Series(dtype=float)
    beta_spy = _beta(returns, benchmark_returns) if benchmark_returns is not None else np.nan
    information_ratio = _information_ratio(active_returns, ppy) if benchmark_returns is not None else np.nan

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
        "closed_trade_count": trade_stats.get("closed_trade_count", np.nan),
        "periods_per_year": ppy,
        "returns_series": returns,
    }
    return PerformanceReport(
        summary=summary,
        equity_curve=curve.reset_index(),
        drawdown_curve=drawdown_curve,
        monthly_returns=monthly_returns,
        extras=extras,
    )
