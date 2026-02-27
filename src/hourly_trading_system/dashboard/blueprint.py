"""Institutional dashboard blueprint and panel builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PLOTLY_AVAILABLE = False


@dataclass(slots=True)
class DashboardDataBundle:
    equity_curve: pd.DataFrame
    risk_history: pd.DataFrame
    signals: pd.DataFrame
    fills: pd.DataFrame
    weights: pd.DataFrame
    monthly_returns: pd.DataFrame
    metrics_summary: dict[str, float]


def _empty_fig() -> dict[str, Any]:
    return {"type": "empty", "message": "Install plotly for interactive charts."}


def portfolio_overview_panel(bundle: DashboardDataBundle) -> dict[str, Any]:
    if not PLOTLY_AVAILABLE:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bundle.equity_curve["event_time"],
            y=bundle.equity_curve["equity"],
            mode="lines",
            name="Equity",
        )
    )
    fig.update_layout(title="Portfolio Equity Curve", template="plotly_white")
    return fig


def risk_panel(bundle: DashboardDataBundle) -> dict[str, Any]:
    if not PLOTLY_AVAILABLE or bundle.risk_history.empty:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bundle.risk_history["event_time"], y=bundle.risk_history["drawdown"], name="Drawdown"))
    fig.add_trace(go.Scatter(x=bundle.risk_history["event_time"], y=bundle.risk_history["var_95"], name="VaR 95%"))
    fig.add_trace(go.Scatter(x=bundle.risk_history["event_time"], y=bundle.risk_history["cvar_95"], name="CVaR 95%"))
    fig.update_layout(title="Risk Panel", template="plotly_white")
    return fig


def signal_diagnostics_panel(bundle: DashboardDataBundle) -> dict[str, Any]:
    if not PLOTLY_AVAILABLE or bundle.signals.empty:
        return _empty_fig()
    sample = bundle.signals.copy()
    sample = sample.sort_values("event_time").tail(1500)
    fig = px.scatter(
        sample,
        x="downside_probability",
        y="expected_excess_return",
        color="symbol",
        title="Signal Diagnostics: Return vs Downside Probability",
    )
    fig.update_layout(template="plotly_white")
    return fig


def trade_analytics_panel(bundle: DashboardDataBundle) -> dict[str, Any]:
    if not PLOTLY_AVAILABLE or bundle.fills.empty:
        return _empty_fig()
    fills = bundle.fills.copy()
    fills["trade_notional_abs"] = fills["notional"].abs()
    fig = px.histogram(fills, x="trade_notional_abs", nbins=40, title="Trade Notional Distribution")
    fig.update_layout(template="plotly_white")
    return fig


def capital_allocation_panel(bundle: DashboardDataBundle) -> dict[str, Any]:
    if not PLOTLY_AVAILABLE or bundle.weights.empty:
        return _empty_fig()
    latest_t = bundle.weights["event_time"].max()
    top = bundle.weights.loc[bundle.weights["event_time"] == latest_t].sort_values("weight", ascending=False).head(10)
    fig = px.bar(top, x="symbol", y="weight", title="Top 10 Holdings")
    fig.update_layout(template="plotly_white")
    return fig


def build_dashboard_payload(bundle: DashboardDataBundle) -> dict[str, Any]:
    """Build all required dashboard modules as figure payloads."""
    return {
        "portfolio_overview": portfolio_overview_panel(bundle),
        "risk_panel": risk_panel(bundle),
        "signal_diagnostics": signal_diagnostics_panel(bundle),
        "trade_analytics": trade_analytics_panel(bundle),
        "capital_allocation": capital_allocation_panel(bundle),
        "monthly_returns_table": bundle.monthly_returns,
        "summary_metrics": bundle.metrics_summary,
    }
