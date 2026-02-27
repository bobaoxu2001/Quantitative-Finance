"""Launch a simple Streamlit dashboard from saved backtest outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from hourly_trading_system.dashboard import DashboardDataBundle, build_dashboard_payload


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(layout="wide", page_title="Hourly Trading System Dashboard")
    out_dir = Path("outputs")
    equity = _load_csv(out_dir / "equity_curve.csv")
    fills = _load_csv(out_dir / "fills.csv")
    weights = _load_csv(out_dir / "weights.csv")
    risk = _load_csv(out_dir / "risk_history.csv")
    monthly = _load_csv(out_dir / "monthly_returns.csv")

    if equity.empty:
        st.warning("No output files found. Run scripts/run_demo_backtest.py first.")
        return

    for frame in (equity, fills, weights, risk):
        if "event_time" in frame.columns:
            frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)

    summary = {}
    if "equity" in equity.columns and len(equity) > 1:
        summary["total_return"] = float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0)
        summary["current_drawdown"] = float(1 - equity["equity"].iloc[-1] / equity["equity"].cummax().iloc[-1])

    bundle = DashboardDataBundle(
        equity_curve=equity,
        risk_history=risk,
        signals=pd.DataFrame(),
        fills=fills,
        weights=weights,
        monthly_returns=monthly,
        metrics_summary=summary,
    )
    payload = build_dashboard_payload(bundle)

    st.title("Institutional Hourly Trading Dashboard")
    st.subheader("Summary Metrics")
    st.json(payload["summary_metrics"])

    st.subheader("Portfolio Overview")
    if hasattr(payload["portfolio_overview"], "to_dict"):
        st.plotly_chart(payload["portfolio_overview"], use_container_width=True)

    st.subheader("Risk Panel")
    if hasattr(payload["risk_panel"], "to_dict"):
        st.plotly_chart(payload["risk_panel"], use_container_width=True)

    st.subheader("Trade Analytics")
    if hasattr(payload["trade_analytics"], "to_dict"):
        st.plotly_chart(payload["trade_analytics"], use_container_width=True)

    st.subheader("Capital Allocation")
    if hasattr(payload["capital_allocation"], "to_dict"):
        st.plotly_chart(payload["capital_allocation"], use_container_width=True)

    st.subheader("Monthly Returns Table")
    st.dataframe(payload["monthly_returns_table"])


if __name__ == "__main__":
    main()
