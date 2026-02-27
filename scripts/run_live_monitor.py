"""Run real-time live monitoring dashboard from queue stream topics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from hourly_trading_system.dashboard import load_live_queue_snapshot, summarize_live_snapshot
from hourly_trading_system.governance import ModelRegistry


def _render_orders(snapshot_orders: pd.DataFrame) -> None:
    st.subheader("Order Flow")
    if snapshot_orders.empty:
        st.info("No orders topic messages yet.")
        return
    orders = snapshot_orders.copy().sort_values("topic_timestamp")
    if "payload.decision_time" in orders.columns:
        orders["payload.decision_time"] = pd.to_datetime(
            orders["payload.decision_time"], utc=True, errors="coerce"
        )
    if {"payload.decision_time", "payload.equity"}.issubset(orders.columns):
        chart_data = orders[["payload.decision_time", "payload.equity"]].dropna()
        if not chart_data.empty:
            st.line_chart(chart_data.set_index("payload.decision_time"))
    cols = [c for c in orders.columns if c.startswith("payload.")]
    st.dataframe(orders[["topic_timestamp", *cols]].tail(50), use_container_width=True)


def _render_alerts(alerts: pd.DataFrame) -> None:
    st.subheader("Alerts Stream")
    if alerts.empty:
        st.success("No alerts received.")
        return
    frame = alerts.copy().sort_values("topic_timestamp")
    cols = [c for c in frame.columns if c.startswith("payload.")]
    st.dataframe(frame[["topic_timestamp", *cols]].tail(100), use_container_width=True)


def _render_fills(fills: pd.DataFrame) -> None:
    st.subheader("Fills Stream")
    if fills.empty:
        st.info("No fills received.")
        return
    frame = fills.copy().sort_values("topic_timestamp")
    cols = [c for c in frame.columns if c.startswith("payload.")]
    st.dataframe(frame[["topic_timestamp", *cols]].tail(100), use_container_width=True)


def _render_reconciliation(recon: pd.DataFrame) -> None:
    st.subheader("Reconciliation Stream")
    if recon.empty:
        st.info("No reconciliation updates.")
        return
    frame = recon.copy().sort_values("topic_timestamp")
    cols = [c for c in frame.columns if c.startswith("payload.")]
    st.dataframe(frame[["topic_timestamp", *cols]].tail(100), use_container_width=True)


def _render_order_updates(order_updates: pd.DataFrame) -> None:
    st.subheader("Order Status Updates Stream")
    if order_updates.empty:
        st.info("No order status updates.")
        return
    frame = order_updates.copy().sort_values("topic_timestamp")
    cols = [c for c in frame.columns if c.startswith("payload.")]
    st.dataframe(frame[["topic_timestamp", *cols]].tail(100), use_container_width=True)


def _render_tca(tca: pd.DataFrame) -> None:
    st.subheader("TCA Stream")
    if tca.empty:
        st.info("No TCA updates.")
        return
    frame = tca.copy().sort_values("topic_timestamp")
    cols = [c for c in frame.columns if c.startswith("payload.")]
    st.dataframe(frame[["topic_timestamp", *cols]].tail(100), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Live Queue Monitor", layout="wide")
    st.title("Live Trading Monitor (Queue Stream)")
    queue_dir = st.sidebar.text_input("Queue directory", "outputs/live_queue")
    registry_dir = st.sidebar.text_input("Model registry directory", "outputs/model_registry")
    st.sidebar.caption("Reads topic JSONL files directly from live queue backend.")

    refresh_seconds = st.sidebar.slider("Auto refresh (sec)", 2, 30, 5)
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=refresh_seconds * 1000, key="live-refresh")
    except Exception:
        if st.sidebar.button("Refresh now"):
            st.rerun()

    snapshot = load_live_queue_snapshot(Path(queue_dir))
    summary = summarize_live_snapshot(snapshot)
    deployment_state = None
    try:
        registry = ModelRegistry(root_dir=registry_dir)
        deployment_state = registry.deployment_state().to_dict()
    except Exception as exc:
        st.sidebar.warning(f"Registry unavailable: {exc}")
        registry = None

    st.sidebar.subheader("Deployment Controls")
    if registry is not None:
        st.sidebar.json(
            {
                "champion": deployment_state.get("champion_version") if deployment_state else None,
                "canary": deployment_state.get("canary_version") if deployment_state else None,
                "canary_allocation": deployment_state.get("canary_allocation") if deployment_state else None,
            }
        )
        rollback_reason = st.sidebar.text_input("Rollback reason", "manual_dashboard_rollback")
        if st.sidebar.button("Rollback to previous champion"):
            try:
                new_state = registry.rollback(reason=rollback_reason).to_dict()
                st.sidebar.success("Rollback executed.")
                st.sidebar.json(new_state)
            except Exception as exc:
                st.sidebar.error(f"Rollback failed: {exc}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Messages", summary["total_messages"])
    c2.metric("Latest Equity", summary["latest_equity"])
    c3.metric("Orders Sent", summary["latest_orders_sent"])
    c4.metric("Kill Switch", summary["kill_switch"])
    c5.metric("Health Score", f"{summary['health_score']:.1f}")
    c6.metric("TCA Avg Cost (bps)", summary["tca_avg_cost_bps"])
    st.json({"queue_depths": summary["topics"], "latest_decision_time": summary["latest_decision_time"]})

    _render_orders(snapshot.orders)
    _render_alerts(snapshot.alerts)
    _render_fills(snapshot.fills)
    _render_reconciliation(snapshot.reconciliations)
    _render_order_updates(snapshot.order_updates)
    _render_tca(snapshot.tca)


if __name__ == "__main__":
    main()
