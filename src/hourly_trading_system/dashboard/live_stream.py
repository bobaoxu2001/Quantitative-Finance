"""Live queue stream analytics for real-time dashboard monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass(slots=True)
class LiveQueueSnapshot:
    orders: pd.DataFrame
    predictions: pd.DataFrame
    alerts: pd.DataFrame
    fills: pd.DataFrame
    reconciliations: pd.DataFrame
    order_updates: pd.DataFrame
    tca: pd.DataFrame
    queue_depths: dict[str, int]


def load_live_queue_snapshot(queue_dir: str | Path) -> LiveQueueSnapshot:
    base = Path(queue_dir)
    topics = ["orders", "predictions", "alerts", "fills", "reconciliation", "order_updates", "tca"]
    raw: dict[str, list[dict[str, Any]]] = {topic: _read_jsonl(base / f"{topic}.jsonl") for topic in topics}
    depths = {topic: len(rows) for topic, rows in raw.items()}

    def _topic_frame(topic: str) -> pd.DataFrame:
        rows = raw[topic]
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["topic_timestamp"] = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
        if "payload" in frame.columns:
            payload = pd.json_normalize(frame["payload"])
            payload.columns = [f"payload.{c}" for c in payload.columns]
            frame = pd.concat([frame.drop(columns=["payload"]), payload], axis=1)
        return frame

    orders = _topic_frame("orders")
    predictions = _topic_frame("predictions")
    alerts = _topic_frame("alerts")
    fills = _topic_frame("fills")
    reconciliations = _topic_frame("reconciliation")
    order_updates = _topic_frame("order_updates")
    tca = _topic_frame("tca")
    return LiveQueueSnapshot(
        orders=orders,
        predictions=predictions,
        alerts=alerts,
        fills=fills,
        reconciliations=reconciliations,
        order_updates=order_updates,
        tca=tca,
        queue_depths=depths,
    )


def compute_health_score(snapshot: LiveQueueSnapshot) -> float:
    score = 100.0
    orders = snapshot.orders.sort_values("topic_timestamp") if not snapshot.orders.empty else pd.DataFrame()
    if not orders.empty:
        tail = orders.tail(20)
        sent = float(tail.get("payload.orders_sent", pd.Series(dtype=float)).fillna(0.0).sum())
        rej = float(tail.get("payload.orders_rejected", pd.Series(dtype=float)).fillna(0.0).sum())
        if sent + rej > 0:
            rejection_rate = rej / (sent + rej)
            score -= rejection_rate * 35.0
        latest_kill = bool(tail.iloc[-1].get("payload.kill_switch", False))
        if latest_kill:
            score -= 50.0
        recon_breaks = tail.get("payload.reconciliation_breaks", pd.Series(dtype=bool)).fillna(False)
        score -= float(recon_breaks.astype(float).sum() * 2.0)

    if not snapshot.alerts.empty:
        alerts_tail = snapshot.alerts.sort_values("topic_timestamp").tail(50)
        severity = alerts_tail.get("payload.severity", pd.Series(dtype=str)).astype(str).str.lower()
        score -= float((severity == "critical").sum() * 4.0)
        score -= float((severity == "warning").sum() * 1.0)

    if not snapshot.reconciliations.empty:
        tail = snapshot.reconciliations.sort_values("topic_timestamp").tail(10)
        unresolved = tail.get("payload.summary.unresolved_orders", pd.Series(dtype=float)).fillna(0.0).sum()
        dup = tail.get("payload.summary.duplicate_fill_count", pd.Series(dtype=float)).fillna(0.0).sum()
        score -= float(unresolved * 0.5 + dup * 0.2)

    return float(max(0.0, min(100.0, score)))


def summarize_live_snapshot(snapshot: LiveQueueSnapshot) -> dict[str, Any]:
    def _native(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    summary: dict[str, Any] = {
        "total_messages": int(sum(snapshot.queue_depths.values())),
        "topics": snapshot.queue_depths,
        "latest_decision_time": None,
        "latest_equity": None,
        "latest_orders_sent": None,
        "latest_orders_rejected": None,
        "kill_switch": None,
        "recent_alerts": 0,
        "health_score": compute_health_score(snapshot),
        "tca_avg_cost_bps": None,
    }
    if not snapshot.orders.empty:
        latest = snapshot.orders.sort_values("topic_timestamp").tail(1).iloc[0]
        summary["latest_decision_time"] = _native(latest.get("payload.decision_time"))
        summary["latest_equity"] = _native(latest.get("payload.equity"))
        summary["latest_orders_sent"] = _native(latest.get("payload.orders_sent"))
        summary["latest_orders_rejected"] = _native(latest.get("payload.orders_rejected"))
        summary["kill_switch"] = _native(latest.get("payload.kill_switch"))
    if not snapshot.alerts.empty:
        summary["recent_alerts"] = int(len(snapshot.alerts.tail(20)))
    if not snapshot.tca.empty:
        val = snapshot.tca.sort_values("topic_timestamp").tail(1).iloc[0].get("payload.summary.avg_total_cost_bps")
        summary["tca_avg_cost_bps"] = _native(val)
    return summary
