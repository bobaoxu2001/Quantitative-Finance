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
    queue_depths: dict[str, int]


def load_live_queue_snapshot(queue_dir: str | Path) -> LiveQueueSnapshot:
    base = Path(queue_dir)
    topics = ["orders", "predictions", "alerts", "fills", "reconciliation", "order_updates"]
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
    return LiveQueueSnapshot(
        orders=orders,
        predictions=predictions,
        alerts=alerts,
        fills=fills,
        reconciliations=reconciliations,
        order_updates=order_updates,
        queue_depths=depths,
    )


def summarize_live_snapshot(snapshot: LiveQueueSnapshot) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_messages": int(sum(snapshot.queue_depths.values())),
        "topics": snapshot.queue_depths,
        "latest_decision_time": None,
        "latest_equity": None,
        "latest_orders_sent": None,
        "latest_orders_rejected": None,
        "kill_switch": None,
        "recent_alerts": 0,
    }
    if not snapshot.orders.empty:
        latest = snapshot.orders.sort_values("topic_timestamp").tail(1).iloc[0]
        summary["latest_decision_time"] = latest.get("payload.decision_time")
        summary["latest_equity"] = latest.get("payload.equity")
        summary["latest_orders_sent"] = latest.get("payload.orders_sent")
        summary["latest_orders_rejected"] = latest.get("payload.orders_rejected")
        summary["kill_switch"] = latest.get("payload.kill_switch")
    if not snapshot.alerts.empty:
        summary["recent_alerts"] = int(len(snapshot.alerts.tail(20)))
    return summary
