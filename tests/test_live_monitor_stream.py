from __future__ import annotations

from hourly_trading_system.dashboard import load_live_queue_snapshot, summarize_live_snapshot
from hourly_trading_system.live import FileBackedRealtimeQueue, QueueMessage


def test_live_queue_snapshot_reads_stream_topics(tmp_path) -> None:
    queue = FileBackedRealtimeQueue(tmp_path / "queue")
    queue.publish(
        QueueMessage(
            topic="orders",
            payload={
                "decision_time": "2026-02-26T15:00:00+00:00",
                "equity": 101000.0,
                "orders_sent": 4,
                "orders_rejected": 0,
                "kill_switch": False,
            },
        )
    )
    queue.publish(
        QueueMessage(
            topic="alerts",
            payload={"severity": "warning", "source": "live_runner", "message": "test-alert"},
        )
    )
    snapshot = load_live_queue_snapshot(tmp_path / "queue")
    summary = summarize_live_snapshot(snapshot)
    assert summary["total_messages"] >= 2
    assert summary["latest_equity"] == 101000.0
    assert summary["recent_alerts"] >= 1
