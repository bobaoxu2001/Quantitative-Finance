"""Template: broker WebSocket listener forwarding order updates into live queue."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
import json

from hourly_trading_system.execution import (
    BrokerWebSocketClient,
    HMACRequestSigner,
    parse_ws_order_status_message,
)
from hourly_trading_system.live import FileBackedRealtimeQueue, QueueMessage


def main() -> None:
    ws_url = os.environ.get("BROKER_WS_URL", "")
    api_key = os.environ.get("BROKER_API_KEY", "")
    api_secret = os.environ.get("BROKER_API_SECRET", "")
    passphrase = os.environ.get("BROKER_API_PASSPHRASE")
    queue_path = os.environ.get("LIVE_QUEUE_PATH", "outputs/live_queue")
    sequence_field = os.environ.get("BROKER_WS_SEQUENCE_FIELD", "sequence")
    sequence_state_path = os.environ.get("BROKER_WS_SEQUENCE_STATE_PATH", "outputs/ws_sequence.state")
    replay_url = os.environ.get("BROKER_REPLAY_URL")
    if not ws_url or not api_key or not api_secret:
        raise RuntimeError("Set BROKER_WS_URL, BROKER_API_KEY, BROKER_API_SECRET")

    signer = HMACRequestSigner(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
    queue = FileBackedRealtimeQueue(queue_path)

    def on_message(message: dict) -> None:
        parsed = parse_ws_order_status_message(message)
        queue.publish(
            QueueMessage(
                topic="order_updates",
                payload={"source": "ws", "raw": message, "parsed": parsed.to_dict() if parsed else None},
            )
        )
        if parsed and parsed.last_fill is not None:
            queue.publish(
                QueueMessage(
                    topic="fills",
                    payload=parsed.last_fill.to_dict(),
                )
            )

    def on_error(exc: Exception) -> None:
        queue.publish(
            QueueMessage(
                topic="alerts",
                payload={
                    "severity": "critical",
                    "source": "broker_ws_listener",
                    "message": str(exc),
                },
            )
        )

    def subscribe_msg(last_sequence: int | None = None) -> dict:
        payload = {"op": "subscribe", "args": ["orders", "fills"]}
        if last_sequence is not None:
            payload["resume_from_sequence"] = last_sequence
        return payload

    def gap_recovery(last_sequence: int, current_sequence: int) -> list[dict]:
        if not replay_url:
            return []
        query = urllib.parse.urlencode(
            {
                "from_sequence": last_sequence + 1,
                "to_sequence": current_sequence - 1,
            }
        )
        parsed = urllib.parse.urlparse(replay_url)
        path = parsed.path + (f"?{query}" if query else "")
        headers = signer.auth_headers(method="GET", path=path, body="")
        request = urllib.request.Request(url=f"{replay_url}?{query}", method="GET", headers=headers)
        with urllib.request.urlopen(request, timeout=8) as response:
            text = response.read().decode("utf-8")
        payload = json.loads(text) if text else {}
        events = payload.get("events", [])
        queue.publish(
            QueueMessage(
                topic="order_updates",
                payload={
                    "source": "ws_replay",
                    "from_sequence": last_sequence + 1,
                    "to_sequence": current_sequence - 1,
                    "count": len(events),
                },
            )
        )
        return events

    client = BrokerWebSocketClient(
        ws_url=ws_url,
        signer=signer,
        on_message=on_message,
        on_error=on_error,
        subscribe_message_factory=subscribe_msg,
        heartbeat_interval_seconds=15.0,
        sequence_field=sequence_field,
        sequence_state_path=sequence_state_path,
        gap_recovery_callback=gap_recovery,
    )
    print("Starting broker WebSocket listener...")
    client.run_forever()


if __name__ == "__main__":
    main()
