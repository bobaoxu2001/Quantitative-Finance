"""Template: broker WebSocket listener forwarding order updates into live queue."""

from __future__ import annotations

import json
import os

from hourly_trading_system.execution import BrokerWebSocketClient, HMACRequestSigner
from hourly_trading_system.live import FileBackedRealtimeQueue, QueueMessage


def main() -> None:
    ws_url = os.environ.get("BROKER_WS_URL", "")
    api_key = os.environ.get("BROKER_API_KEY", "")
    api_secret = os.environ.get("BROKER_API_SECRET", "")
    passphrase = os.environ.get("BROKER_API_PASSPHRASE")
    queue_path = os.environ.get("LIVE_QUEUE_PATH", "outputs/live_queue")
    if not ws_url or not api_key or not api_secret:
        raise RuntimeError("Set BROKER_WS_URL, BROKER_API_KEY, BROKER_API_SECRET")

    signer = HMACRequestSigner(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
    queue = FileBackedRealtimeQueue(queue_path)

    def on_message(message: dict) -> None:
        queue.publish(
            QueueMessage(
                topic="order_updates",
                payload={"source": "ws", "message": message},
            )
        )
        # Route explicit fill payload when available.
        if message.get("event") == "fill":
            queue.publish(
                QueueMessage(
                    topic="fills",
                    payload=message.get("data", {}),
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

    def subscribe_msg() -> dict:
        return {"op": "subscribe", "args": ["orders", "fills"]}

    client = BrokerWebSocketClient(
        ws_url=ws_url,
        signer=signer,
        on_message=on_message,
        on_error=on_error,
        subscribe_message_factory=subscribe_msg,
    )
    print("Starting broker WebSocket listener...")
    client.run_forever()


if __name__ == "__main__":
    main()
