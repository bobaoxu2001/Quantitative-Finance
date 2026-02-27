"""Template: broker WebSocket listener forwarding order updates into live queue."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
import json

from hourly_trading_system.execution import (
    BinanceExecutionReplayClient,
    BinanceSpotSigner,
    BrokerWebSocketClient,
    HMACRequestSigner,
    parse_binance_execution_report,
    parse_ws_order_status_message,
)
from hourly_trading_system.live import (
    FileBackedRealtimeQueue,
    QueueMessage,
    RedisDedupStore,
    SQLiteEventStore,
    stable_event_key,
)


def main() -> None:
    ws_url = os.environ.get("BROKER_WS_URL", "")
    api_key = os.environ.get("BROKER_API_KEY", "")
    api_secret = os.environ.get("BROKER_API_SECRET", "")
    passphrase = os.environ.get("BROKER_API_PASSPHRASE")
    queue_path = os.environ.get("LIVE_QUEUE_PATH", "outputs/live_queue")
    sequence_field = os.environ.get("BROKER_WS_SEQUENCE_FIELD", "sequence")
    sequence_state_path = os.environ.get("BROKER_WS_SEQUENCE_STATE_PATH", "outputs/ws_sequence.state")
    replay_url = os.environ.get("BROKER_REPLAY_URL")
    dedup_backend = os.environ.get("LIVE_DEDUP_BACKEND", "sqlite").lower()
    dedup_sqlite_path = os.environ.get("LIVE_DEDUP_SQLITE_PATH", "outputs/live_events.db")
    dedup_redis_url = os.environ.get("LIVE_DEDUP_REDIS_URL")
    broker_kind = os.environ.get("BROKER_KIND", "binance_spot").lower()
    binance_rest_url = os.environ.get("BINANCE_REST_URL", "https://api.binance.com")
    replay_symbol = os.environ.get("BROKER_REPLAY_SYMBOL")
    if not ws_url or not api_key or not api_secret:
        raise RuntimeError("Set BROKER_WS_URL, BROKER_API_KEY, BROKER_API_SECRET")

    signer = HMACRequestSigner(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
    queue = FileBackedRealtimeQueue(queue_path)
    sqlite_store = SQLiteEventStore(dedup_sqlite_path) if dedup_backend == "sqlite" else None
    redis_store = RedisDedupStore(dedup_redis_url) if dedup_backend == "redis" and dedup_redis_url else None

    binance_replay = None
    if broker_kind == "binance_spot":
        binance_replay = BinanceExecutionReplayClient(
            rest_base_url=binance_rest_url,
            signer=BinanceSpotSigner(api_key=api_key, api_secret=api_secret),
            timeout_seconds=8,
        )

    def _record_dedup(topic: str, payload: dict) -> bool:
        event_key = stable_event_key(payload)
        if sqlite_store is not None:
            return sqlite_store.record_event(topic=topic, event_key=event_key, payload=payload)
        if redis_store is not None:
            return redis_store.record_event(topic=topic, event_key=event_key)
        return True

    def on_message(message: dict) -> None:
        if not _record_dedup("ws_raw", message):
            return
        parsed = parse_ws_order_status_message(message)
        queue.publish(
            QueueMessage(
                topic="order_updates",
                payload={"source": "ws", "raw": message, "parsed": parsed.to_dict() if parsed else None},
            )
        )
        if parsed and parsed.last_fill is not None:
            if not _record_dedup("fills", parsed.last_fill.to_dict()):
                return
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
        # Prefer official Binance replay when configured for symbol windows.
        if binance_replay is not None and replay_symbol:
            start_ms = max(int(last_sequence), 0)
            end_ms = max(int(current_sequence), start_ms + 1)
            events = [
                {
                    "e": "executionReport",
                    "s": evt.symbol,
                    "S": evt.side,
                    "X": evt.status,
                    "i": evt.order_id,
                    "q": evt.requested_quantity,
                    "z": evt.filled_quantity,
                    "E": int(evt.broker_timestamp.timestamp() * 1000),
                    "c": evt.client_order_id,
                    "r": evt.reason,
                    "x": evt.metadata.get("execution_type", "TRADE"),
                    "l": evt.last_fill.quantity if evt.last_fill else 0.0,
                    "L": evt.last_fill.fill_price if evt.last_fill else 0.0,
                    "n": evt.last_fill.fees if evt.last_fill else 0.0,
                    "N": evt.last_fill.metadata.get("fee_asset") if evt.last_fill else None,
                    "t": evt.last_fill.execution_id if evt.last_fill else None,
                    "T": int(evt.last_fill.fill_timestamp.timestamp() * 1000) if evt.last_fill else None,
                }
                for evt in binance_replay.replay_symbol_window(
                    symbol=replay_symbol,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    limit=1000,
                )
            ]
            queue.publish(
                QueueMessage(
                    topic="order_updates",
                    payload={
                        "source": "binance_rest_replay",
                        "symbol": replay_symbol,
                        "from_sequence": last_sequence + 1,
                        "to_sequence": current_sequence - 1,
                        "count": len(events),
                    },
                )
            )
            return events
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
