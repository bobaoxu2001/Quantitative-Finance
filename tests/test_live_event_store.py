from __future__ import annotations

from hourly_trading_system.live import SQLiteEventStore, stable_event_key


def test_sqlite_event_store_exactly_once(tmp_path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    payload = {"sequence": 1001, "event": "executionReport", "symbol": "AAPL"}
    key = stable_event_key(payload)
    inserted_1 = store.record_event(topic="ws_raw", event_key=key, payload=payload)
    inserted_2 = store.record_event(topic="ws_raw", event_key=key, payload=payload)
    assert inserted_1
    assert not inserted_2
    rows = store.fetch_since(topic="ws_raw", last_id=0, limit=10)
    assert len(rows) == 1
    assert rows[0]["payload"]["symbol"] == "AAPL"
