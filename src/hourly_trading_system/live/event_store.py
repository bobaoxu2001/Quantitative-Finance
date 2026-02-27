"""Durable exactly-once event storage and deduplication primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import sqlite3


def stable_event_key(payload: dict[str, Any], preferred_fields: list[str] | None = None) -> str:
    """Derive deterministic event key from canonical payload fields."""
    fields = preferred_fields or ["id", "sequence", "event_id", "execution_id", "order_id", "client_order_id"]
    for field in fields:
        value = payload.get(field)
        if value not in (None, "", []):
            return f"{field}:{value}"
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"hash:{hashlib.sha256(raw).hexdigest()}"


@dataclass(slots=True)
class SQLiteEventStore:
    """
    SQLite-backed durable event store with exactly-once semantics.

    Uses INSERT OR IGNORE on `(topic, event_key)` unique index to dedupe.
    """

    db_path: str | Path

    def __post_init__(self) -> None:
        self.db_path = str(self.db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(topic, event_key)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_topic_id
                ON events(topic, id);
                """
            )

    def record_event(self, topic: str, event_key: str, payload: dict[str, Any]) -> bool:
        """Return True if event inserted (new), False if duplicate."""
        payload_json = json.dumps(payload, default=str, sort_keys=True)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO events(topic, event_key, payload_json) VALUES (?, ?, ?)",
                (topic, event_key, payload_json),
            )
            return cursor.rowcount == 1

    def fetch_since(self, topic: str, last_id: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, topic, event_key, payload_json, received_at
                FROM events
                WHERE topic = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (topic, last_id, limit),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "id": row[0],
                    "topic": row[1],
                    "event_key": row[2],
                    "payload": json.loads(row[3]),
                    "received_at": row[4],
                }
            )
        return out


@dataclass(slots=True)
class RedisDedupStore:
    """
    Optional Redis dedup template (requires redis package and server).
    """

    redis_url: str
    namespace: str = "hourly-live"
    ttl_seconds: int = 7 * 24 * 3600

    def _client(self):
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("redis package not installed; install optional live deps.") from exc
        return redis.Redis.from_url(self.redis_url, decode_responses=True)

    def record_event(self, topic: str, event_key: str) -> bool:
        client = self._client()
        key = f"{self.namespace}:{topic}:{event_key}"
        # SETNX + EXPIRE for exactly-once window.
        inserted = client.setnx(key, "1")
        if inserted:
            client.expire(key, self.ttl_seconds)
            return True
        return False
