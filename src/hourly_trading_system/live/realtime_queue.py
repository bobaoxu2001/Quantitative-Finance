"""Real-time queue abstraction for live orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable
import json

from .contracts import QueueMessage


class BaseRealtimeQueue(ABC):
    """Abstract message queue used by live trading services."""

    @abstractmethod
    def publish(self, message: QueueMessage) -> None:
        """Publish one message."""

    @abstractmethod
    def consume(self, topic: str, max_messages: int = 1) -> list[QueueMessage]:
        """Consume up to `max_messages` from topic."""

    @abstractmethod
    def size(self, topic: str) -> int:
        """Return queue depth for topic."""


class InMemoryRealtimeQueue(BaseRealtimeQueue):
    """Low-latency queue backend suitable for tests and single-process live runs."""

    def __init__(self) -> None:
        self._topics: dict[str, deque[QueueMessage]] = defaultdict(deque)

    def publish(self, message: QueueMessage) -> None:
        self._topics[message.topic].append(message)

    def consume(self, topic: str, max_messages: int = 1) -> list[QueueMessage]:
        out: list[QueueMessage] = []
        queue = self._topics[topic]
        n = max(0, int(max_messages))
        for _ in range(n):
            if not queue:
                break
            out.append(queue.popleft())
        return out

    def size(self, topic: str) -> int:
        return len(self._topics[topic])


class FileBackedRealtimeQueue(BaseRealtimeQueue):
    """
    Durable local queue backend using JSONL append logs.

    This backend is designed for reliability in cloud agents and staging
    environments where external queue services are unavailable.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._read_offsets: dict[str, int] = defaultdict(int)

    def _topic_file(self, topic: str) -> Path:
        return self.base_dir / f"{topic}.jsonl"

    def publish(self, message: QueueMessage) -> None:
        path = self._topic_file(message.topic)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message.to_dict(), default=str) + "\n")

    def _read_lines(self, topic: str) -> list[str]:
        path = self._topic_file(topic)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return handle.readlines()

    def consume(self, topic: str, max_messages: int = 1) -> list[QueueMessage]:
        lines = self._read_lines(topic)
        if not lines:
            return []
        start = self._read_offsets[topic]
        end = min(start + max(0, int(max_messages)), len(lines))
        messages = [
            QueueMessage.from_dict(json.loads(raw))
            for raw in lines[start:end]
        ]
        self._read_offsets[topic] = end
        return messages

    def size(self, topic: str) -> int:
        total = len(self._read_lines(topic))
        read = self._read_offsets[topic]
        return max(total - read, 0)


def drain_topic(queue: BaseRealtimeQueue, topic: str, batch_size: int = 100) -> Iterable[QueueMessage]:
    """Yield all currently available messages from a topic."""
    while queue.size(topic) > 0:
        batch = queue.consume(topic=topic, max_messages=batch_size)
        if not batch:
            break
        for msg in batch:
            yield msg
