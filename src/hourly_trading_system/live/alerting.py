"""Alert routing layer for production monitoring and incidents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json
import urllib.request

from .contracts import AlertEvent, AlertSeverity


class AlertSink(ABC):
    """Abstract sink for alert events."""

    @abstractmethod
    def send(self, event: AlertEvent) -> None:
        """Deliver one alert event."""


class ConsoleAlertSink(AlertSink):
    """Simple sink that prints alerts to stdout."""

    def send(self, event: AlertEvent) -> None:
        payload = event.to_dict()
        print(
            f"[{payload['timestamp']}] [{payload['severity'].upper()}] "
            f"{payload['source']}: {payload['message']} details={payload['details']}"
        )


class FileAlertSink(AlertSink):
    """Persist alerts as JSONL for audit and incident review."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, event: AlertEvent) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict()) + "\n")


class WebhookAlertSink(AlertSink):
    """Forward alerts to webhook destinations (Slack/Teams/PagerDuty proxy)."""

    def __init__(self, webhook_url: str, timeout_seconds: int = 5) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds

    def send(self, event: AlertEvent) -> None:
        payload = json.dumps(event.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            url=self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds):
            return


@dataclass(slots=True)
class AlertRouter:
    """
    Routes alert events by severity to configured sinks.

    default_sinks are always used; severity_sinks are additive.
    """

    default_sinks: list[AlertSink] = field(default_factory=list)
    severity_sinks: dict[AlertSeverity, list[AlertSink]] = field(default_factory=dict)

    def route(self, event: AlertEvent) -> None:
        sinks: list[AlertSink] = list(self.default_sinks)
        sinks.extend(self.severity_sinks.get(event.severity, []))
        for sink in sinks:
            sink.send(event)

    def info(self, source: str, message: str, details: dict | None = None) -> None:
        self.route(
            AlertEvent(
                severity=AlertSeverity.INFO,
                source=source,
                message=message,
                details=details or {},
            )
        )

    def warning(self, source: str, message: str, details: dict | None = None) -> None:
        self.route(
            AlertEvent(
                severity=AlertSeverity.WARNING,
                source=source,
                message=message,
                details=details or {},
            )
        )

    def critical(self, source: str, message: str, details: dict | None = None) -> None:
        self.route(
            AlertEvent(
                severity=AlertSeverity.CRITICAL,
                source=source,
                message=message,
                details=details or {},
            )
        )

    @staticmethod
    def with_console_and_file(file_path: str | Path) -> "AlertRouter":
        sink_file = FileAlertSink(file_path)
        sink_console = ConsoleAlertSink()
        return AlertRouter(default_sinks=[sink_console, sink_file])


def broadcast(
    router: AlertRouter,
    severity: AlertSeverity,
    source: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Helper for alert broadcasting from procedural code."""
    event = AlertEvent(
        severity=severity,
        source=source,
        message=message,
        details=details or {},
    )
    router.route(event)
