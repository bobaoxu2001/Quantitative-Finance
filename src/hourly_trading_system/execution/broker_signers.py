"""Broker-specific signing and nonce templates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Any
import urllib.parse

from hourly_trading_system.live.contracts import now_utc


def _ts_ms() -> int:
    return int(now_utc().timestamp() * 1000)


@dataclass(slots=True)
class BinanceSpotSigner:
    """Binance-style query signing with timestamp nonce."""

    api_key: str
    api_secret: str
    recv_window_ms: int = 5000

    def sign_params(
        self,
        params: dict[str, Any],
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        out = {k: v for k, v in params.items() if v is not None}
        out["timestamp"] = timestamp_ms if timestamp_ms is not None else _ts_ms()
        out["recvWindow"] = self.recv_window_ms
        query = urllib.parse.urlencode(sorted(out.items()))
        sig = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        out["signature"] = sig
        return out

    def headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}


@dataclass(slots=True)
class AlpacaSigner:
    """Alpaca-style API key headers (no payload signature)."""

    api_key: str
    api_secret: str

    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def nonce(self) -> str:
        return str(_ts_ms())


@dataclass(slots=True)
class IBKRGatewaySigner:
    """IBKR gateway request nonce/auth header template."""

    session_token: str

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.session_token}"}

    def nonce(self) -> str:
        return str(_ts_ms())
