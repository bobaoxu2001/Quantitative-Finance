"""Regression tests for live trading, broker and governance fixes."""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse

import pandas as pd
import pytest

from hourly_trading_system.analytics import LiveTCAAttributor
from hourly_trading_system.config import SystemConfig
from hourly_trading_system.execution import (
    BinanceExecutionReplayClient,
    BinanceSpotSigner,
    BrokerWebSocketClient,
    HMACRequestSigner,
    parse_binance_execution_report,
)
from hourly_trading_system.governance import ArtifactIntegrityError, ModelRegistry
from hourly_trading_system.io_utils import atomic_write_json
from hourly_trading_system.live import FillEvent, LiveControlPlane, OrderAck, OrderRequest
from hourly_trading_system.orchestration.live_loop import HourlyLiveLoop, LiveDataProvider, LiveLoopConfig
from hourly_trading_system.orchestration.live_runner import LiveTradingRunner
from hourly_trading_system.portfolio import HourlyPortfolioAllocator
from hourly_trading_system.risk import RiskController


def test_tca_measures_shortfall_against_reference_not_protective_limit() -> None:
    tca = LiveTCAAttributor()
    req = OrderRequest(
        symbol="MSFT",
        side="BUY",
        quantity=10.0,
        limit_price=100.30,  # +30 bps protection band
        client_order_id="cid-1",
        metadata={"reference_price": 100.0},
    )
    ack = OrderAck(order_id="o-1", accepted=True, status="accepted", client_order_id="cid-1")
    tca.register_submissions([req], [ack])
    row = tca.record_fill(FillEvent(order_id="o-1", symbol="MSFT", side="BUY", quantity=10.0, fill_price=100.10))
    assert row is not None
    assert row.arrival_price == 100.0
    assert row.shortfall_bps == pytest.approx(10.0)


def test_binance_average_price_handles_fractional_quantities() -> None:
    event = parse_binance_execution_report(
        {
            "e": "executionReport",
            "s": "BTCUSDT",
            "i": 1,
            "S": "BUY",
            "q": "0.10",
            "z": "0.05",
            "Z": "3000.0",  # cumulative quote: 0.05 BTC * 60,000
            "l": "0.05",
            "L": "60000",
            "X": "PARTIALLY_FILLED",
        }
    )
    assert event is not None
    assert event.average_fill_price == pytest.approx(60_000.0)


def test_binance_signature_covers_the_query_string_actually_sent() -> None:
    signer = BinanceSpotSigner(api_key="k", api_secret="secret")
    signed = signer.sign_params({"symbol": "BTCUSDT", "startTime": 2, "limit": 10}, timestamp_ms=1_700_000_000_000)
    sent = urllib.parse.urlencode(list(signed.items()))
    payload, _, signature = sent.rpartition("&signature=")
    expected = hmac.new(b"secret", payload.encode(), hashlib.sha256).hexdigest()
    assert signature == expected


def test_binance_replay_emits_every_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    client = BinanceExecutionReplayClient(rest_base_url="https://example", signer=BinanceSpotSigner("k", "s"))
    orders = [{"orderId": 7, "side": "BUY", "origQty": "1.0", "executedQty": "1.0", "status": "FILLED",
               "cummulativeQuoteQty": "100.5", "clientOrderId": "c7", "updateTime": 3}]
    trades = [
        {"orderId": 7, "id": 1, "qty": "0.4", "price": "100", "commission": "0", "time": 1},
        {"orderId": 7, "id": 2, "qty": "0.6", "price": "100.8333", "commission": "0", "time": 2},
    ]
    monkeypatch.setattr(
        BinanceExecutionReplayClient,
        "_signed_get",
        lambda self, path, params: orders if "allOrders" in path else trades,
    )
    events = client.replay_symbol_window("BTCUSDT", 0, 10)
    assert [e.last_fill.execution_id for e in events if e.last_fill] == ["1", "2"]
    assert sum(e.last_fill.quantity for e in events if e.last_fill) == pytest.approx(1.0)
    assert events[0].average_fill_price == pytest.approx(100.5)


def test_websocket_skips_stale_sequences_and_checkpoints_after_delivery(tmp_path) -> None:
    delivered: list[int] = []
    client = BrokerWebSocketClient(
        ws_url="wss://example",
        signer=HMACRequestSigner("k", "s"),
        on_message=lambda m: delivered.append(m["sequence"]),
        sequence_field="sequence",
        sequence_state_path=tmp_path / "seq",
    )
    for seq in (1, 2, 2, 1, 3):
        msg = {"sequence": seq}
        if client._accept_sequence(msg):
            client.on_message(msg)
            client._commit_sequence(msg)
    assert delivered == [1, 2, 3]
    assert (tmp_path / "seq").read_text() == "3"
    assert client._handle_sequence({"sequence": 3}) is False


def test_unlock_requires_independent_approvers(tmp_path) -> None:
    control = LiveControlPlane(tmp_path / "ctl.json", required_unlock_approvals=1)
    control.force_kill_switch(actor="ops", reason="incident")
    control.request_unlock(requestor="alice", reason="fixed")
    with pytest.raises(ValueError, match="own unlock"):
        control.approve_unlock(approver="alice")
    control.approve_unlock(approver="bob")
    control.finalize_unlock(actor="carol")
    with pytest.raises(ValueError, match="already finalized"):
        control.approve_unlock(approver="dave")


def test_registry_rejects_tampered_artifacts(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    record = registry.register_model({"weights": [1, 2, 3]}, "m", "B", ["f"], {"sharpe": 2.0, "calmar": 1.0})
    assert record.artifact_sha256
    assert registry.load_model(record.version_id) == {"weights": [1, 2, 3]}
    with open(record.artifact_path, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        registry.load_model(record.version_id)


def test_atomic_write_leaves_no_temp_files(tmp_path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2})
    assert target.read_text().strip().endswith("}")
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


class _Provider(LiveDataProvider):
    def get_features(self, decision_time):
        return pd.DataFrame()

    def get_market_snapshot(self, decision_time):
        return pd.DataFrame()


@pytest.mark.parametrize(
    ("utc_time", "expected"),
    [
        ("2026-01-15 14:00", False),  # 09:00 EST -> before 09:30 open
        ("2026-01-15 15:00", True),   # 10:00 EST
        ("2026-01-15 21:00", True),   # 16:00 EST close (inclusive)
        ("2026-07-15 20:00", True),   # 16:00 EDT close
        ("2026-07-15 21:00", False),  # 17:00 EDT -> after close
        ("2026-07-18 15:00", False),  # Saturday
    ],
)
def test_live_loop_session_is_dst_aware(utc_time: str, expected: bool) -> None:
    loop = HourlyLiveLoop(runner=None, provider=_Provider(), config=LiveLoopConfig())  # type: ignore[arg-type]
    assert loop._should_run(pd.Timestamp(utc_time, tz="UTC")) is expected


class _NoopRegistry:
    pass


def test_live_runner_values_positions_missing_from_snapshot(tmp_path) -> None:
    config = SystemConfig(initial_capital=1_000.0)
    runner = LiveTradingRunner(
        config=config,
        registry=_NoopRegistry(),  # type: ignore[arg-type]
        allocator=HourlyPortfolioAllocator(config=config.portfolio),
        risk_controller=RiskController(),
        strategy_id="t",
    )
    runner.current_positions = {"AAA": 10.0, "HALTED": 5.0}
    runner.cash = 0.0
    runner.last_prices = {"HALTED": 100.0}
    weights = runner._current_weights(pd.DataFrame({"symbol": ["AAA"], "close": [100.0]}), equity=1_500.0)
    assert weights["HALTED"] == pytest.approx(500.0 / 1_500.0)
