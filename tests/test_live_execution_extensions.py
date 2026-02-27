from __future__ import annotations

from datetime import datetime, timezone

from hourly_trading_system.analytics import LiveTCAAttributor
from hourly_trading_system.execution import (
    BrokerConnectionConfig,
    BrokerKind,
    BrokerWebSocketClient,
    HMACRequestSigner,
    build_broker_oms_adapter,
    parse_ws_order_status_message,
)
from hourly_trading_system.live import FillEvent, OrderAck, OrderRequest


def test_broker_adapter_factory_and_ws_message_parser() -> None:
    adapter = build_broker_oms_adapter(
        BrokerConnectionConfig(
            broker=BrokerKind.ALPACA,
            submit_url="https://example.com/orders",
            cancel_url="https://example.com/cancel",
        )
    )
    assert adapter.__class__.__name__ == "AlpacaOMSAdapter"

    msg = {
        "event": "execution_report",
        "data": {
            "order_id": "oid-1",
            "symbol": "AAPL",
            "side": "BUY",
            "status": "filled",
            "orig_qty": 10,
            "filled_qty": 10,
            "avg_fill_price": 101.2,
            "execution_id": "ex-1",
            "fees": 0.5,
        },
    }
    parsed = parse_ws_order_status_message(msg)
    assert parsed is not None
    assert parsed.order_id == "oid-1"
    assert parsed.last_fill is not None
    assert parsed.last_fill.fill_price == 101.2


def test_live_tca_attributor_and_ws_sequence_checkpoint(tmp_path) -> None:
    tca = LiveTCAAttributor()
    req = OrderRequest(
        symbol="MSFT",
        side="BUY",
        quantity=5.0,
        client_order_id="cid-1",
        metadata={"reference_price": 100.0, "decision_time": "2026-01-01T10:00:00+00:00"},
    )
    ack = OrderAck(
        order_id="order-1",
        accepted=True,
        status="accepted",
        client_order_id="cid-1",
        broker_timestamp=datetime.now(timezone.utc),
    )
    tca.register_submissions([req], [ack])
    row = tca.record_fill(
        FillEvent(
            order_id="order-1",
            symbol="MSFT",
            side="BUY",
            quantity=5.0,
            fill_price=100.2,
            fees=0.1,
            execution_id="fill-1",
        )
    )
    assert row is not None
    summary = tca.summary()
    assert summary["trade_count"] == 1.0
    assert summary["avg_total_cost_bps"] > 0.0

    seq_file = tmp_path / "ws.seq"
    signer = HMACRequestSigner(api_key="k", api_secret="s")
    client = BrokerWebSocketClient(
        ws_url="wss://example.com/ws",
        signer=signer,
        on_message=lambda _: None,
        sequence_field="sequence",
        sequence_state_path=seq_file,
    )
    client._handle_sequence({"sequence": 12})
    client._handle_sequence({"sequence": 13})
    assert client.last_sequence == 13
    assert seq_file.read_text(encoding="utf-8").strip() == "13"
