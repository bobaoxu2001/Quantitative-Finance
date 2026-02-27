from __future__ import annotations

from hourly_trading_system.execution import HMACRequestSigner, IdempotencyStore, PaperOMSClient
from hourly_trading_system.live import FillReconciler, OrderRequest
from hourly_trading_system.risk import LiveSafetyGuard


def test_hmac_signer_and_idempotency_store() -> None:
    signer = HMACRequestSigner(api_key="k", api_secret="s")
    sig1 = signer.sign(method="POST", path="/orders", timestamp="1700000000000", body='{"x":1}')
    sig2 = signer.sign(method="POST", path="/orders", timestamp="1700000000000", body='{"x":1}')
    assert sig1 == sig2

    store = IdempotencyStore()
    key = "idem-abc"
    assert not store.exists(key)
    store.mark(key)
    assert store.exists(key)


def test_paper_oms_updates_and_fill_reconciliation() -> None:
    oms = PaperOMSClient()
    order = OrderRequest(
        symbol="AAPL",
        side="BUY",
        quantity=10.0,
        order_type="LMT",
        limit_price=100.0,
        client_order_id="cid-1",
        metadata={"reference_price": 99.5},
    )
    batch = oms.submit_orders([order])
    assert batch.accepted_count() == 1
    updates = oms.poll_order_updates(strategy_id="hourly_system", max_events=10)
    assert len(updates) == 1
    assert updates[0].status == "filled"

    reconciler = FillReconciler(cash=1_000.0)
    reconciler.register_submissions([order], batch.acks)
    reconciler.apply_status(updates[0])
    assert reconciler.positions["AAPL"] == 10.0
    assert reconciler.cash == 0.0


def test_live_safety_guard_limit_protection_and_breakers() -> None:
    guard = LiveSafetyGuard()
    reference_prices = {"AAPL": 200.0}
    orders = [
        OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=5.0,
            order_type="MKT",
            client_order_id="cid-1",
        )
    ]
    decision = guard.pre_trade_check(orders=orders, reference_prices=reference_prices)
    assert decision.allowed
    assert decision.adjusted_orders[0].order_type == "LMT"
    assert decision.adjusted_orders[0].limit_price is not None

    guard.register_batch_outcome(accepted_count=0, rejected_count=1)
    guard.register_batch_outcome(accepted_count=0, rejected_count=1)
    guard.register_batch_outcome(accepted_count=0, rejected_count=1)
    guard.register_batch_outcome(accepted_count=0, rejected_count=1)
    guard.register_batch_outcome(accepted_count=0, rejected_count=1)
    assert guard.kill_switch_engaged
