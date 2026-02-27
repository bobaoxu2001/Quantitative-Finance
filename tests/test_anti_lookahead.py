from __future__ import annotations

import pandas as pd

from hourly_trading_system.data.contracts import enforce_information_set, validate_time_contract


def test_information_set_filters_future_data() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "event_time": "2025-01-01 10:00:00+00:00",
                "public_release_time": "2025-01-01 10:00:00+00:00",
                "ingest_time": "2025-01-01 10:01:00+00:00",
                "known_time": "2025-01-01 10:02:00+00:00",
                "value": 1,
            },
            {
                "symbol": "AAPL",
                "event_time": "2025-01-01 11:00:00+00:00",
                "public_release_time": "2025-01-01 11:00:00+00:00",
                "ingest_time": "2025-01-01 11:01:00+00:00",
                "known_time": "2025-01-01 11:02:00+00:00",
                "value": 2,
            },
        ]
    )
    validate_time_contract(frame, "sample")
    out = enforce_information_set(frame, decision_time=pd.Timestamp("2025-01-01 11:00:00+00:00"))
    assert len(out) == 1
    assert int(out.iloc[0]["value"]) == 1
