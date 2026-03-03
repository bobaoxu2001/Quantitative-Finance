"""Tests for feature engineering library, including new volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hourly_trading_system.features.library import build_feature_panel, required_features


def _make_minimal_market() -> pd.DataFrame:
    rng = np.random.default_rng(55)
    symbols = ["SPY", "AAPL", "MSFT"]
    sectors = {"SPY": "ETF", "AAPL": "TECH", "MSFT": "TECH"}
    times = pd.date_range("2025-06-01", periods=80, freq="h", tz="UTC")
    times = times[(times.hour >= 14) & (times.hour <= 20) & (times.dayofweek < 5)]
    rows = []
    for sym in symbols:
        p = 400.0 if sym == "SPY" else float(rng.uniform(100, 300))
        for t in times:
            p *= np.exp(rng.normal(0.0001, 0.005))
            open_ = p * (1 + rng.normal(0, 0.001))
            close = p
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.002)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.002)))
            volume = float(rng.integers(500_000, 2_000_000))
            rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "sector": sectors[sym],
                    "shares_outstanding": 1_000_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_new_volatility_features_present() -> None:
    market = _make_minimal_market()
    panel = build_feature_panel(market)

    new_features = ["garman_klass_vol_20h", "parkinson_vol_20h", "kyle_lambda_20h"]
    for feat in new_features:
        assert feat in panel.columns, f"Missing feature: {feat}"
        # At least some values should be non-NaN (after warm-up period).
        non_null = panel[feat].dropna()
        assert len(non_null) > 0, f"Feature {feat} is all NaN"


def test_required_features_includes_new_features() -> None:
    features = required_features()
    assert "garman_klass_vol_20h" in features
    assert "parkinson_vol_20h" in features
    assert "kyle_lambda_20h" in features


def test_feature_panel_shape_consistency() -> None:
    market = _make_minimal_market()
    panel = build_feature_panel(market)
    features = required_features()

    # All required features should be present as columns.
    for feat in features:
        assert feat in panel.columns, f"Missing required feature: {feat}"

    # Panel should have standard metadata columns plus all features.
    expected_cols = {"symbol", "event_time", "sector", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(panel.columns))
