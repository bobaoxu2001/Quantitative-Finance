"""Tests for Route C adaptive ensemble model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hourly_trading_system.features import required_features
from hourly_trading_system.models import (
    AdaptiveEnsembleModel,
    DeepCrossSectionalModel,
    EnsembleConfig,
    FactorLightGBMBaseline,
)


def _make_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(99)
    n = 400
    feature_cols = required_features()
    symbols = rng.choice(["AAPL", "MSFT", "GOOGL", "AMZN"], size=n)
    times = pd.date_range("2025-06-01", periods=n, freq="h", tz="UTC")
    data = {"symbol": symbols, "event_time": times}
    for col in feature_cols:
        data[col] = rng.normal(0, 1, n)
    features = pd.DataFrame(data)
    target_ret = pd.Series(rng.normal(0, 0.01, n), index=features.index)
    target_down = pd.Series((rng.uniform(size=n) > 0.7).astype(float), index=features.index)
    return features, target_ret, target_down


def test_ensemble_fit_and_predict() -> None:
    features, target_ret, target_down = _make_training_data()
    feature_cols = required_features()

    model_a = DeepCrossSectionalModel(feature_columns=feature_cols)
    model_b = FactorLightGBMBaseline(feature_columns=feature_cols)

    ensemble = AdaptiveEnsembleModel(
        sub_models=[model_a, model_b],
        feature_columns=feature_cols,
        config=EnsembleConfig(lookback_window=20, recalibrate_every=5),
    )
    ensemble.fit(features, target_ret, target_down)

    # Meta-learner weights should be computed and sum to ~1.
    assert ensemble._meta_return_weights is not None
    assert abs(ensemble._meta_return_weights.sum() - 1.0) < 1e-6
    assert ensemble._meta_down_weights is not None
    assert abs(ensemble._meta_down_weights.sum() - 1.0) < 1e-6

    # Predict should produce valid output.
    pred_features = features.iloc[:50].copy()
    pred = ensemble.predict(pred_features)
    frame = pred.to_frame()
    assert len(frame) == 50
    assert "expected_excess_return" in frame.columns
    assert "downside_probability" in frame.columns
    assert pred.model_name == "route_c_adaptive_ensemble"
    assert pred.metadata["route"] == "C_ensemble"
    assert "route_a_deep_cross_sectional" in pred.metadata["sub_models"]
    assert "route_b_factor_lgbm" in pred.metadata["sub_models"]


def test_ensemble_register_realized_return() -> None:
    features, target_ret, target_down = _make_training_data()
    feature_cols = required_features()

    model_a = DeepCrossSectionalModel(feature_columns=feature_cols)
    model_b = FactorLightGBMBaseline(feature_columns=feature_cols)

    ensemble = AdaptiveEnsembleModel(
        sub_models=[model_a, model_b],
        feature_columns=feature_cols,
    )
    ensemble.fit(features, target_ret, target_down)

    rng = np.random.default_rng(42)
    realized = rng.normal(0, 0.01, 10)
    preds_a = rng.normal(0, 0.01, 10)
    preds_b = rng.normal(0, 0.01, 10)
    ensemble.register_realized_return([preds_a, preds_b], realized)

    assert len(ensemble._return_error_history[0]) == 1
    assert len(ensemble._return_error_history[1]) == 1


def test_ensemble_recalibrate_weights() -> None:
    features, target_ret, target_down = _make_training_data()
    feature_cols = required_features()

    model_a = DeepCrossSectionalModel(feature_columns=feature_cols)
    model_b = FactorLightGBMBaseline(feature_columns=feature_cols)

    ensemble = AdaptiveEnsembleModel(
        sub_models=[model_a, model_b],
        feature_columns=feature_cols,
        config=EnsembleConfig(lookback_window=20),
    )
    ensemble.fit(features, target_ret, target_down)

    # Without enough history, recalibration returns current weights unchanged.
    current = ensemble._meta_return_weights
    assert current is not None
    result = ensemble._recalibrate_weights(current)
    np.testing.assert_array_almost_equal(result, current)

    # With enough history, recalibration should adjust weights.
    rng = np.random.default_rng(7)
    for _ in range(15):
        realized = rng.normal(0, 0.01, 5)
        preds_a = realized + rng.normal(0, 0.002, 5)  # model A: lower error
        preds_b = realized + rng.normal(0, 0.005, 5)  # model B: higher error
        ensemble.register_realized_return([preds_a, preds_b], realized)

    recalibrated = ensemble._recalibrate_weights(current)
    assert abs(recalibrated.sum() - 1.0) < 1e-6
