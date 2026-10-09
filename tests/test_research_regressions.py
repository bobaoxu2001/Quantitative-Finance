"""Regression tests for research-layer fixes (labels, features, models, metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hourly_trading_system.analytics import compute_performance_report
from hourly_trading_system.analytics.metrics import closed_trades_from_fills, infer_periods_per_year
from hourly_trading_system.features import build_feature_panel, required_features
from hourly_trading_system.labels.engineering import build_labels, train_validation_test_split
from hourly_trading_system.models import (
    AdaptiveEnsembleModel,
    DeepCrossSectionalModel,
    FactorLightGBMBaseline,
)
from hourly_trading_system.models.base import (
    BaseSignalModel,
    FeaturePreprocessor,
    PredictionFrame,
    cross_sectional_rank_score,
)
from hourly_trading_system.time_utils import to_utc_period_end


def _market(n_hours: int = 60, symbols: tuple[str, ...] = ("SPY", "AAA", "BBB", "CCC"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-03-03 14:00", periods=n_hours, freq="h", tz="UTC")
    rows = []
    for sym in symbols:
        price = 100.0
        for t in times:
            price *= float(np.exp(rng.normal(0.0, 0.01)))
            rows.append(
                {
                    "symbol": sym,
                    "event_time": t,
                    "open": price * 0.999,
                    "high": price * 1.004,
                    "low": price * 0.995,
                    "close": price,
                    "volume": 1_000_000.0,
                    "sector": "ETF" if sym == "SPY" else "TECH",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- labels


def test_downside_label_is_undefined_when_horizon_incomplete() -> None:
    labels = build_labels(_market(n_hours=20), horizon_hours=5)
    ordered = labels.sort_values("event_time").groupby("symbol")
    # Downside needs lows of bars t+1..t+5; the return label also needs the exit bar t+6.
    assert ordered.tail(5)["target_downside_event"].isna().all()
    assert ordered.tail(6)["label_end_time"].isna().all()
    assert ordered.tail(6)["target_forward_return"].isna().all()
    head = labels.dropna(subset=["label_end_time"])
    assert set(head["target_downside_event"].unique()) <= {0.0, 1.0}


def test_label_end_time_is_exit_bar_and_split_purges_overlap() -> None:
    market = _market(n_hours=40)
    labels = build_labels(market, horizon_hours=5)
    times = sorted(market["event_time"].unique())
    first = labels.loc[(labels["symbol"] == "AAA") & (labels["event_time"] == times[0])].iloc[0]
    assert first["label_end_time"] == times[6]  # entry t+1, exit open(t+1+5)

    cut = times[20]
    splits = train_validation_test_split(
        labels,
        train_end=cut.isoformat(),
        val_start=times[21].isoformat(),
        val_end=times[30].isoformat(),
        test_start=times[31].isoformat(),
        test_end=times[-1].isoformat(),
    )
    train = splits["train"]
    assert not train.empty
    assert (pd.to_datetime(train["label_end_time"], utc=True) <= cut).all()


def test_date_only_period_end_is_inclusive() -> None:
    assert to_utc_period_end("2026-02-26") == pd.Timestamp("2026-02-26 23:59:59.999999999", tz="UTC")
    explicit = "2026-02-26T15:00:00+00:00"
    assert to_utc_period_end(explicit) == pd.Timestamp(explicit)


# ------------------------------------------------------------------------- features


def test_freshness_features_age_with_time() -> None:
    market = _market(n_hours=30, symbols=("SPY", "AAA"))
    t0 = market["event_time"].min()
    analyst = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "event_time": [t0],
            "public_release_time": [t0],
            "ingest_time": [t0 + pd.Timedelta(minutes=5)],
            "known_time": [t0 + pd.Timedelta(minutes=5)],
            "net_upgrades_downgrades": [1.0],
            "target_price_revision": [0.02],
            "hours_since_update": [0.0],
            "analyst_update_time": [t0 + pd.Timedelta(minutes=5)],
        }
    )
    sentiment = analyst.rename(columns={"analyst_update_time": "sentiment_update_time"}).assign(
        sentiment_score=0.3, sentiment_dispersion=0.1, mention_abnormal=0.0
    )
    panel = build_feature_panel(market, sentiment=sentiment, analyst=analyst)
    aaa = panel.loc[panel["symbol"] == "AAA"].sort_values("event_time").dropna(subset=["analyst_net_updown"])
    for col in ("hours_since_analyst_update", "analyst_action_recency_hours", "hours_since_sentiment_update"):
        values = aaa[col].to_numpy()
        assert np.all(np.isfinite(values)), col
        assert np.all(np.diff(values) > 0), f"{col} must increase as the record ages"


def test_cross_sectional_rank_features_match_reference() -> None:
    panel = build_feature_panel(_market(n_hours=30))
    expected = panel.groupby("event_time")["ret_1h"].transform(lambda s: s.rank(pct=True))
    np.testing.assert_allclose(panel["breadth_percentile"], expected, equal_nan=True)
    expected_rev = panel.groupby("event_time")["ret_1h"].transform(lambda s: (-s).rank(pct=True))
    np.testing.assert_allclose(panel["reversal_rank_1h"], expected_rev, equal_nan=True)


# --------------------------------------------------------------------------- models


def _training_frame(n_times: int = 80, n_symbols: int = 6, seed: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n_times, freq="h", tz="UTC")
    rows = [{"symbol": f"S{j}", "event_time": t} for t in times for j in range(n_symbols)]
    features = pd.DataFrame(rows)
    for col in required_features():
        features[col] = rng.normal(0.0, 1.0, len(features))
    signal = features[required_features()[0]].to_numpy()
    target_ret = pd.Series(0.002 * signal + rng.normal(0, 0.001, len(features)), index=features.index)
    target_down = pd.Series((rng.uniform(size=len(features)) > 0.7).astype(float), index=features.index)
    return features, target_ret, target_down


def test_preprocessor_uses_training_statistics_at_inference() -> None:
    features, _, _ = _training_frame()
    cols = required_features()
    prep = FeaturePreprocessor(cols).fit(features)
    row = features.iloc[[0]].copy()
    row[cols[0]] = np.nan
    z = prep.transform(row)
    # NaN is imputed with the *training* median, not the (undefined) batch median.
    expected = (np.nanmedian(features[cols[0]]) - prep.means_[0]) / prep.stds_[0]
    assert z[0, 0] == pytest.approx(expected)


def test_prediction_for_a_row_does_not_depend_on_batch_composition() -> None:
    features, target_ret, target_down = _training_frame()
    model = DeepCrossSectionalModel(feature_columns=required_features())
    model.fit(features, target_ret, target_down)
    batch = features.iloc[:12].copy()
    # The scored row has a missing value; it must be imputed identically whether
    # the row is scored alone or alongside other symbols.
    batch.loc[batch.index[0], required_features()[0]] = np.nan
    alone = model.predict(batch.iloc[[0]]).expected_excess_return.iloc[0]
    together = model.predict(batch).expected_excess_return.iloc[0]
    assert alone == pytest.approx(together)


class _ConstantScaleModel(BaseSignalModel):
    """Toy model emitting a fixed transform of the first feature at a chosen scale."""

    def __init__(self, name: str, scale: float, sign: float = 1.0) -> None:
        self.model_name = name
        self.scale = scale
        self.sign = sign

    def fit(self, features, target_excess_return, target_downside_event):
        return self

    def predict(self, features: pd.DataFrame) -> PredictionFrame:
        raw = self.sign * self.scale * features[required_features()[0]]
        return PredictionFrame(
            symbol=features["symbol"],
            event_time=pd.to_datetime(features["event_time"], utc=True),
            expected_excess_return=raw,
            downside_probability=pd.Series(0.2, index=features.index),
            model_name=self.model_name,
        )


def test_ensemble_blends_sub_models_on_a_common_scale() -> None:
    features, target_ret, target_down = _training_frame()
    # A huge-scale model that is anti-correlated with the signal must not dominate.
    big_wrong = _ConstantScaleModel("big_wrong", scale=1_000.0, sign=-1.0)
    small_right = _ConstantScaleModel("small_right", scale=1e-4, sign=1.0)
    ensemble = AdaptiveEnsembleModel([big_wrong, small_right], feature_columns=required_features())
    ensemble.fit(features, target_ret, target_down)
    weights = ensemble._meta_return_weights
    assert weights is not None
    assert weights[1] > weights[0]

    pred = ensemble.predict(features.iloc[-6:])
    scores = pred.expected_excess_return.to_numpy()
    assert scores.min() >= -0.5 and scores.max() <= 0.5
    signal = features.iloc[-6:][required_features()[0]].to_numpy()
    assert np.corrcoef(scores, signal)[0, 1] > 0


def test_ensemble_recalibration_is_persisted() -> None:
    features, target_ret, target_down = _training_frame()
    from hourly_trading_system.models import EnsembleConfig

    ensemble = AdaptiveEnsembleModel(
        [DeepCrossSectionalModel(required_features()), FactorLightGBMBaseline(required_features())],
        feature_columns=required_features(),
        config=EnsembleConfig(recalibrate_every=1, lookback_window=20),
    )
    ensemble.fit(features, target_ret, target_down)
    rng = np.random.default_rng(3)
    for _ in range(12):
        realized = rng.normal(0, 0.01, 5)
        ensemble.register_realized_return([realized + 0.0001, realized + 0.05], realized)
    before = ensemble._meta_return_weights.copy()
    ensemble.predict(features.iloc[:6])  # count 0 -> no recalibration
    ensemble.predict(features.iloc[:6])  # count 1 -> recalibrates and persists
    after = ensemble._meta_return_weights
    assert after[0] > before[0]
    ensemble.predict(features.iloc[:6])
    assert ensemble._meta_return_weights[0] >= after[0]


def test_cross_sectional_rank_score_is_per_timestamp() -> None:
    times = pd.Series(pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 3, utc=True))
    scores = cross_sectional_rank_score(np.array([1.0, 2.0, 3.0, 300.0, 200.0, 100.0]), times)
    np.testing.assert_allclose(scores[:3], scores[3:][::-1])
    assert scores.max() <= 0.5 and scores.min() >= -0.5


# -------------------------------------------------------------------------- metrics


def test_sortino_uses_downside_deviation_over_all_periods() -> None:
    times = pd.date_range("2026-01-05", periods=200, freq="h", tz="UTC")
    returns = np.random.default_rng(5).normal(0.0002, 0.003, len(times))
    returns[0] = 0.0
    equity = 100_000 * np.cumprod(1 + returns)
    report = compute_performance_report(
        pd.DataFrame({"event_time": times, "equity": equity}), periods_per_year=1000.0
    )
    r = report.extras["returns_series"].to_numpy()
    expected = r.mean() / np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(1000.0)
    assert report.summary["sortino"] == pytest.approx(expected)


def test_periods_per_year_is_inferred_from_session_calendar() -> None:
    idx = pd.date_range("2025-01-01", "2026-01-01", freq="h", tz="UTC")
    idx = idx[(idx.hour >= 14) & (idx.hour <= 20) & (idx.dayofweek < 5)]
    ppy = infer_periods_per_year(pd.DatetimeIndex(idx))
    assert 7 * 250 <= ppy <= 7 * 262


def test_trade_stats_are_derived_from_fills() -> None:
    t = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "event_time": t, "quantity": 10.0, "fill_price": 100.0},
            {"symbol": "AAA", "event_time": t + pd.Timedelta(hours=2), "quantity": 10.0, "fill_price": 110.0},
            {"symbol": "AAA", "event_time": t + pd.Timedelta(hours=4), "quantity": -5.0, "fill_price": 120.0},
            {"symbol": "BBB", "event_time": t, "quantity": 4.0, "fill_price": 50.0},
            {"symbol": "BBB", "event_time": t + pd.Timedelta(hours=3), "quantity": -4.0, "fill_price": 45.0},
        ]
    )
    closed = closed_trades_from_fills(fills).set_index("symbol")
    assert closed.loc["AAA", "pnl"] == pytest.approx(5 * (120.0 - 105.0))
    assert closed.loc["AAA", "holding_hours"] == pytest.approx(3.0)  # avg entry 1h after t
    assert closed.loc["BBB", "pnl"] == pytest.approx(-20.0)

    times = pd.date_range(t, periods=10, freq="h")
    report = compute_performance_report(
        pd.DataFrame({"event_time": times, "equity": np.linspace(100.0, 101.0, len(times))}), trades=fills
    )
    assert report.summary["win_rate"] == pytest.approx(0.5)
    assert report.summary["profit_factor"] == pytest.approx(75.0 / 20.0)
    assert report.extras["closed_trade_count"] == 2.0
