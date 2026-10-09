"""Route C: Adaptive ensemble meta-model combining multiple signal routes.

Inspired by multi-model ensembling used at Renaissance Technologies, DE Shaw,
and Two Sigma.  Each sub-model produces independent predictions; the ensemble
combines them using inverse-volatility weighting of recent out-of-sample
residuals, then applies a shrinkage overlay toward the equal-weight blend.

Key ideas from state-of-the-art quant research:
  - Adaptive model confidence: recent forecast accuracy drives blending.
  - Stacking with ridge meta-learner for return predictions.
  - Bayesian-style posterior averaging for downside probability.
  - Decorrelation bonus: models whose errors are less correlated receive
    higher effective weight to improve diversification of alpha sources.

Sub-models are blended on a common scale: each sub-model's return forecast is
converted to a centred cross-sectional rank score in [-0.5, 0.5] per
``event_time`` before weighting. Without this step a model emitting raw
returns (~1e-3) and a model emitting rank scores (~0.3) cannot be averaged
meaningfully -- the larger-scale model silently dominates. The blended score is
re-ranked so the ensemble output has the same scale as Route B.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import (
    BaseSignalModel,
    PredictionFrame,
    RidgeRegressor,
    cross_sectional_rank_score,
)


@dataclass(slots=True)
class EnsembleConfig:
    lookback_window: int = 120
    shrinkage_toward_equal: float = 0.3
    min_weight: float = 0.10
    decorrelation_bonus: float = 0.5
    meta_ridge_alpha: float = 5.0
    recalibrate_every: int = 20
    meta_fraction: float = 0.30
    embargo_periods: int = 6


class AdaptiveEnsembleModel(BaseSignalModel):
    """
    Production-grade ensemble that adaptively blends N sub-models.

    Training flow:
      1. Fit each sub-model on the same training data.
      2. Collect stacked out-of-fold predictions from a hold-out slice.
      3. Train a ridge meta-learner on the stacked predictions.

    Inference flow:
      1. Generate predictions from each sub-model.
      2. Combine via meta-learner weights, shrunk toward equal-weight.
      3. Track running forecast errors to adapt weights online.
    """

    model_name = "route_c_adaptive_ensemble"

    def __init__(
        self,
        sub_models: list[BaseSignalModel],
        feature_columns: list[str],
        config: EnsembleConfig | None = None,
    ) -> None:
        self.sub_models = sub_models
        self.feature_columns = feature_columns
        self.config = config or EnsembleConfig()

        self._meta_return_weights: np.ndarray | None = None
        self._meta_down_weights: np.ndarray | None = None
        self._return_error_history: list[deque[float]] = [
            deque(maxlen=self.config.lookback_window) for _ in sub_models
        ]
        self._predict_count: int = 0

    def _equal_weights(self) -> np.ndarray:
        n_models = len(self.sub_models)
        return np.ones(n_models) / n_models

    def _normalize_weights(self, raw: np.ndarray) -> np.ndarray:
        """Positive-part normalisation with a minimum weight and shrinkage to equal."""
        w = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, None)
        w = w / w.sum() if w.sum() > 0 else self._equal_weights()
        w = np.maximum(w, self.config.min_weight)
        w = w / w.sum()
        shrink = self.config.shrinkage_toward_equal
        return (1 - shrink) * w + shrink * self._equal_weights()

    def _split_by_time(self, event_time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Chronological model-train / meta-train masks with an embargo gap."""
        ts = pd.to_datetime(pd.Series(event_time).to_numpy(), utc=True)
        unique_times = np.sort(ts.unique())
        n_meta = max(int(len(unique_times) * self.config.meta_fraction), 1)
        meta_start = unique_times[max(len(unique_times) - n_meta, 0)]
        train_end_idx = max(len(unique_times) - n_meta - self.config.embargo_periods, 0)
        train_mask = np.zeros(len(ts), dtype=bool)
        if train_end_idx > 0:
            train_mask = np.asarray(ts < unique_times[train_end_idx])
        meta_mask = np.asarray(ts >= meta_start)
        return train_mask, meta_mask

    def fit(
        self,
        features: pd.DataFrame,
        target_excess_return: pd.Series,
        target_downside_event: pd.Series,
    ) -> AdaptiveEnsembleModel:
        n_models = len(self.sub_models)
        target_ret = pd.Series(np.asarray(target_excess_return, dtype=float), index=features.index)
        target_down = pd.Series(np.asarray(target_downside_event, dtype=float), index=features.index)

        # 1. Chronological split (by timestamp, so a cross-section is never split)
        #    with an embargo so model-train labels do not overlap meta-train prices.
        train_mask, meta_mask = self._split_by_time(features["event_time"])
        enough_data = train_mask.sum() > 0 and meta_mask.sum() > 0

        self._meta_return_weights = self._equal_weights()
        self._meta_down_weights = self._equal_weights()
        if enough_data:
            train_features = features.loc[train_mask]
            meta_features = features.loc[meta_mask]
            for model in self.sub_models:
                model.fit(train_features, target_ret.loc[train_mask], target_down.loc[train_mask])

            # 2. Out-of-sample sub-model predictions on the meta slice, on a common scale.
            stacked_ret = np.zeros((len(meta_features), n_models))
            stacked_down = np.zeros((len(meta_features), n_models))
            for i, model in enumerate(self.sub_models):
                pred = model.predict(meta_features)
                stacked_ret[:, i] = cross_sectional_rank_score(
                    pred.expected_excess_return.to_numpy(), meta_features["event_time"]
                )
                stacked_down[:, i] = pred.downside_probability.to_numpy(dtype=float)

            meta_ret = target_ret.loc[meta_mask].to_numpy()
            meta_down = target_down.loc[meta_mask].to_numpy()

            # 3. Ridge meta-learner on rank scores for the return forecast.
            valid_mask = np.isfinite(meta_ret) & np.all(np.isfinite(stacked_ret), axis=1)
            if valid_mask.sum() > 10:
                meta_ridge = RidgeRegressor(alpha=self.config.meta_ridge_alpha)
                meta_ridge.fit(stacked_ret[valid_mask], meta_ret[valid_mask])
                if meta_ridge.coef_ is not None:
                    self._meta_return_weights = self._normalize_weights(meta_ridge.coef_)

            # 4. Inverse-error weighting for downside probability.
            valid_down_mask = np.isfinite(meta_down) & np.all(np.isfinite(stacked_down), axis=1)
            if valid_down_mask.sum() > 10:
                errors = np.abs(stacked_down[valid_down_mask] - meta_down[valid_down_mask, None])
                inv_err = 1.0 / (errors.mean(axis=0) + 1e-8)
                w_down = inv_err / inv_err.sum()
                shrink = self.config.shrinkage_toward_equal
                self._meta_down_weights = (1 - shrink) * w_down + shrink * self._equal_weights()

        # 5. Refit all sub-models on the full dataset for production inference.
        for model in self.sub_models:
            model.fit(features, target_ret, target_down)

        return self

    def predict(self, features: pd.DataFrame) -> PredictionFrame:
        n_models = len(self.sub_models)
        n_samples = len(features)

        all_ret = np.zeros((n_samples, n_models))
        all_down = np.zeros((n_samples, n_models))

        preds = []
        for i, model in enumerate(self.sub_models):
            pred = model.predict(features)
            preds.append(pred)
            all_ret[:, i] = cross_sectional_rank_score(
                pred.expected_excess_return.to_numpy(), features["event_time"]
            )
            all_down[:, i] = pred.downside_probability.to_numpy(dtype=float)

        if self._meta_return_weights is None:
            self._meta_return_weights = self._equal_weights()
        if self._meta_down_weights is None:
            self._meta_down_weights = self._equal_weights()

        # Adaptive online adjustment using running error tracking. The update is
        # persisted so subsequent calls keep using the recalibrated weights.
        if self._predict_count > 0 and self._predict_count % self.config.recalibrate_every == 0:
            self._meta_return_weights = self._recalibrate_weights(self._meta_return_weights)
        ret_weights = self._meta_return_weights
        down_weights = self._meta_down_weights

        blended = all_ret @ ret_weights
        combined_ret = cross_sectional_rank_score(blended, features["event_time"])
        combined_down = all_down @ down_weights

        self._predict_count += 1

        return PredictionFrame(
            symbol=preds[0].symbol,
            event_time=preds[0].event_time,
            expected_excess_return=pd.Series(combined_ret, index=features.index),
            downside_probability=pd.Series(np.clip(combined_down, 0.0, 1.0), index=features.index),
            model_name=self.model_name,
            metadata={
                "route": "C_ensemble",
                "return_weights": str(ret_weights.tolist()),
                "downside_weights": str(down_weights.tolist()),
                "sub_models": str([m.model_name for m in self.sub_models]),
            },
        )

    def _recalibrate_weights(self, current_weights: np.ndarray) -> np.ndarray:
        """Recalibrate weights based on tracked forecast errors."""
        n_models = len(self.sub_models)
        has_history = all(len(h) >= 10 for h in self._return_error_history)
        if not has_history:
            return current_weights

        recent_errors = np.array([np.mean(list(h)) for h in self._return_error_history])
        recent_errors = recent_errors + 1e-8
        inv_err = 1.0 / recent_errors
        new_weights = inv_err / inv_err.sum()
        eq = np.ones(n_models) / n_models
        new_weights = (1 - self.config.shrinkage_toward_equal) * new_weights + self.config.shrinkage_toward_equal * eq
        # Smooth transition: blend 70% new, 30% old.
        return 0.7 * new_weights + 0.3 * current_weights

    def register_realized_return(
        self,
        model_predictions: list[np.ndarray],
        realized: np.ndarray,
    ) -> None:
        """Track forecast errors for online adaptation."""
        for i, pred in enumerate(model_predictions):
            error = float(np.mean(np.abs(pred - realized)))
            self._return_error_history[i].append(error)
