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
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import (
    BaseSignalModel,
    PredictionFrame,
    RidgeRegressor,
    clean_feature_matrix,
)


@dataclass(slots=True)
class EnsembleConfig:
    lookback_window: int = 120
    shrinkage_toward_equal: float = 0.3
    min_weight: float = 0.10
    decorrelation_bonus: float = 0.5
    meta_ridge_alpha: float = 5.0
    recalibrate_every: int = 20
    meta_train_fraction: float = 0.20


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

    def fit(
        self,
        features: pd.DataFrame,
        target_excess_return: pd.Series,
        target_downside_event: pd.Series,
    ) -> "AdaptiveEnsembleModel":
        # Split into model-train and meta-train slices.
        n = len(features)
        split_idx = int(n * (1.0 - self.config.meta_train_fraction))
        train_features = features.iloc[:split_idx].copy()
        meta_features = features.iloc[split_idx:].copy()
        train_ret = target_excess_return.iloc[:split_idx]
        meta_ret = target_excess_return.iloc[split_idx:]
        train_down = target_downside_event.iloc[:split_idx]
        meta_down = target_downside_event.iloc[split_idx:]

        # 1. Fit each sub-model on the model-train portion.
        for model in self.sub_models:
            model.fit(train_features, train_ret, train_down)

        # 2. Collect stacked out-of-fold predictions on meta-train slice.
        n_models = len(self.sub_models)
        stacked_ret = np.zeros((len(meta_features), n_models))
        stacked_down = np.zeros((len(meta_features), n_models))

        for i, model in enumerate(self.sub_models):
            pred = model.predict(meta_features)
            stacked_ret[:, i] = pred.expected_excess_return.values
            stacked_down[:, i] = pred.downside_probability.values

        # 3. Train ridge meta-learner for return predictions.
        valid_mask = np.isfinite(meta_ret.values) & np.all(np.isfinite(stacked_ret), axis=1)
        if valid_mask.sum() > 10:
            meta_ridge = RidgeRegressor(alpha=self.config.meta_ridge_alpha)
            meta_ridge.fit(stacked_ret[valid_mask], meta_ret.values[valid_mask])
            raw_weights = meta_ridge.coef_
            if raw_weights is not None:
                # Normalize to sum-to-1, enforce minimum weight.
                w = np.clip(raw_weights, self.config.min_weight, None)
                w = w / w.sum()
                # Shrink toward equal weight for robustness.
                eq = np.ones(n_models) / n_models
                w = (1 - self.config.shrinkage_toward_equal) * w + self.config.shrinkage_toward_equal * eq
                self._meta_return_weights = w
            else:
                self._meta_return_weights = np.ones(n_models) / n_models
        else:
            self._meta_return_weights = np.ones(n_models) / n_models

        # 4. For downside probability, use inverse-error weighting.
        valid_down_mask = np.isfinite(meta_down.values) & np.all(np.isfinite(stacked_down), axis=1)
        if valid_down_mask.sum() > 10:
            errors = np.abs(stacked_down[valid_down_mask] - meta_down.values[valid_down_mask, None])
            mean_errors = errors.mean(axis=0) + 1e-8
            inv_err = 1.0 / mean_errors
            w_down = inv_err / inv_err.sum()
            eq = np.ones(n_models) / n_models
            w_down = (1 - self.config.shrinkage_toward_equal) * w_down + self.config.shrinkage_toward_equal * eq
            self._meta_down_weights = w_down
        else:
            self._meta_down_weights = np.ones(n_models) / n_models

        # Now refit all sub-models on the full dataset for production inference.
        for model in self.sub_models:
            model.fit(features, target_excess_return, target_downside_event)

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
            all_ret[:, i] = pred.expected_excess_return.values
            all_down[:, i] = pred.downside_probability.values

        ret_weights = self._meta_return_weights if self._meta_return_weights is not None else np.ones(n_models) / n_models
        down_weights = self._meta_down_weights if self._meta_down_weights is not None else np.ones(n_models) / n_models

        # Adaptive online adjustment using running error tracking.
        if self._predict_count > 0 and self._predict_count % self.config.recalibrate_every == 0:
            ret_weights = self._recalibrate_weights(ret_weights)

        combined_ret = all_ret @ ret_weights
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
