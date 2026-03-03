"""Route B: Factor library + LightGBM-style baseline ranker."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseSignalModel, LogisticClassifier, PredictionFrame, RidgeRegressor, clean_feature_matrix

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LIGHTGBM_AVAILABLE = False


class FactorLightGBMBaseline(BaseSignalModel):
    """
    Production baseline route.

    Uses LightGBM when available, otherwise falls back to deterministic
    ridge/logistic implementation to keep pipeline executable.
    """

    model_name = "route_b_factor_lgbm"

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._reg: RidgeRegressor | None = None
        self._clf: LogisticClassifier | None = None
        self._ranker: object | None = None
        self._down_classifier: object | None = None
        self._feature_importances: np.ndarray | None = None

    def _normalize(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit or self._x_mean is None or self._x_std is None:
            self._x_mean = x.mean(axis=0)
            self._x_std = x.std(axis=0)
            self._x_std[self._x_std == 0.0] = 1.0
        return (x - self._x_mean) / self._x_std

    def fit(
        self,
        features: pd.DataFrame,
        target_excess_return: pd.Series,
        target_downside_event: pd.Series,
    ) -> "FactorLightGBMBaseline":
        x, _, _ = clean_feature_matrix(features, self.feature_columns)
        y_ret = target_excess_return.to_numpy(dtype=float)
        y_down = target_downside_event.to_numpy(dtype=float)
        valid = np.isfinite(y_ret) & np.isfinite(y_down)
        x = x[valid]
        y_ret = y_ret[valid]
        y_down = y_down[valid]
        x = self._normalize(x, fit=True)

        if LIGHTGBM_AVAILABLE:
            # Ranking objective approximates cross-sectional alpha ordering.
            self._ranker = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbosity=-1,
            )
            self._down_classifier = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbosity=-1,
            )
            # Early stopping with train/validation split to prevent overfitting.
            split_idx = int(len(x) * 0.85)
            if split_idx > 20 and len(x) - split_idx > 10:
                x_tr, x_vl = x[:split_idx], x[split_idx:]
                yr_tr, yr_vl = y_ret[:split_idx], y_ret[split_idx:]
                yd_tr, yd_vl = y_down[:split_idx], y_down[split_idx:]
                self._ranker.fit(
                    x_tr, yr_tr,
                    eval_set=[(x_vl, yr_vl)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                self._down_classifier.fit(
                    x_tr, yd_tr,
                    eval_set=[(x_vl, yd_vl)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            else:
                self._ranker.fit(x, y_ret)
                self._down_classifier.fit(x, y_down)
            self._feature_importances = self._ranker.feature_importances_.astype(float)
            return self

        self._reg = RidgeRegressor(alpha=1.5).fit(x, y_ret)
        self._clf = LogisticClassifier(learning_rate=0.015, epochs=450, l2=1.5).fit(x, y_down)
        return self

    def predict(self, features: pd.DataFrame) -> PredictionFrame:
        x, symbols, event_time = clean_feature_matrix(features, self.feature_columns)
        x = self._normalize(x, fit=False)

        if LIGHTGBM_AVAILABLE and self._ranker is not None and self._down_classifier is not None:
            pred_ret = pd.Series(self._ranker.predict(x), index=features.index)
            pred_down = pd.Series(self._down_classifier.predict_proba(x)[:, 1], index=features.index)
        else:
            if self._reg is None or self._clf is None:
                raise RuntimeError("Model must be fit before inference.")
            pred_ret = pd.Series(self._reg.predict(x), index=features.index)
            pred_down = pd.Series(self._clf.predict_proba(x)[:, 1], index=features.index)

        # Transform expected return into cross-sectional rank score per hour.
        out = pd.DataFrame(
            {
                "symbol": symbols.values,
                "event_time": event_time.values,
                "pred_ret": pred_ret.values,
                "pred_down": pred_down.values,
            },
            index=features.index,
        )
        out["expected_excess_return"] = out.groupby("event_time")["pred_ret"].transform(
            lambda s: s.rank(pct=True) - 0.5
        )

        return PredictionFrame(
            symbol=symbols,
            event_time=event_time,
            expected_excess_return=out["expected_excess_return"],
            downside_probability=out["pred_down"].clip(0.0, 1.0),
            model_name=self.model_name,
            metadata={"route": "B"},
        )

    @property
    def feature_importances(self) -> pd.Series | None:
        """Return feature importances from the LightGBM ranker (if available)."""
        if self._feature_importances is None:
            return None
        return pd.Series(self._feature_importances, index=self.feature_columns[:len(self._feature_importances)])
