"""Base model interfaces and preprocessing helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PredictionFrame:
    symbol: pd.Series
    event_time: pd.Series
    expected_excess_return: pd.Series
    downside_probability: pd.Series
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "symbol": self.symbol.values,
                "event_time": self.event_time.values,
                "expected_excess_return": self.expected_excess_return.values,
                "downside_probability": self.downside_probability.values,
                "model_name": self.model_name,
            }
        )
        for key, value in self.metadata.items():
            frame[key] = value
        return frame


class BaseSignalModel(ABC):
    """Abstract interface for signal models."""

    model_name: str = "base"

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        target_excess_return: pd.Series,
        target_downside_event: pd.Series,
    ) -> BaseSignalModel:
        """Train model parameters."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> PredictionFrame:
        """Run inference and return expected return and downside probability."""


def clean_feature_matrix(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Prepare matrix with deterministic missing-value treatment."""
    missing = [col for col in feature_columns if col not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    matrix = features.loc[:, feature_columns].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    x = matrix.to_numpy(dtype=float)
    symbols = features["symbol"].copy()
    event_time = pd.to_datetime(features["event_time"], utc=True)
    return x, symbols, event_time


def cross_sectional_rank_score(values: pd.Series | np.ndarray, event_time: pd.Series) -> np.ndarray:
    """
    Map raw model outputs to a centred cross-sectional rank score in [-0.5, 0.5].

    Ranks are computed independently per ``event_time`` so that scores from
    models with very different output scales (raw returns vs. probabilities vs.
    ranks) become directly comparable and can be blended.
    """
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "event_time": pd.to_datetime(pd.Series(event_time).to_numpy(), utc=True),
        }
    )
    ranks = frame.groupby("event_time", sort=False)["value"].rank(pct=True)
    return (ranks - 0.5).fillna(0.0).to_numpy(dtype=float)


class FeaturePreprocessor:
    """
    Train-fitted feature preprocessing for consistent training and inference.

    Statistics (median for imputation, mean/std for standardisation) are learned
    once from the training sample and re-used at inference time. Imputing with
    statistics of the *inference batch* would make the same row's features
    depend on which other symbols happen to be scored alongside it.
    """

    def __init__(self, feature_columns: list[str], clip_zscore: float | None = 5.0) -> None:
        self.feature_columns = list(feature_columns)
        self.clip_zscore = clip_zscore
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    def _raw(self, features: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_columns if col not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        x = features.loc[:, self.feature_columns].to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x

    @property
    def is_fitted(self) -> bool:
        return self.medians_ is not None

    def fit(self, features: pd.DataFrame) -> FeaturePreprocessor:
        x = self._raw(features)
        medians = pd.DataFrame(x).median(axis=0, skipna=True).to_numpy(dtype=float)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isnan(x), medians, x)
        means = filled.mean(axis=0) if len(filled) else np.zeros(len(self.feature_columns))
        stds = filled.std(axis=0) if len(filled) else np.ones(len(self.feature_columns))
        stds = np.where(np.isfinite(stds) & (stds > 0.0), stds, 1.0)
        self.medians_, self.means_, self.stds_ = medians, means, stds
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform.")
        x = self._raw(features)
        x = np.where(np.isnan(x), self.medians_, x)
        z = (x - self.means_) / self.stds_
        if self.clip_zscore is not None:
            z = np.clip(z, -self.clip_zscore, self.clip_zscore)
        return z

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        return self.fit(features).transform(features)


def time_order(event_time: pd.Series) -> np.ndarray:
    """Return a stable permutation that sorts rows by event_time."""
    ts = pd.to_datetime(pd.Series(event_time).to_numpy(), utc=True)
    return np.argsort(ts.to_numpy(), kind="stable")


class RidgeRegressor:
    """Lightweight closed-form ridge regressor without third-party dependencies."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeRegressor:
        ones = np.ones((x.shape[0], 1))
        x_aug = np.hstack([ones, x])
        identity = np.eye(x_aug.shape[1])
        identity[0, 0] = 0.0  # Do not regularize intercept.
        beta = np.linalg.pinv(x_aug.T @ x_aug + self.alpha * identity) @ (x_aug.T @ y)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before predict.")
        return self.intercept_ + x @ self.coef_


class LogisticClassifier:
    """Numerically stable logistic regression using gradient descent."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 400, l2: float = 1.0) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticClassifier:
        n_samples, n_features = x.shape
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0
        y = y.astype(float)
        for _ in range(self.epochs):
            logits = x @ weights + bias
            preds = self._sigmoid(logits)
            diff = preds - y
            grad_w = (x.T @ diff) / n_samples + self.l2 * weights / n_samples
            grad_b = float(diff.mean())
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b
        self.weights_ = weights
        self.bias_ = bias
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fit before predict.")
        logits = x @ self.weights_ + self.bias_
        probs = self._sigmoid(logits)
        return np.vstack([1.0 - probs, probs]).T
