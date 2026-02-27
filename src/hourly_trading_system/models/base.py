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
    ) -> "BaseSignalModel":
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


class RidgeRegressor:
    """Lightweight closed-form ridge regressor without third-party dependencies."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
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
