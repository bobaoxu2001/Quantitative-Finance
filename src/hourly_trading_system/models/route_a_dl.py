"""Route A: Deep-learning-style multi-task cross-sectional model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseSignalModel, LogisticClassifier, PredictionFrame, RidgeRegressor, clean_feature_matrix

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False


@dataclass(slots=True)
class DLTrainingConfig:
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 512
    hidden_dim: int = 96
    dropout: float = 0.2
    early_stopping_patience: int = 4


class _TorchMultiTaskNet(nn.Module):  # pragma: no cover - tested through integration if torch exists
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(hidden_dim, 1)
        self.downside_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        ret = self.return_head(h).squeeze(-1)
        downside_logit = self.downside_head(h).squeeze(-1)
        return ret, downside_logit


class DeepCrossSectionalModel(BaseSignalModel):
    """
    Production-friendly DL route.

    If torch is not available, this model automatically falls back to
    lightweight ridge + logistic surrogates while preserving interface.
    """

    model_name = "route_a_deep_cross_sectional"

    def __init__(
        self,
        feature_columns: list[str],
        config: DLTrainingConfig | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.config = config or DLTrainingConfig()
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._reg: RidgeRegressor | None = None
        self._clf: LogisticClassifier | None = None
        self._torch_model: _TorchMultiTaskNet | None = None

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
    ) -> "DeepCrossSectionalModel":
        x, _, _ = clean_feature_matrix(features, self.feature_columns)
        y_ret = target_excess_return.to_numpy(dtype=float)
        y_down = target_downside_event.to_numpy(dtype=float)

        valid_mask = np.isfinite(y_ret) & np.isfinite(y_down)
        x = x[valid_mask]
        y_ret = y_ret[valid_mask]
        y_down = y_down[valid_mask]
        x = self._normalize(x, fit=True)

        if not TORCH_AVAILABLE:
            self._reg = RidgeRegressor(alpha=3.0).fit(x, y_ret)
            self._clf = LogisticClassifier(learning_rate=0.02, epochs=500, l2=2.0).fit(x, y_down)
            return self

        if len(x) < 32:  # pragma: no cover - small-sample fallback
            self._reg = RidgeRegressor(alpha=3.0).fit(x, y_ret)
            self._clf = LogisticClassifier(learning_rate=0.02, epochs=500, l2=2.0).fit(x, y_down)
            return self

        idx = int(len(x) * 0.85)
        x_train, x_val = x[:idx], x[idx:]
        yr_train, yr_val = y_ret[:idx], y_ret[idx:]
        yd_train, yd_val = y_down[:idx], y_down[idx:]

        model = _TorchMultiTaskNet(
            input_dim=x.shape[1],
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        yr_train_t = torch.tensor(yr_train, dtype=torch.float32)
        yd_train_t = torch.tensor(yd_train, dtype=torch.float32)
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        yr_val_t = torch.tensor(yr_val, dtype=torch.float32)
        yd_val_t = torch.tensor(yd_val, dtype=torch.float32)

        best_val = float("inf")
        patience = 0
        best_state = None
        for _ in range(self.config.epochs):
            model.train()
            permutation = torch.randperm(x_train_t.size(0))
            batch_size = self.config.batch_size
            for batch_start in range(0, x_train_t.size(0), batch_size):
                idx_batch = permutation[batch_start : batch_start + batch_size]
                xb = x_train_t[idx_batch]
                yrb = yr_train_t[idx_batch]
                ydb = yd_train_t[idx_batch]
                optimizer.zero_grad()
                pred_r, pred_d_logit = model(xb)
                loss = mse_loss(pred_r, yrb) + bce_loss(pred_d_logit, ydb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_r, val_d_logit = model(x_val_t)
                val_loss = mse_loss(val_r, yr_val_t) + bce_loss(val_d_logit, yd_val_t)
                val_loss_float = float(val_loss.item())
            if val_loss_float < best_val:
                best_val = val_loss_float
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._torch_model = model
        return self

    def predict(self, features: pd.DataFrame) -> PredictionFrame:
        x, symbols, event_time = clean_feature_matrix(features, self.feature_columns)
        x = self._normalize(x, fit=False)

        if self._torch_model is not None and TORCH_AVAILABLE:
            self._torch_model.eval()
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32)
                pred_r, pred_d_logit = self._torch_model(x_t)
                expected_excess_return = pd.Series(pred_r.numpy(), index=features.index)
                downside_probability = pd.Series(torch.sigmoid(pred_d_logit).numpy(), index=features.index)
        else:
            if self._reg is None or self._clf is None:
                raise RuntimeError("Model must be fit before inference.")
            expected_excess_return = pd.Series(self._reg.predict(x), index=features.index)
            downside_probability = pd.Series(self._clf.predict_proba(x)[:, 1], index=features.index)

        return PredictionFrame(
            symbol=symbols,
            event_time=event_time,
            expected_excess_return=expected_excess_return,
            downside_probability=downside_probability.clip(0.0, 1.0),
            model_name=self.model_name,
            metadata={"route": "A"},
        )
