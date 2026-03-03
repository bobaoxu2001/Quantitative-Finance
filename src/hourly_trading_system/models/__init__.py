"""Model routes and base interfaces."""

from .base import BaseSignalModel, PredictionFrame
from .route_a_dl import DLTrainingConfig, DeepCrossSectionalModel
from .route_b_lgbm import FactorLightGBMBaseline
from .route_c_ensemble import AdaptiveEnsembleModel, EnsembleConfig

__all__ = [
    "BaseSignalModel",
    "PredictionFrame",
    "DLTrainingConfig",
    "DeepCrossSectionalModel",
    "FactorLightGBMBaseline",
    "AdaptiveEnsembleModel",
    "EnsembleConfig",
]
