"""Football forecast scoring and Kalshi expected-value tools."""

from .kalshi import (
    DIRECT_MEMBER_PRECISION,
    FCM_PRECISION,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    ContractEvaluation,
    Recommendation,
    american_to_prob,
    evaluate_contract,
    evaluate_match,
    fee_rate,
    kalshi_fee,
    recommend,
)
from .scoring import fit_sharpness, log_loss, multiclass_brier, sharpen, summarize

__all__ = [
    "DIRECT_MEMBER_PRECISION",
    "FCM_PRECISION",
    "MAKER_FEE_RATE",
    "TAKER_FEE_RATE",
    "ContractEvaluation",
    "Recommendation",
    "american_to_prob",
    "evaluate_contract",
    "evaluate_match",
    "fee_rate",
    "fit_sharpness",
    "kalshi_fee",
    "log_loss",
    "multiclass_brier",
    "recommend",
    "sharpen",
    "summarize",
]
