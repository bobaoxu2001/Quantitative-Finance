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
from .scoring import log_loss, multiclass_brier, summarize

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
    "kalshi_fee",
    "log_loss",
    "multiclass_brier",
    "recommend",
    "summarize",
]
