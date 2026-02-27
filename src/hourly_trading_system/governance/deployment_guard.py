"""Canary promotion and rollback decision logic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CanaryPolicy:
    """Decision thresholds for canary rollout governance."""

    min_observations: int = 40
    max_relative_drawdown_gap: float = 0.03
    min_information_ratio_gap: float = -0.20
    max_turnover_gap: float = 0.10

    def evaluate(
        self,
        champion_returns: pd.Series,
        canary_returns: pd.Series,
        champion_turnover: pd.Series | None = None,
        canary_turnover: pd.Series | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Return one of: hold, promote, rollback."""
        aligned = pd.concat(
            [champion_returns.rename("champ"), canary_returns.rename("canary")],
            axis=1,
        ).dropna()
        if len(aligned) < self.min_observations:
            return "hold", {"observations": float(len(aligned))}

        champ = aligned["champ"]
        canary = aligned["canary"]
        champ_dd = float((1.0 + champ).cumprod().div((1.0 + champ).cumprod().cummax()).sub(1.0).min())
        canary_dd = float((1.0 + canary).cumprod().div((1.0 + canary).cumprod().cummax()).sub(1.0).min())
        dd_gap = abs(canary_dd) - abs(champ_dd)

        champ_ir = float(champ.mean() / champ.std(ddof=0)) if champ.std(ddof=0) > 0 else 0.0
        canary_ir = float(canary.mean() / canary.std(ddof=0)) if canary.std(ddof=0) > 0 else 0.0
        ir_gap = canary_ir - champ_ir

        turnover_gap = 0.0
        if champion_turnover is not None and canary_turnover is not None:
            c1 = champion_turnover.reindex(aligned.index).fillna(0.0)
            c2 = canary_turnover.reindex(aligned.index).fillna(0.0)
            turnover_gap = float(c2.mean() - c1.mean())

        stats = {
            "observations": float(len(aligned)),
            "dd_gap": dd_gap,
            "ir_gap": ir_gap,
            "turnover_gap": turnover_gap,
        }
        if dd_gap > self.max_relative_drawdown_gap:
            return "rollback", stats
        if ir_gap < self.min_information_ratio_gap:
            return "rollback", stats
        if turnover_gap > self.max_turnover_gap:
            return "rollback", stats
        if ir_gap > 0.10 and dd_gap <= 0.0:
            return "promote", stats
        return "hold", stats


def deterministic_canary_assignment(key: str, canary_allocation: float) -> bool:
    """
    Deterministically assign key to canary traffic bucket.

    This is stable across services and processes for identical keys.
    """
    if canary_allocation <= 0:
        return False
    if canary_allocation >= 1:
        return True
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    as_int = int(digest[:16], 16)
    bucket = (as_int % 10_000) / 10_000
    return bucket < canary_allocation


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    """Small helper for routing live performance to logs/alerts."""
    if returns.empty:
        return {"mean": 0.0, "vol": 0.0, "p95": 0.0, "p05": 0.0}
    return {
        "mean": float(returns.mean()),
        "vol": float(returns.std(ddof=0)),
        "p95": float(np.quantile(returns, 0.95)),
        "p05": float(np.quantile(returns, 0.05)),
    }
