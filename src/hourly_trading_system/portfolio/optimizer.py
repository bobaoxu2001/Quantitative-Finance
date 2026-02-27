"""Portfolio construction and constrained weight allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hourly_trading_system.config import PortfolioConfig


@dataclass(slots=True)
class PortfolioDecision:
    timestamp: pd.Timestamp
    weights: pd.Series
    cash_weight: float
    estimated_portfolio_vol: float
    turnover: float
    reason: str = "normal"


def _cap_and_redistribute(weights: pd.Series, max_weight: float, max_iter: int = 20) -> pd.Series:
    out = weights.copy()
    out = out.clip(lower=0.0)
    if out.sum() <= 0:
        return out
    out = out / out.sum()
    for _ in range(max_iter):
        capped = out.clip(upper=max_weight)
        excess = float((out - capped).sum())
        out = capped
        if excess <= 1e-10:
            break
        free = out[out < max_weight]
        if free.empty:
            break
        add = excess * (free / free.sum())
        out.loc[free.index] = (out.loc[free.index] + add).clip(upper=max_weight)
    return out


class HourlyPortfolioAllocator:
    """Translate model predictions to constrained long-only target weights."""

    def __init__(
        self,
        config: PortfolioConfig,
        downside_penalty: float = 0.4,
        cost_penalty: float = 0.2,
    ) -> None:
        self.config = config
        self.downside_penalty = downside_penalty
        self.cost_penalty = cost_penalty

    def allocate(
        self,
        timestamp: pd.Timestamp,
        predictions: pd.DataFrame,
        market_snapshot: pd.DataFrame,
        current_weights: pd.Series,
        risk_scaler: float = 1.0,
    ) -> PortfolioDecision:
        """
        Allocate target weights at one timestamp.

        Required columns:
          predictions: symbol, expected_excess_return, downside_probability
          market_snapshot: symbol, close, volume, sector, rv_20h
        """
        if predictions.empty or market_snapshot.empty or risk_scaler <= 0.0:
            return PortfolioDecision(
                timestamp=timestamp,
                weights=pd.Series(dtype=float),
                cash_weight=1.0,
                estimated_portfolio_vol=0.0,
                turnover=float(current_weights.abs().sum()),
                reason="risk_off_or_no_signal",
            )

        snapshot = market_snapshot.copy()
        snapshot["dollar_volume"] = snapshot["close"] * snapshot["volume"]
        snapshot = snapshot.loc[snapshot["dollar_volume"] >= self.config.min_dollar_volume]
        if snapshot.empty:
            return PortfolioDecision(
                timestamp=timestamp,
                weights=pd.Series(dtype=float),
                cash_weight=1.0,
                estimated_portfolio_vol=0.0,
                turnover=float(current_weights.abs().sum()),
                reason="liquidity_filter",
            )

        merged = predictions.merge(snapshot, on="symbol", how="inner", suffixes=("", "_m"))
        if merged.empty:
            return PortfolioDecision(
                timestamp=timestamp,
                weights=pd.Series(dtype=float),
                cash_weight=1.0,
                estimated_portfolio_vol=0.0,
                turnover=float(current_weights.abs().sum()),
                reason="no_tradable_intersection",
            )

        # Score net of downside and estimated implementation burden.
        merged["est_cost_score"] = 1.0 / np.sqrt(merged["dollar_volume"].clip(lower=1.0))
        merged["score"] = (
            merged["expected_excess_return"]
            - self.downside_penalty * merged["downside_probability"]
            - self.cost_penalty * merged["est_cost_score"]
        )
        merged = merged.sort_values("score", ascending=False)
        merged = merged.loc[merged["score"] > 0.0]
        if merged.empty:
            return PortfolioDecision(
                timestamp=timestamp,
                weights=pd.Series(dtype=float),
                cash_weight=1.0,
                estimated_portfolio_vol=0.0,
                turnover=float(current_weights.abs().sum()),
                reason="all_scores_negative",
            )

        raw = merged.set_index("symbol")["score"]
        raw_weights = raw / raw.sum()
        raw_weights = _cap_and_redistribute(raw_weights, self.config.max_position_weight)

        # Soft sector control.
        sectors = merged.set_index("symbol")["sector"]
        sector_weight = raw_weights.groupby(sectors).sum()
        for sector, weight in sector_weight.items():
            if weight <= self.config.max_sector_weight:
                continue
            overflow = weight - self.config.max_sector_weight
            sector_symbols = sectors[sectors == sector].index
            raw_weights.loc[sector_symbols] *= self.config.max_sector_weight / max(weight, 1e-9)
            remaining_symbols = raw_weights.index.difference(sector_symbols)
            if len(remaining_symbols) > 0:
                raw_weights.loc[remaining_symbols] += overflow * (
                    raw_weights.loc[remaining_symbols] / raw_weights.loc[remaining_symbols].sum()
                )
            raw_weights = _cap_and_redistribute(raw_weights, self.config.max_position_weight)

        # Volatility scaling to keep within target band without leverage.
        per_asset_vol = merged.set_index("symbol")["rv_20h"].replace(0.0, np.nan).fillna(0.03)
        estimated_portfolio_vol = float(np.sqrt(np.sum((raw_weights * per_asset_vol) ** 2)) * np.sqrt(252 * 6.5))
        target_mid = (self.config.target_vol_lower + self.config.target_vol_upper) / 2
        if estimated_portfolio_vol > self.config.target_vol_upper and estimated_portfolio_vol > 0:
            scale = target_mid / estimated_portfolio_vol
            raw_weights *= max(min(scale, 1.0), 0.0)
        elif estimated_portfolio_vol < self.config.target_vol_lower and estimated_portfolio_vol > 0:
            scale = min(target_mid / estimated_portfolio_vol, 1.0 / raw_weights.sum())
            raw_weights *= max(scale, 0.0)

        raw_weights *= risk_scaler
        raw_weights = raw_weights.clip(lower=0.0, upper=self.config.max_position_weight)
        if raw_weights.sum() > 1.0:
            raw_weights = raw_weights / raw_weights.sum()

        # Turnover control using blend with existing weights.
        current = current_weights.reindex(raw_weights.index).fillna(0.0)
        turnover = float((raw_weights - current).abs().sum())
        if turnover > self.config.max_turnover_per_rebalance:
            blend = self.config.max_turnover_per_rebalance / turnover
            raw_weights = current + blend * (raw_weights - current)
            turnover = float((raw_weights - current).abs().sum())

        cash_weight = float(max(0.0, 1.0 - raw_weights.sum()))
        return PortfolioDecision(
            timestamp=timestamp,
            weights=raw_weights.sort_values(ascending=False),
            cash_weight=cash_weight,
            estimated_portfolio_vol=estimated_portfolio_vol,
            turnover=turnover,
            reason="normal",
        )
