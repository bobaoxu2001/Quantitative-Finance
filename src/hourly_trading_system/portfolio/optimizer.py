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


def estimate_portfolio_volatility(
    weights: pd.Series,
    hourly_vol: pd.Series,
    assumed_correlation: float = 0.3,
    periods_per_year: float = 252 * 6.5,
) -> float:
    """
    Annualised portfolio volatility under a constant-correlation model.

    sigma_p^2 = (1 - rho) * sum_i (w_i sigma_i)^2 + rho * (sum_i w_i sigma_i)^2

    ``rho = 0`` reduces to the diagonal (independent assets) approximation and
    ``rho = 1`` to perfectly correlated assets. Equity books typically sit at
    rho ~ 0.2-0.4; ignoring correlation materially understates risk.
    """
    if weights.empty:
        return 0.0
    contrib = (weights * hourly_vol.reindex(weights.index)).fillna(0.0)
    rho = float(np.clip(assumed_correlation, 0.0, 1.0))
    variance = (1.0 - rho) * float((contrib**2).sum()) + rho * float(contrib.sum()) ** 2
    return float(np.sqrt(max(variance, 0.0)) * np.sqrt(periods_per_year))


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
        require_positive_scores: bool = True,
        selection_quantile: float = 0.0,
        score_power: float = 1.0,
        min_total_exposure: float = 0.0,
        vol_column: str = "rv_20h",
        vol_horizon_bars: int = 20,
        assumed_correlation: float = 0.3,
        periods_per_year: float = 252 * 6.5,
    ) -> None:
        """
        Parameters
        ----------
        vol_column, vol_horizon_bars:
            Column holding realised volatility and the number of hourly bars it
            spans. ``rv_20h`` is a 20-bar horizon volatility (``std * sqrt(20)``),
            so it is divided by ``sqrt(vol_horizon_bars)`` to get hourly vol
            before annualising. Use ``vol_horizon_bars=1`` for hourly vol input.
        assumed_correlation:
            Constant pairwise correlation used for the portfolio risk estimate.
        periods_per_year:
            Hourly bars per year used for annualisation.
        """
        self.config = config
        self.downside_penalty = downside_penalty
        self.cost_penalty = cost_penalty
        self.require_positive_scores = require_positive_scores
        self.selection_quantile = float(np.clip(selection_quantile, 0.0, 1.0))
        self.score_power = max(float(score_power), 0.1)
        self.min_total_exposure = float(np.clip(min_total_exposure, 0.0, 1.0))
        self.vol_column = vol_column
        self.vol_horizon_bars = max(int(vol_horizon_bars), 1)
        self.assumed_correlation = float(np.clip(assumed_correlation, 0.0, 1.0))
        self.periods_per_year = float(periods_per_year)

    def _hourly_vol(self, selected: pd.DataFrame) -> pd.Series:
        if self.vol_column in selected.columns:
            raw = selected.set_index("symbol")[self.vol_column].astype(float)
        else:
            raw = pd.Series(np.nan, index=selected["symbol"], dtype=float)
        hourly = raw.where(raw > 0.0) / np.sqrt(self.vol_horizon_bars)
        fallback = float(hourly.median()) if hourly.notna().any() else 0.01
        return hourly.fillna(fallback)

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

        Symbols present in ``current_weights`` but absent from the new target
        (e.g. names that dropped out of the universe) are treated as target 0
        and are subject to the same turnover budget as every other trade.
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
        if self.require_positive_scores:
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

        if self.selection_quantile > 0.0:
            score_cutoff = float(merged["score"].quantile(self.selection_quantile))
            selected = merged.loc[merged["score"] >= score_cutoff].copy()
            if selected.empty:
                selected = merged.head(1).copy()
        else:
            selected = merged.copy()

        raw = selected.set_index("symbol")["score"].astype(float)
        if not self.require_positive_scores:
            # In aggressive mode we preserve ranking even when absolute score
            # levels are negative by shifting scores into strictly positive space.
            raw = raw - float(raw.min()) + 1e-9
        if self.score_power != 1.0:
            raw = raw.pow(self.score_power)
        raw_weights = raw / raw.sum()
        raw_weights = _cap_and_redistribute(raw_weights, self.config.max_position_weight)

        # Soft sector control.
        sectors = selected.set_index("symbol")["sector"]
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
        per_asset_vol = self._hourly_vol(selected)
        estimated_portfolio_vol = estimate_portfolio_volatility(
            raw_weights,
            per_asset_vol,
            assumed_correlation=self.assumed_correlation,
            periods_per_year=self.periods_per_year,
        )
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
        # Exposure floor supports aggressive profiles while still respecting
        # current risk regime through the risk_scaler multiplier.
        effective_min_exposure = min(self.min_total_exposure * risk_scaler, 1.0)
        if 0.0 < raw_weights.sum() < effective_min_exposure:
            floor_scale = effective_min_exposure / max(float(raw_weights.sum()), 1e-9)
            raw_weights = (raw_weights * floor_scale).clip(upper=self.config.max_position_weight)
            raw_weights = _cap_and_redistribute(raw_weights, self.config.max_position_weight)
            raw_weights *= min(effective_min_exposure, 1.0) / max(float(raw_weights.sum()), 1e-9)

        # Turnover control using blend with existing weights. The universe is the
        # union of target and currently held names, so liquidating a name that is
        # no longer in the target also consumes turnover budget.
        universe = raw_weights.index.union(current_weights.index)
        current = current_weights.reindex(universe).fillna(0.0)
        target = raw_weights.reindex(universe).fillna(0.0)
        turnover = float((target - current).abs().sum())
        if turnover > self.config.max_turnover_per_rebalance:
            blend = self.config.max_turnover_per_rebalance / turnover
            target = current + blend * (target - current)
            turnover = float((target - current).abs().sum())
        raw_weights = target.clip(lower=0.0)
        raw_weights = raw_weights.loc[raw_weights > 1e-12]

        cash_weight = float(max(0.0, 1.0 - raw_weights.sum()))
        return PortfolioDecision(
            timestamp=timestamp,
            weights=raw_weights.sort_values(ascending=False),
            cash_weight=cash_weight,
            estimated_portfolio_vol=estimated_portfolio_vol,
            turnover=turnover,
            reason="normal",
        )
