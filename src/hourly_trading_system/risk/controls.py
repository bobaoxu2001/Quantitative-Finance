"""Risk controls and drawdown governance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskState:
    peak_equity: float
    current_equity: float
    drawdown: float
    regime: str = "normal"
    realized_vol_annualized: float = 0.0
    beta_to_spy: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0


@dataclass(slots=True)
class RiskController:
    max_drawdown: float = 0.20
    risk_off_drawdown_reentry: float = 0.15
    target_vol_low: float = 0.15
    target_vol_high: float = 0.20
    returns_history: list[float] = field(default_factory=list)
    benchmark_history: list[float] = field(default_factory=list)
    _peak_equity: float = 0.0
    _regime: str = "normal"

    def update(self, equity: float, benchmark_return: float | None = None) -> RiskState:
        if self._peak_equity == 0.0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = 1.0 - (equity / self._peak_equity if self._peak_equity else 1.0)

        if drawdown >= self.max_drawdown:
            self._regime = "risk_off"
        elif self._regime == "risk_off" and drawdown <= self.risk_off_drawdown_reentry:
            self._regime = "conservative"
        elif self._regime == "conservative" and drawdown <= self.risk_off_drawdown_reentry * 0.6:
            self._regime = "normal"

        if benchmark_return is not None and self.returns_history:
            self.benchmark_history.append(float(benchmark_return))

        returns = np.asarray(self.returns_history[-500:], dtype=float)
        bench = np.asarray(self.benchmark_history[-500:], dtype=float)
        vol = float(np.nanstd(returns) * np.sqrt(252 * 6.5)) if returns.size else 0.0
        beta = 0.0
        if returns.size > 20 and bench.size == returns.size:
            var_b = np.var(bench)
            if var_b > 0:
                beta = float(np.cov(returns, bench)[0, 1] / var_b)
        var_95 = 0.0
        cvar_95 = 0.0
        if returns.size > 20:
            var_95 = float(np.quantile(returns, 0.05))
            cvar_95 = float(returns[returns <= var_95].mean()) if np.any(returns <= var_95) else var_95

        return RiskState(
            peak_equity=self._peak_equity,
            current_equity=equity,
            drawdown=drawdown,
            regime=self._regime,
            realized_vol_annualized=vol,
            beta_to_spy=beta,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def register_return(self, portfolio_return: float, benchmark_return: float | None = None) -> None:
        self.returns_history.append(float(portfolio_return))
        if benchmark_return is not None:
            self.benchmark_history.append(float(benchmark_return))

    def capital_scaler(self) -> float:
        """Return capital scaling factor from current risk regime."""
        if self._regime == "risk_off":
            return 0.0
        if self._regime == "conservative":
            return 0.5
        return 1.0


def sector_exposure(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    """Compute sector level exposures from symbol weights."""
    merged = pd.DataFrame({"weight": weights, "sector": sectors})
    return merged.groupby("sector")["weight"].sum().sort_values(ascending=False)
