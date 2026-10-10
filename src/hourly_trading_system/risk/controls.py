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
    """
    Drawdown-driven risk regime state machine.

    Regimes and capital scalers::

        normal (1.0) --dd >= conservative_drawdown--> conservative (0.5)
        any         --dd >= max_drawdown----------->  risk_off (0.0)
        risk_off    --dd <= risk_off_drawdown_reentry or cooldown elapsed--> conservative
        conservative --dd <= normal_reentry_drawdown and min periods held--> normal

    In ``risk_off`` the book is flat, so equity -- and therefore drawdown --
    cannot recover on its own. After ``risk_off_cooldown_periods`` updates the
    controller re-arms in ``conservative`` and re-baselines the peak to current
    equity, so the hard stop then applies to the new episode. Set the cooldown
    to ``None`` to keep the strategy flat until drawdown recovers externally.
    """

    max_drawdown: float = 0.20
    risk_off_drawdown_reentry: float = 0.15
    target_vol_low: float = 0.15
    target_vol_high: float = 0.20
    conservative_drawdown: float = 0.10
    normal_reentry_drawdown: float | None = None
    risk_off_cooldown_periods: int | None = 35
    min_conservative_periods: int = 7
    returns_history: list[float] = field(default_factory=list)
    benchmark_history: list[float] = field(default_factory=list)
    _peak_equity: float = 0.0
    _regime: str = "normal"
    _periods_in_regime: int = 0

    def _set_regime(self, regime: str) -> None:
        if regime != self._regime:
            self._regime = regime
            self._periods_in_regime = 0

    def update(self, equity: float, benchmark_return: float | None = None) -> RiskState:
        """
        Update the regime from the latest equity mark.

        ``benchmark_return`` is accepted for backwards compatibility but no
        longer mutates history; record returns with :meth:`register_return`
        so portfolio and benchmark histories stay aligned.
        """
        if self._peak_equity == 0.0:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        self._periods_in_regime += 1
        drawdown = 1.0 - (equity / self._peak_equity if self._peak_equity else 1.0)
        normal_reentry = (
            self.normal_reentry_drawdown
            if self.normal_reentry_drawdown is not None
            else self.risk_off_drawdown_reentry * 0.6
        )

        if drawdown >= self.max_drawdown and self._regime != "risk_off":
            self._set_regime("risk_off")
        elif self._regime == "risk_off":
            if drawdown <= self.risk_off_drawdown_reentry:
                self._set_regime("conservative")
            elif (
                self.risk_off_cooldown_periods is not None
                and self._periods_in_regime >= self.risk_off_cooldown_periods
            ):
                self._set_regime("conservative")
                self._peak_equity = equity
                drawdown = 0.0
        elif self._regime == "conservative":
            if drawdown <= normal_reentry and self._periods_in_regime >= self.min_conservative_periods:
                self._set_regime("normal")
        elif self._regime == "normal" and drawdown >= self.conservative_drawdown:
            self._set_regime("conservative")

        returns = np.asarray(self.returns_history[-500:], dtype=float)
        bench = np.asarray(self.benchmark_history[-500:], dtype=float)
        vol = float(np.nanstd(returns) * np.sqrt(252 * 6.5)) if returns.size else 0.0
        beta = 0.0
        if returns.size > 20 and bench.size == returns.size:
            var_b = np.var(bench)
            if var_b > 0:
                beta = float(np.cov(returns, bench, ddof=0)[0, 1] / var_b)
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
        """Record one period return; benchmark history is kept index-aligned."""
        self.returns_history.append(float(portfolio_return))
        self.benchmark_history.append(float(benchmark_return) if benchmark_return is not None else 0.0)

    @property
    def regime(self) -> str:
        return self._regime

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
