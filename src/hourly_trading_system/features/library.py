"""Feature engineering library for hourly multi-asset modeling."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TRADING_HOURS_PER_YEAR = 252 * 6.5

FEATURE_COLUMNS = [
    # Price/trend/reversion
    "ret_1h",
    "ret_3h",
    "ret_5h",
    "mom_10h",
    "gap_prev_close_open",
    "intrahour_range",
    "close_open_ret",
    "dist_to_vwap_20h",
    "rsi_14h",
    "macd_hist",
    "ma20_zscore",
    "rolling_max_dd_20h",
    # Volatility/liquidity
    "rv_5h",
    "rv_20h",
    "downside_semivol_20h",
    "atr_norm_14h",
    "dollar_volume",
    "dollar_vol_z_20h",
    "turnover_proxy",
    "amihud_20h",
    "hl_spread_proxy",
    "participation_need",
    "volume_shock_20h",
    # Cross-sectional
    "beta_spy_60h",
    "idio_vol_60h",
    "residual_mom_10h",
    "sector_rel_rank_5h",
    "reversal_rank_1h",
    "corr_spy_60h",
    "breadth_percentile",
    # Fundamentals
    "f_eps_surprise_std",
    "f_revenue_surprise_std",
    "f_guidance_sentiment",
    "f_est_revision_rate",
    "f_days_since_filing",
    "f_gross_margin_ttm",
    "f_roe_ttm",
    "f_net_debt_ebitda",
    # Sentiment and analyst
    "sent_level",
    "sent_mom_6h",
    "sent_dispersion",
    "sent_mention_abnormal",
    "analyst_net_updown",
    "analyst_target_revision",
    "analyst_action_recency_hours",
    # Macro and regime
    "m_vix",
    "m_vix_chg_1h",
    "m_spy_rv_20h",
    "m_yield_curve_slope",
    "m_y2_chg_1h",
    "m_dxy_ret_5h",
    "m_wti_ret_5h",
    "m_gold_ret_5h",
    "m_cpi_surprise_state",
    "m_liquidity_stress",
    "m_regime_prob",
    # Data quality and freshness
    "hours_since_sentiment_update",
    "hours_since_fundamental_update",
    "hours_since_analyst_update",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _rolling_max_drawdown_from_returns(returns: pd.Series, window: int = 20) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    roll_max = wealth.rolling(window=window, min_periods=window).max()
    dd = wealth / roll_max - 1.0
    return dd.rolling(window=window, min_periods=1).min()


def _asof_join_symbol(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    feature_map: dict[str, str],
    known_time_col: str = "known_time",
) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    use_cols = {"symbol", known_time_col, *feature_map.keys()}
    missing = [col for col in use_cols if col not in right.columns]
    if missing:
        return left

    out = left.sort_values(["symbol", "event_time"]).copy()
    rhs = right.loc[:, sorted(use_cols)].copy().sort_values(["symbol", known_time_col])
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    rhs[known_time_col] = pd.to_datetime(rhs[known_time_col], utc=True)

    merged = pd.merge_asof(
        out,
        rhs,
        left_on="event_time",
        right_on=known_time_col,
        by="symbol",
        allow_exact_matches=False,
        direction="backward",
    )
    for src_col, dst_col in feature_map.items():
        merged[dst_col] = merged[src_col]
    return merged.drop(columns=list(feature_map.keys()) + [known_time_col], errors="ignore")


def _asof_join_global(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    feature_map: dict[str, str],
    known_time_col: str = "known_time",
) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    use_cols = {known_time_col, *feature_map.keys()}
    missing = [col for col in use_cols if col not in right.columns]
    if missing:
        return left
    out = left.sort_values("event_time").copy()
    rhs = right.loc[:, sorted(use_cols)].copy().sort_values(known_time_col)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    rhs[known_time_col] = pd.to_datetime(rhs[known_time_col], utc=True)
    merged = pd.merge_asof(
        out,
        rhs,
        left_on="event_time",
        right_on=known_time_col,
        allow_exact_matches=False,
        direction="backward",
    )
    for src_col, dst_col in feature_map.items():
        merged[dst_col] = merged[src_col]
    return merged.drop(columns=list(feature_map.keys()) + [known_time_col], errors="ignore")


def _safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, method="average", ascending=ascending)


def build_feature_panel(
    market: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    analyst: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    target_participation: float = 0.02,
) -> pd.DataFrame:
    """
    Build point-in-time safe feature panel.

    Required columns in market:
    symbol, event_time, open, high, low, close, volume
    """
    if market.empty:
        return market.copy()

    df = market.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.sort_values(["symbol", "event_time"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"market is missing required column '{col}'")

    if "sector" not in df.columns:
        df["sector"] = "UNKNOWN"
    if "shares_outstanding" not in df.columns:
        df["shares_outstanding"] = np.nan

    grouped = df.groupby("symbol", group_keys=False)
    prev_close = grouped["close"].shift(1)
    df["ret_1h"] = np.log(df["close"] / prev_close)
    df["ret_3h"] = grouped["close"].transform(lambda s: np.log(s / s.shift(3)))
    df["ret_5h"] = grouped["close"].transform(lambda s: np.log(s / s.shift(5)))
    df["mom_10h"] = grouped["close"].transform(lambda s: np.log(s / s.shift(10)))
    df["gap_prev_close_open"] = df["open"] / prev_close - 1.0
    df["intrahour_range"] = (df["high"] - df["low"]) / df["open"].replace(0.0, np.nan)
    df["close_open_ret"] = df["close"] / df["open"].replace(0.0, np.nan) - 1.0
    df["dollar_volume"] = df["close"] * df["volume"]

    rolling_volume = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).sum())
    rolling_pv = grouped.apply(
        lambda g: (g["close"] * g["volume"]).rolling(20, min_periods=5).sum()
    ).reset_index(level=0, drop=True)
    vwap_20h = rolling_pv / rolling_volume.replace(0.0, np.nan)
    df["dist_to_vwap_20h"] = df["close"] / vwap_20h - 1.0
    df["rsi_14h"] = grouped["close"].transform(_compute_rsi)

    ema12 = grouped["close"].transform(lambda s: _ema(s, 12))
    ema26 = grouped["close"].transform(lambda s: _ema(s, 26))
    macd = ema12 - ema26
    signal = macd.groupby(df["symbol"]).transform(lambda s: _ema(s, 9))
    df["macd_hist"] = macd - signal

    ma20 = grouped["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    std20 = grouped["close"].transform(lambda s: s.rolling(20, min_periods=5).std())
    df["ma20_zscore"] = (df["close"] - ma20) / std20.replace(0.0, np.nan)
    df["rolling_max_dd_20h"] = grouped["ret_1h"].transform(_rolling_max_drawdown_from_returns)

    df["rv_5h"] = grouped["ret_1h"].transform(lambda s: s.rolling(5, min_periods=3).std()) * np.sqrt(5)
    df["rv_20h"] = grouped["ret_1h"].transform(lambda s: s.rolling(20, min_periods=10).std()) * np.sqrt(20)
    df["downside_semivol_20h"] = grouped["ret_1h"].transform(
        lambda s: s.clip(upper=0.0).rolling(20, min_periods=10).std()
    ) * np.sqrt(20)

    true_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=5).mean())
    df["atr_norm_14h"] = atr_14 / df["close"].replace(0.0, np.nan)

    dvol_mean = grouped["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    dvol_std = grouped["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=5).std())
    df["dollar_vol_z_20h"] = (df["dollar_volume"] - dvol_mean) / dvol_std.replace(0.0, np.nan)
    df["turnover_proxy"] = df["volume"] / df["shares_outstanding"].replace(0.0, np.nan)
    df["amihud_20h"] = grouped.apply(
        lambda g: (g["ret_1h"].abs() / g["dollar_volume"].replace(0.0, np.nan))
        .rolling(20, min_periods=5)
        .mean()
    ).reset_index(level=0, drop=True)
    df["hl_spread_proxy"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
    df["participation_need"] = target_participation / (
        grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean()).replace(0.0, np.nan)
    )
    df["volume_shock_20h"] = df["volume"] / grouped["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).median()
    ).replace(0.0, np.nan)

    spy = (
        df.loc[df["symbol"] == benchmark_symbol, ["event_time", "ret_1h"]]
        .rename(columns={"ret_1h": "spy_ret_1h"})
        .drop_duplicates("event_time")
        .sort_values("event_time")
    )
    if spy.empty:
        df["spy_ret_1h"] = np.nan
    else:
        df = df.merge(spy, on="event_time", how="left")

    beta_list: list[pd.Series] = []
    corr_list: list[pd.Series] = []
    for _, g in df.groupby("symbol"):
        cov = g["ret_1h"].rolling(60, min_periods=20).cov(g["spy_ret_1h"])
        var = g["spy_ret_1h"].rolling(60, min_periods=20).var()
        beta = cov / var.replace(0.0, np.nan)
        corr = g["ret_1h"].rolling(60, min_periods=20).corr(g["spy_ret_1h"])
        beta_list.append(beta)
        corr_list.append(corr)
    df["beta_spy_60h"] = pd.concat(beta_list).sort_index()
    df["corr_spy_60h"] = pd.concat(corr_list).sort_index()
    residual = df["ret_1h"] - df["beta_spy_60h"] * df["spy_ret_1h"]
    df["idio_vol_60h"] = residual.groupby(df["symbol"]).transform(
        lambda s: s.rolling(60, min_periods=20).std()
    ) * np.sqrt(TRADING_HOURS_PER_YEAR)
    df["residual_mom_10h"] = (
        df["mom_10h"] - df["beta_spy_60h"] * df["spy_ret_1h"].groupby(df["symbol"]).transform(lambda s: s.rolling(10, min_periods=3).sum())
    )

    df["sector_rel_rank_5h"] = df.groupby(["event_time", "sector"])["ret_5h"].transform(_safe_rank)
    df["reversal_rank_1h"] = df.groupby("event_time")["ret_1h"].transform(lambda s: _safe_rank(-s))
    df["breadth_percentile"] = df.groupby("event_time")["ret_1h"].transform(_safe_rank)

    df = _asof_join_symbol(
        df,
        fundamentals,
        {
            "eps_surprise_std": "f_eps_surprise_std",
            "revenue_surprise_std": "f_revenue_surprise_std",
            "guidance_sentiment": "f_guidance_sentiment",
            "est_revision_rate": "f_est_revision_rate",
            "days_since_filing": "f_days_since_filing",
            "gross_margin_ttm": "f_gross_margin_ttm",
            "roe_ttm": "f_roe_ttm",
            "net_debt_ebitda": "f_net_debt_ebitda",
        },
    )
    df = _asof_join_symbol(
        df,
        sentiment,
        {
            "sentiment_score": "sent_level",
            "sentiment_dispersion": "sent_dispersion",
            "mention_abnormal": "sent_mention_abnormal",
            "hours_since_update": "hours_since_sentiment_update",
            "sentiment_update_time": "sentiment_update_time",
        },
    )
    df = _asof_join_symbol(
        df,
        analyst,
        {
            "net_upgrades_downgrades": "analyst_net_updown",
            "target_price_revision": "analyst_target_revision",
            "hours_since_update": "hours_since_analyst_update",
            "analyst_update_time": "analyst_update_time",
        },
    )
    df = _asof_join_global(
        df,
        macro,
        {
            "vix": "m_vix",
            "yield_10y": "m_yield_10y",
            "yield_2y": "m_yield_2y",
            "dxy": "m_dxy",
            "wti": "m_wti",
            "gold": "m_gold",
            "cpi_surprise_state": "m_cpi_surprise_state",
            "liquidity_stress": "m_liquidity_stress",
            "regime_prob": "m_regime_prob",
        },
    )

    # Derived sentiment/analyst/macro features after joins.
    df["sent_mom_6h"] = df.groupby("symbol")["sent_level"].transform(lambda s: s - s.shift(6))
    if "sentiment_update_time" in df.columns:
        sent_update = pd.to_datetime(df["sentiment_update_time"], utc=True, errors="coerce")
        df["hours_since_sentiment_update"] = (
            (df["event_time"] - sent_update).dt.total_seconds() / 3600.0
        )
    if "analyst_update_time" in df.columns:
        analyst_update = pd.to_datetime(df["analyst_update_time"], utc=True, errors="coerce")
        df["analyst_action_recency_hours"] = (
            (df["event_time"] - analyst_update).dt.total_seconds() / 3600.0
        )
    else:
        df["analyst_action_recency_hours"] = np.nan

    if "known_time" in df.columns:
        fund_known = pd.to_datetime(df["known_time"], utc=True, errors="coerce")
        df["hours_since_fundamental_update"] = (
            (df["event_time"] - fund_known).dt.total_seconds() / 3600.0
        )

    df["m_vix_chg_1h"] = df.groupby("symbol")["m_vix"].transform(lambda s: s - s.shift(1))
    df["m_spy_rv_20h"] = (
        df["spy_ret_1h"]
        .rolling(20, min_periods=10)
        .std()
        .reindex(df.index)
    )
    df["m_yield_curve_slope"] = df["m_yield_10y"] - df["m_yield_2y"]
    df["m_y2_chg_1h"] = df.groupby("symbol")["m_yield_2y"].transform(lambda s: s - s.shift(1))
    df["m_dxy_ret_5h"] = df.groupby("symbol")["m_dxy"].transform(lambda s: np.log(s / s.shift(5)))
    df["m_wti_ret_5h"] = df.groupby("symbol")["m_wti"].transform(lambda s: np.log(s / s.shift(5)))
    df["m_gold_ret_5h"] = df.groupby("symbol")["m_gold"].transform(lambda s: np.log(s / s.shift(5)))

    # Ensure all required features exist.
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = np.nan

    return df[["symbol", "event_time", "sector", "open", "high", "low", "close", "volume", *FEATURE_COLUMNS]].copy()


def required_features() -> list[str]:
    """Return canonical list of feature names."""
    return list(FEATURE_COLUMNS)


def ensure_feature_columns(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Add missing features as NaN to keep train/inference schema stable."""
    out = frame.copy()
    for feature in features:
        if feature not in out.columns:
            out[feature] = np.nan
    return out
