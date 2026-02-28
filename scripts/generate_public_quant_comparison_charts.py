#!/usr/bin/env python3
"""Generate public quant model comparison charts for README."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "public_quant_model_comparison.csv"
OUT_DIR = ROOT / "docs" / "assets"


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    numeric_cols = [
        "annualized_return_pct",
        "aum_or_capacity_usd_bn",
        "cumulative_net_gains_usd_bn",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["estimated_annual_pnl_usd_bn"] = (
        df["annualized_return_pct"] / 100.0 * df["aum_or_capacity_usd_bn"]
    )
    return df


def _save_fig(fig, filename: str, width: int = 1300, height: int = 760) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.update_layout(template="plotly_white")
    fig.write_image(str(out_path), width=width, height=height, scale=2)
    print(f"Saved: {out_path.relative_to(ROOT)}")


def chart_annualized_return(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["annualized_return_pct"]).sort_values(
        "annualized_return_pct", ascending=False
    )
    fig = px.bar(
        plot_df,
        x="annualized_return_pct",
        y="name",
        orientation="h",
        color="category",
        text="annualized_return_pct",
        title="Public Quant Models: Annualized Return Comparison",
        labels={
            "annualized_return_pct": "Annualized return (%)",
            "name": "",
            "category": "Vehicle type",
        },
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        xaxis_tickformat=".1f",
        legend_title_text="Vehicle type",
        margin=dict(l=80, r=40, t=80, b=50),
    )
    _save_fig(fig, "public_quant_returns.png")


def chart_aum(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["aum_or_capacity_usd_bn"]).sort_values(
        "aum_or_capacity_usd_bn", ascending=False
    )
    fig = px.bar(
        plot_df,
        x="aum_or_capacity_usd_bn",
        y="name",
        orientation="h",
        color="category",
        text="aum_or_capacity_usd_bn",
        title="Public Quant Models: AUM / Capacity (USD bn)",
        labels={
            "aum_or_capacity_usd_bn": "AUM or capacity (USD bn)",
            "name": "",
            "category": "Vehicle type",
        },
    )
    fig.update_traces(texttemplate="%{text:.3g} bn", textposition="outside")
    fig.update_layout(
        xaxis_type="log",
        legend_title_text="Vehicle type",
        margin=dict(l=80, r=40, t=80, b=50),
    )
    _save_fig(fig, "public_quant_aum.png")


def chart_estimated_profit(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["estimated_annual_pnl_usd_bn"]).sort_values(
        "estimated_annual_pnl_usd_bn", ascending=False
    )
    fig = px.bar(
        plot_df,
        x="estimated_annual_pnl_usd_bn",
        y="name",
        orientation="h",
        color="category",
        text="estimated_annual_pnl_usd_bn",
        title="Estimated Annual Profit Potential = AUM x Annualized Return",
        labels={
            "estimated_annual_pnl_usd_bn": "Estimated annual profit (USD bn, rough)",
            "name": "",
            "category": "Vehicle type",
        },
    )
    fig.update_traces(texttemplate="%{text:.3g} bn", textposition="outside")
    fig.update_layout(
        xaxis_type="log",
        legend_title_text="Vehicle type",
        margin=dict(l=80, r=40, t=80, b=50),
    )
    _save_fig(fig, "public_quant_estimated_profit.png")


def chart_cumulative_gains(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["cumulative_net_gains_usd_bn"]).sort_values(
        "cumulative_net_gains_usd_bn", ascending=False
    )
    fig = px.bar(
        plot_df,
        x="cumulative_net_gains_usd_bn",
        y="name",
        orientation="h",
        color="category",
        text="cumulative_net_gains_usd_bn",
        title="Publicly Reported Cumulative Net Gains to Investors (USD bn)",
        labels={
            "cumulative_net_gains_usd_bn": "Cumulative net gains (USD bn)",
            "name": "",
            "category": "Vehicle type",
        },
    )
    fig.update_traces(texttemplate="%{text:.1f} bn", textposition="outside")
    fig.update_layout(
        legend_title_text="Vehicle type",
        margin=dict(l=80, r=40, t=80, b=50),
    )
    _save_fig(fig, "public_quant_cumulative_gains.png")


def chart_return_vs_scale(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["annualized_return_pct", "aum_or_capacity_usd_bn"]).copy()
    plot_df["bubble"] = plot_df["cumulative_net_gains_usd_bn"].fillna(1.0)
    fig = px.scatter(
        plot_df,
        x="aum_or_capacity_usd_bn",
        y="annualized_return_pct",
        size="bubble",
        size_max=60,
        color="category",
        text="name",
        hover_data={
            "strategy_family": True,
            "period_basis": True,
            "annualized_return_pct": ":.2f",
            "aum_or_capacity_usd_bn": ":.3f",
            "cumulative_net_gains_usd_bn": ":.1f",
            "bubble": False,
        },
        title="Return vs Scale (bubble size = cumulative net gains when available)",
        labels={
            "aum_or_capacity_usd_bn": "AUM / capacity (USD bn, log scale)",
            "annualized_return_pct": "Annualized return (%)",
            "category": "Vehicle type",
        },
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_type="log",
        legend_title_text="Vehicle type",
        margin=dict(l=80, r=40, t=80, b=50),
    )
    _save_fig(fig, "public_quant_return_vs_scale.png")


def main() -> None:
    df = _load_data()
    chart_annualized_return(df)
    chart_aum(df)
    chart_estimated_profit(df)
    chart_cumulative_gains(df)
    chart_return_vs_scale(df)
    print("Done.")


if __name__ == "__main__":
    main()
