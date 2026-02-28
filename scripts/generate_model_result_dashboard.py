"""Generate an HTML dashboard and summary artifacts for model results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hourly_trading_system.analytics import compute_performance_report
from hourly_trading_system.dashboard import load_live_queue_snapshot, summarize_live_snapshot


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _build_backtest_figure(equity_curve: pd.DataFrame) -> go.Figure:
    curve = equity_curve.copy()
    curve["event_time"] = pd.to_datetime(curve["event_time"], utc=True)
    curve = curve.sort_values("event_time")
    curve["returns"] = curve["equity"].pct_change().fillna(0.0)
    curve["drawdown"] = _drawdown_series(curve["equity"])
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Equity Curve", "Drawdown", "Returns Distribution", "Exposure / Turnover"),
    )
    fig.add_trace(
        go.Scatter(x=curve["event_time"], y=curve["equity"], mode="lines", name="Equity"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["event_time"], y=curve["drawdown"], mode="lines", name="Drawdown"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=curve["returns"], nbinsx=60, name="Returns"),
        row=2,
        col=1,
    )
    if "exposure" in curve.columns:
        fig.add_trace(
            go.Scatter(x=curve["event_time"], y=curve["exposure"], mode="lines", name="Exposure"),
            row=2,
            col=2,
        )
    if "turnover" in curve.columns:
        fig.add_trace(
            go.Scatter(x=curve["event_time"], y=curve["turnover"], mode="lines", name="Turnover"),
            row=2,
            col=2,
        )
    fig.update_layout(height=850, width=1450, title_text="Quant Model Backtest Overview", template="plotly_white")
    return fig


def _monthly_heatmap(monthly_returns: pd.DataFrame) -> go.Figure:
    if monthly_returns.empty:
        return go.Figure()
    data = monthly_returns.copy()
    if "year" in data.columns:
        index_col = "year"
        data = data.set_index(index_col)
    z = data.to_numpy(dtype=float)
    x = [str(c) for c in data.columns]
    y = [str(idx) for idx in data.index]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar={"title": "Monthly Return"},
        )
    )
    fig.update_layout(title="Monthly Returns Heatmap", template="plotly_white")
    return fig


def _live_queue_figure(queue_dir: Path) -> go.Figure:
    snapshot = load_live_queue_snapshot(queue_dir)
    summary = summarize_live_snapshot(snapshot)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Queue Depths", "Health Score / TCA"),
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )
    topics = summary["topics"]
    fig.add_trace(
        go.Bar(x=list(topics.keys()), y=list(topics.values()), name="Queue Depth"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=float(summary.get("health_score", 0.0) or 0.0),
            title={"text": "Health Score"},
            gauge={"axis": {"range": [0, 100]}},
        ),
        row=1,
        col=2,
    )
    tca = summary.get("tca_avg_cost_bps")
    fig.add_annotation(
        x=0.81,
        y=0.12,
        xref="paper",
        yref="paper",
        text=f"TCA Avg Cost (bps): {tca}",
        showarrow=False,
        font={"size": 13},
    )
    fig.update_layout(height=500, width=1450, title_text="Live Queue Health Overview", template="plotly_white")
    return fig


def generate_report(output_dir: Path = Path("outputs")) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity = _read_csv(output_dir / "equity_curve.csv")
    fills = _read_csv(output_dir / "fills.csv")
    monthly = _read_csv(output_dir / "monthly_returns.csv")
    if equity.empty:
        raise RuntimeError("Missing outputs/equity_curve.csv. Run demo backtest first.")

    for frame in (equity, fills):
        if "event_time" in frame.columns:
            frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)

    perf = compute_performance_report(
        equity_curve=equity[["event_time", "equity"]],
        turnover_series=equity["turnover"] if "turnover" in equity.columns else None,
        exposure_series=equity["exposure"] if "exposure" in equity.columns else None,
        trades=fills,
    )
    summary = perf.summary
    summary["total_return"] = perf.extras["total_return"]
    summary["trade_count"] = perf.extras["trade_count"]

    backtest_fig = _build_backtest_figure(equity)
    monthly_fig = _monthly_heatmap(monthly)
    live_fig = _live_queue_figure(output_dir / "live_queue")

    dashboard_path = output_dir / "model_result_dashboard.html"
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Model Result Dashboard</title></head><body>",
        "<h1>Quant Model Result Dashboard</h1>",
        "<h2>Summary</h2>",
        f"<pre>{json.dumps(summary, indent=2)}</pre>",
        backtest_fig.to_html(full_html=False, include_plotlyjs="cdn"),
        monthly_fig.to_html(full_html=False, include_plotlyjs=False),
        live_fig.to_html(full_html=False, include_plotlyjs=False),
        "</body></html>",
    ]
    dashboard_path.write_text("\n".join(html_parts), encoding="utf-8")

    summary_path = output_dir / "model_result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional PNG export when kaleido is available.
    png_path = output_dir / "equity_drawdown.png"
    try:
        backtest_fig.write_image(str(png_path), width=1500, height=900)
    except Exception:
        png_path = None

    return {
        "dashboard_html": str(dashboard_path),
        "summary_json": str(summary_path),
        "equity_png": str(png_path) if png_path else None,
        "summary": summary,
    }


def main() -> None:
    result = generate_report(Path("outputs"))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
