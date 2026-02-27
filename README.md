# Institutional Hourly Trading System (Production Blueprint)

This repository contains a production-grade blueprint implementation of an
**hourly long-only trading system** on **dynamic S&P 500 membership** with
strict anti-lookahead controls, cost-aware execution simulation, portfolio
optimization constraints, and analytics/dashboard outputs.

## What is included

- Point-in-time data contracts (`event_time`, `public_release_time`,
  `ingest_time`, `known_time`) and anti-lookahead enforcement.
- Feature engineering library (50+ features across price, volatility,
  cross-sectional, fundamentals, macro, sentiment, analyst, and freshness).
- Label engineering for:
  - 5-hour forward excess return vs SPY
  - downside event probability
- Dual model routes:
  - Route A: deep cross-sectional multitask model (Torch optional, deterministic fallback)
  - Route B: factor + LightGBM-style baseline (LightGBM optional, deterministic fallback)
- Long-only constrained portfolio construction:
  - max 10% per stock
  - liquidity filters
  - soft sector controls
  - target volatility band
  - turnover cap
- Event-driven hourly backtest engine:
  - signal at hour *t*
  - execution at hour *t+1 open*
  - slippage + spread + square-root impact
  - partial fill simulation
  - risk-state drawdown governance
- Metrics engine with required institutional KPIs.
- Dashboard blueprint payload builders and Streamlit starter script.
- Pytest test suite covering anti-lookahead, metrics, and end-to-end flow.

## Directory layout

```text
src/hourly_trading_system/
  data/           # PIT contracts, adapters, and store logic
  features/       # feature library
  labels/         # label engineering
  models/         # Route A + Route B models
  portfolio/      # allocation and constraints
  execution/      # transaction cost + fills
  risk/           # drawdown and risk regime controls
  backtest/       # event-driven simulation engine
  analytics/      # performance metrics
  dashboard/      # panel blueprint
  orchestration/  # research/paper/live pipeline wrapper
scripts/
  run_demo_backtest.py
  run_dashboard.py
tests/
```

## Quickstart

### 1) Install

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[dashboard,ml]"
```

### 2) Run end-to-end demo

```bash
python scripts/run_demo_backtest.py
```

This generates output files under `outputs/`:

- `equity_curve.csv`
- `fills.csv`
- `weights.csv`
- `risk_history.csv`
- `monthly_returns.csv`

### 3) Run tests

```bash
pytest
```

### 4) Dashboard (optional)

```bash
streamlit run scripts/run_dashboard.py
```

## Production assumptions and controls

- Long-only, no leverage, cash allowed.
- Position cap: 10% per stock.
- Signal at `t`, execute at `t+1 open`.
- Cost model:
  - slippage: 1.0 bps
  - half-spread: 0.5 bps
  - impact: `eta * sqrt(participation)`
- Dynamic universe handled through point-in-time membership table.
- Hard drawdown governance via risk state transitions:
  - normal → conservative → risk_off.

## Important note

This codebase is a production **blueprint + executable framework**.
For live deployment, plug in institution-approved data vendors,
OMS/EMS integration, compliance controls, and hardened infrastructure
(secrets management, authZ, full observability, and disaster recovery).
