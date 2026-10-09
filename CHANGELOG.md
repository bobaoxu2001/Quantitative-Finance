# Changelog

## Unreleased — full-project review

A review of the whole codebase found correctness bugs that inflated or
distorted backtest results and several live-trading safety issues. Every fix
below is covered by a regression test (`tests/test_*_regressions.py`); the
tests fail on the previous code and pass now.

### Backtest accounting

- **Benchmark was always zero.** The benchmark (SPY) was dropped together with
  non-member symbols, so beta was `NaN` and the information ratio equalled the
  Sharpe ratio. Benchmark returns now come from the unfiltered market.
- **De-risking never traded.** An empty target (risk-off, or no positive
  scores) made the executor return early, so positions were *held* while the
  backtest reported 0% exposure. The executor now liquidates toward an empty
  target, and reported exposure is the realized `invested / equity`.
- **Index exits stranded positions.** Names leaving the membership were removed
  from pricing as well as from the investable universe, so they were marked at
  a price of zero and could never be sold. Membership now only restricts new
  allocations; held names are priced from full market data, with last-close
  marks when a bar is missing.
- **Initial equity point was missing**, dropping the first period's return and
  costs. The equity curve now starts at the initial capital.
- **Lookahead fallback.** When `train_end` preceded the data, the engine
  silently trained on the first 60% of *all* data, overlapping the test window.
  It now raises a clear error.
- **Label leakage.** Training rows whose 5-bar forward label extends past
  `train_end` are purged (`label_end_time`). Date-only window ends now include
  the whole day.
- Membership lookup is computed in one ordered pass instead of per bar, and
  per-bar data is pre-grouped; together with the feature speed-ups below the
  demo backtest runs in ~24s instead of ~66s.

### Portfolio construction, execution and risk

- **Volatility units.** `rv_20h` is a 20-bar volatility (`std * sqrt(20)`) but
  was annualized as if hourly, overstating portfolio risk by ~4.5x and
  throttling exposure. It is now converted to hourly vol and combined with a
  constant-correlation risk model (`assumed_correlation`, default 0.3) instead
  of assuming independent assets.
- **Turnover cap ignored liquidations.** Names held but absent from the new
  target were excluded from the turnover budget (and from reported turnover).
- **Cash could go negative.** With partially filled sells, buys could exceed
  available cash (implicit leverage). Sells now execute first and buys are
  capped at available cash.
- **Risk-off deadlock.** Once in `risk_off` the book was flat, so drawdown
  could never recover and the strategy stayed in cash forever. It now re-arms
  to `conservative` after `risk_off_cooldown_periods` (re-baselining the peak).
  A `conservative` tier (drawdown ≥ 10%) was added to match the documented
  `normal → conservative → risk_off` progression.
- `RiskController.update(benchmark_return=...)` appended to the benchmark
  history without a matching portfolio return, misaligning beta/VaR inputs.

### Metrics

- **Sortino** used the standard deviation of negative returns only; it now uses
  the downside deviation `sqrt(mean(min(r, 0)^2))` over all periods.
- **Annualization** assumed 1,638 bars/year regardless of the calendar (the
  demo has 7 bars/day). Periods per year are inferred from timestamps; CAGR
  uses elapsed calendar time.
- **Win rate / profit factor / holding period** were always `NaN`; they are now
  derived from fills via average-cost accounting.

### Models and features

- **Ensemble scale mismatch.** Route A emits raw returns (~1e-3) and Route B
  emits rank scores (±0.5); averaging them let Route B dominate regardless of
  the learned weights. Sub-models are now blended on a common cross-sectional
  rank scale. The meta-learner split is chronological with an embargo, weight
  normalization is scale-free, and online recalibration is persisted (it
  previously affected only every 20th call).
- **Train/serve skew.** Missing features were imputed with the median of the
  *inference batch*, so a symbol's score depended on which other symbols were
  scored with it. A `FeaturePreprocessor` now stores training medians, means
  and standard deviations (with z-score winsorization).
- Route A/B early-stopping hold-outs are chronological; Route A is seeded and
  resets state on refit; LightGBM `subsample` now takes effect
  (`subsample_freq=1`).
- **Freshness features were broken.** `hours_since_sentiment_update` was always
  0, `analyst_action_recency_hours` always `NaN`, and
  `hours_since_analyst_update` constant, because the as-of join dropped
  same-named columns. They now age with time.
- Downside labels are `NaN` (not `0`) when the forward window is incomplete.
- Cross-sectional ranks are vectorized: feature building is ~25x faster with
  identical values for every other feature.

### Live trading, brokers and governance

- **TCA** measured shortfall against the protective limit price instead of the
  decision-time reference price, reporting fills inside the band as negative
  cost.
- **Binance**: average fill price divided by `max(qty, 1)` (wrong for
  fractional crypto quantities); the replay client signed a sorted query but
  sent it unsorted (invalid signature); replay emitted only the last trade per
  order, losing fills during gap recovery.
- **WebSocket**: stale/duplicate sequence numbers were still delivered; the
  checkpoint is now committed only after successful delivery (at-least-once,
  deduplicated downstream).
- **HTTP OMS**: retries reused a stale signature timestamp; requests are now
  re-signed per attempt.
- **Control plane**: an unlock requestor could approve their own request; a
  maker-checker rule now prevents it. State files are written atomically.
- **Model registry**: artifacts carry a SHA-256 checksum verified before
  unpickling; records/deployment state are written atomically.
- **Event store** leaked SQLite connections; **Redis dedup** used non-atomic
  `SETNX` + `EXPIRE` (now `SET NX EX`).
- **Live loop** used fixed UTC hours (wrong for half the year due to DST); it
  now schedules on the America/New_York session and alerts/marks the hour on
  failure instead of retrying every few seconds.
- **Live runner** valued positions missing from a snapshot at zero and never
  fed realized returns to the risk controller.

### Tooling and docs

- Ruff configuration and a lint-clean codebase; GitHub Actions CI (ruff,
  pytest with coverage on Python 3.11–3.13, demo smoke run).
- README results regenerated with passive benchmarks and an explicit caveat
  that the synthetic demo data has no exploitable alpha by construction; the
  synthetic demo row was removed from the public-fund comparison.
