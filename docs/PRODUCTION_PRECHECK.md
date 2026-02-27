# Production Preflight Checklist

Run this checklist before enabling live capital.

## A. Environment

- [ ] Python and package environment installed (`python3`, `pip`, editable package).
- [ ] `pytest` passes in deployment image.
- [ ] `cloud/startup.sh` or equivalent bootstrap configured.

## B. Data & Queue

- [ ] Live queue path exists and is writable.
- [ ] Queue topics receiving updates (`orders`, `order_updates`, `fills`, `reconciliation`).
- [ ] Sequence state path writable for WS resume.
- [ ] Replay endpoint configured for WS gap recovery (if enabled).

## C. Model Governance

- [ ] Model registry path exists.
- [ ] Champion model deployed.
- [ ] Canary config reviewed (allocation + promotion thresholds).
- [ ] Rollback path validated in staging.

## D. Broker Connectivity

- [ ] OMS health endpoint reachable.
- [ ] Auth/signature credentials loaded from secret manager.
- [ ] Idempotency behavior validated under retry.
- [ ] REST replay endpoint tested for missing-sequence recovery.

## E. Risk & Control Plane

- [ ] Guardrails configured (max notional, rejection streak, intraday loss kill-switch).
- [ ] Control plane state path writable.
- [ ] RBAC permissions configured.
- [ ] Unlock flow tested with multi-approval requirement.

## F. Monitoring & Alerting

- [ ] Live monitor reachable.
- [ ] Alerts routed to file + on-call channel.
- [ ] Health score, TCA, reconciliation metrics visible.
- [ ] Kill-switch and rollback buttons validated in staging.

## G. Final Go/No-Go

- [ ] Preflight script returns PASS (no critical failures).
- [ ] Incident runbook reviewed by Ops + Risk.
- [ ] Approval captured from risk owner.
