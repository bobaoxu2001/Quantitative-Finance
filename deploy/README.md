# Production Deployment Templates

This folder contains deployment templates for running the live trading stack in
three common production environments:

- Kubernetes
- Airflow scheduler
- systemd (single-host services)

## 1) Kubernetes

Files under `deploy/kubernetes/` provide:

- `configmap.yaml`: runtime environment flags and paths
- `secret.template.yaml`: secret placeholders for broker credentials
- `live-monitor-deployment.yaml`: Streamlit monitor deployment
- `hourly-live-run-cronjob.yaml`: hourly strategy loop trigger

### Notes
- Replace all `<REPLACE_...>` placeholders.
- Mount `/workspace` or build an image that includes this repository.
- Bind persistent volume for `outputs/` to retain queue and audit state.

## 2) Airflow

`deploy/airflow/dag_live_trading.py` includes:

- hourly live cycle task
- websocket listener task
- optional health summary task

Use environment variables or Airflow Connections for broker credentials.

## 3) systemd

`deploy/systemd/` includes:

- `hourly-trading.service` + `hourly-trading.timer` for hourly cycle
- `broker-ws-listener.service` for continuous websocket ingestion

Install (example):

```bash
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now broker-ws-listener.service
sudo systemctl enable --now hourly-trading.timer
```

## Runtime commands referenced

- Build runner from config:  
  `python3 scripts/run_live_from_config_template.py`
- Websocket listener template:  
  `python3 scripts/run_broker_ws_listener_template.py`
- Live monitor:  
  `streamlit run scripts/run_live_monitor.py`

## Operational recommendations

- Store credentials in secret manager, not in plain files.
- Enable centralized logs (CloudWatch/ELK/Datadog).
- Alert on:
  - kill-switch engaged
  - reconciliation breaks
  - order rejection streak
  - queue stagnation (no order_updates/fills within expected window)
