"""Airflow DAG template for live trading operations."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "quant-platform",
    "depends_on_past": False,
    "retries": 1,
}


with DAG(
    dag_id="hourly_live_trading",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 * * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["live", "trading", "hourly"],
) as dag:
    run_live_cycle = BashOperator(
        task_id="run_live_cycle",
        bash_command=(
            "cd /workspace && "
            "export PATH=$HOME/.local/bin:$PATH && "
            "python3 scripts/run_live_from_config_template.py"
        ),
    )

    run_ws_listener_healthcheck = BashOperator(
        task_id="ws_listener_healthcheck",
        bash_command=(
            "cd /workspace && "
            "python3 -c \"from hourly_trading_system.dashboard import "
            "load_live_queue_snapshot,summarize_live_snapshot; "
            "s=load_live_queue_snapshot('outputs/live_queue'); print(summarize_live_snapshot(s))\""
        ),
    )

    run_live_cycle >> run_ws_listener_healthcheck
