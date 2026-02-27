#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PATH="$HOME/.local/bin:$PATH"

echo "=== Running production preflight ==="
python3 scripts/run_production_preflight.py

echo "=== Running smoke tests (control plane + execution extensions) ==="
python3 -m pytest tests/test_live_control_plane.py tests/test_live_execution_extensions.py tests/test_broker_live_templates.py

echo "Preflight complete."
