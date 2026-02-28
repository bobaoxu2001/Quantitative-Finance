#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:-/workspace}"
cd "$WORKDIR"

# Ensure user-level Python bin is available for pytest and helper tools.
export PATH="$HOME/.local/bin:$PATH"

python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,dashboard,live]"

echo "Environment bootstrap complete."
echo "python3: $(python3 --version)"
echo "pip: $(python3 -m pip --version)"
echo "pytest path: $(command -v pytest || true)"
