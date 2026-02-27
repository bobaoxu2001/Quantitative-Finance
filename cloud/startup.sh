#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install -e ".[dev]"
