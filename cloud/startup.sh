#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install -e ".[dev,dashboard,live]"
