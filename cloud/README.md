# Cloud Environment Startup

Use `cloud/startup.sh` as the cloud agent startup command.

It performs:

1. `cd /workspace`
2. `export PATH="$HOME/.local/bin:$PATH"`
3. `python3 -m pip install -e ".[dev]"`

This ensures Python tooling is present and `pytest` is available on PATH.
