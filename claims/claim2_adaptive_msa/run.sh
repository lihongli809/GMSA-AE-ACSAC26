#!/usr/bin/env bash
# Canonical-result verification entry (read-only). Runs run.py.
set -euo pipefail
SCRIPT_DIR="${0%[/\\]*}"
[ "$SCRIPT_DIR" = "$0" ] && SCRIPT_DIR="."
exec python "$SCRIPT_DIR/run.py" "$@"