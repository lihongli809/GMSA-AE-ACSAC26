#!/usr/bin/env bash
# Reproduction entry (frozen snapshot; requires GPU + data). Runs run_repro.py.
set -euo pipefail
SCRIPT_DIR="${0%[/\\]*}"
[ "$SCRIPT_DIR" = "$0" ] && SCRIPT_DIR="."
exec python "$SCRIPT_DIR/run_repro.py" "$@"