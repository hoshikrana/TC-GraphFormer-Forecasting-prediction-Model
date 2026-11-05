#!/usr/bin/env bash
# Run TC PRIMED Preprocessing Pipeline (POSIX Bash)
set -euo pipefail

CONFIG=${1:-"configs/pilot.yaml"}
RESUME_FLAG=${2:-""}

echo "[PIPELINE] Launching TC PRIMED Preprocessing Engine..."
python -m tcprimed_pipeline --config "$CONFIG" $RESUME_FLAG
