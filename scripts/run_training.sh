#!/usr/bin/env bash
# Run TC-GraphFormer Multi-Task Training (POSIX Bash)
set -euo pipefail

CONFIG=${1:-"configs/multitask_training.yaml"}
if [ $# -gt 0 ]; then
    shift
fi

PYTHON_BIN=${PYTHON_BIN:-python}
if [ -f ".venv/bin/python" ] && [ "$PYTHON_BIN" = "python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

echo "[TRAIN] Launching TC-GraphFormer Multi-Task Training with config: $CONFIG"
echo "[TRAIN] Using Python: $PYTHON_BIN"
"$PYTHON_BIN" src/training/train_multitask.py --config "$CONFIG" "$@"
