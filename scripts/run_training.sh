#!/usr/bin/env bash
# Run TC-GraphFormer Multi-Task Training (POSIX Bash)
set -euo pipefail

CONFIG=${1:-"configs/multitask_training.yaml"}

echo "[TRAIN] Launching TC-GraphFormer Multi-Task Training..."
python src/training/train_multitask.py --config "$CONFIG"
