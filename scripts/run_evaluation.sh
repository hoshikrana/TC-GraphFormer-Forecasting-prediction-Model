#!/usr/bin/env bash
# Run TC-GraphFormer Baseline Evaluation (POSIX Bash)
set -euo pipefail

echo "[EVAL] Evaluating Models & Baselines on Val and Test partitions..."
python src/evaluation/evaluate_baselines.py

