#!/usr/bin/env bash
# Run TC-GraphFormer Baseline Evaluation (POSIX Bash)
set -euo pipefail

CHECKPOINT=${1:-"checkpoints/best_checkpoint.pt"}
SAMPLE_INDEX=${2:-"reports/pipeline/sample_index.csv"}
if [ $# -gt 1 ]; then
    shift 2
elif [ $# -gt 0 ]; then
    shift 1
fi

PYTHON_BIN=${PYTHON_BIN:-python}
if [ -f ".venv/bin/python" ] && [ "$PYTHON_BIN" = "python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

echo "[EVAL] Evaluating Models & Baselines on Val and Test partitions..."
echo "[EVAL] Checkpoint: $CHECKPOINT | Sample Index: $SAMPLE_INDEX"
"$PYTHON_BIN" src/evaluation/evaluate_baselines.py --checkpoint "$CHECKPOINT" --sample-index "$SAMPLE_INDEX" "$@"
