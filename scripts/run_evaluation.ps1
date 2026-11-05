# Run TC-GraphFormer Baseline Evaluation (PowerShell)
param (
    [string]$Checkpoint = "checkpoints/best_checkpoint.pt",
    [string]$SampleIndex = "reports/pipeline/sample_index.csv"
)

$ErrorActionPreference = "Stop"
Write-Host "Evaluating Models & Baselines on Val and Test splits..." -ForegroundColor Cyan

python src/evaluation/evaluate_baselines.py

