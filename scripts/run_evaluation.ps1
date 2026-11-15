# Run TC-GraphFormer Baseline Evaluation (PowerShell)
param (
    [string]$Checkpoint = "checkpoints/best_checkpoint.pt",
    [string]$SampleIndex = "reports/pipeline/sample_index.csv",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$PyBin = "python"
if ($env:PYTHON_BIN) {
    $PyBin = $env:PYTHON_BIN
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $PyBin = ".venv\Scripts\python.exe"
}

Write-Host "Evaluating Models & Baselines on Val and Test splits..." -ForegroundColor Cyan
Write-Host "Checkpoint: $Checkpoint | Sample Index: $SampleIndex" -ForegroundColor Gray

& $PyBin src/evaluation/evaluate_baselines.py --checkpoint $Checkpoint --sample-index $SampleIndex @ExtraArgs
