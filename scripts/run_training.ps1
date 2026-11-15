# Run TC-GraphFormer Multi-Task Training (PowerShell)
param (
    [string]$Config = "configs/multitask_training.yaml",
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

Write-Host "Launching TC-GraphFormer Multi-Task Training with config: $Config" -ForegroundColor Cyan
Write-Host "Using Python: $PyBin" -ForegroundColor Gray

& $PyBin src/training/train_multitask.py --config $Config @ExtraArgs
