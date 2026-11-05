# Run TC-GraphFormer Multi-Task Training (PowerShell)
param (
    [string]$Config = "configs/multitask_training.yaml"
)

$ErrorActionPreference = "Stop"
Write-Host "Launching TC-GraphFormer Multi-Task Training..." -ForegroundColor Cyan

python src/training/train_multitask.py --config $Config
