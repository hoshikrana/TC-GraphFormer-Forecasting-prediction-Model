# Run TC PRIMED Preprocessing Pipeline (PowerShell)
param (
    [string]$Config = "configs/pilot.yaml",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Write-Host "Running TC PRIMED Preprocessing Pipeline..." -ForegroundColor Cyan

$cmd = @("-m", "tcprimed_pipeline", "--config", $Config)
if ($Resume) {
    $cmd += "--resume"
}

python @cmd
