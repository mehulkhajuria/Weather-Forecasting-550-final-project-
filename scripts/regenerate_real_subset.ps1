# Run from any directory. Configured requests stay bounded and cached.
param([string]$Periods = "scripts/collection_periods.json", [int]$Workers = 4, [switch]$RetryFailed, [switch]$Selected)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$weatherPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$weatherOptions = @('--periods', $Periods, '--workers', $Workers)
if ($RetryFailed) { $weatherOptions += '--retry-failed' }
if ($Selected) { $weatherOptions += '--selected' }
& $weatherPython scripts/build_delivery.py @weatherOptions --fetch
if ($LASTEXITCODE -ne 0) { throw 'Dataset acquisition/assembly failed' }
& $weatherPython scripts/verify_real_alignment.py --dataset data/processed/supervised.npz --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Independent raw alignment verification failed' }
# Refresh the package after the alignment report has been updated; cached, no network.
& $weatherPython scripts/build_delivery.py @weatherOptions
if ($LASTEXITCODE -ne 0) { throw 'Verified handoff packaging failed' }
& $weatherPython scripts/load_dataset.py data/processed/supervised.npz
if ($LASTEXITCODE -ne 0) { throw 'Loading example failed' }
