# Run from any directory; each source request and batch is bounded.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$weatherPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
function Invoke-Weather {
    & $weatherPython -m weather550.cli @args
    if ($LASTEXITCODE -ne 0) { throw "weather550 failed (exit $LASTEXITCODE)" }
}
Invoke-Weather fetch-sample --at 2026-10-20T18:00:00Z --list-only
Invoke-Weather fetch-asos --stations TTN EWR --start 2026-10-20T00:00:00Z --end 2026-10-22T00:00:00Z
Invoke-Weather fetch-asos --stations TTN EWR --start 2026-10-10T00:00:00Z --end 2026-10-18T00:00:00Z --raw data/raw/iem_asos_oct10_18.csv --out data/processed/stations_oct10_18.csv
Invoke-Weather fetch-sample --at 2026-10-20T18:00:00Z
Invoke-Weather fetch-batch --start 2026-10-11T14:00:00Z --end 2026-10-11T20:00:00Z --max-issuances 6 --audit data/processed/fetch_rainy_audit.json
Invoke-Weather fetch-batch --start 2026-10-20T15:00:00Z --end 2026-10-20T21:00:00Z --max-issuances 6 --audit data/processed/fetch_dry_audit.json
& $weatherPython scripts/build_delivery.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset assembly/validation failed' }
& $weatherPython scripts/verify_real_alignment.py
if ($LASTEXITCODE -ne 0) { throw 'Independent raw alignment verification failed' }
& $weatherPython scripts/load_dataset.py data/processed/supervised.npz
if ($LASTEXITCODE -ne 0) { throw 'Loading example failed' }
