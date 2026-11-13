# Run from repository root. Safe, repeatable. Does not write to GitHub.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path 'pyproject.toml')) { throw 'Open PowerShell in the project root (where pyproject.toml exists).' }
$pythonVersion = $null
foreach ($v in @('3.13','3.12','3.11')) {
    & py "-$v" --version 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $pythonVersion=$v; break }
}
if ($null -eq $pythonVersion) { throw 'Install Python 3.11+ x64. Run py -0p to see available versions.' }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & py "-$pythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& .\.venv\Scripts\python.exe -m pip install -e ".[all]"
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency integrity check failed' }
& .\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 .venv\requirements-installed.txt
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed' }
& .\.venv\Scripts\weather550.exe demo --out data/demo
if ($LASTEXITCODE -ne 0) { throw 'Offline demo failed' }
Write-Host 'Environment ready. Tests and offline synthetic demo passed.'
