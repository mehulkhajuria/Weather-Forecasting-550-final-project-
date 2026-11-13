$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv\Scripts\python.exe')) { throw 'Run setup_windows.ps1 or see QUICKSTART_WINDOWS.md first.' }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& .\.venv\Scripts\weather550.exe demo --out data/demo
if ($LASTEXITCODE -ne 0) { throw 'Demo failed' }
& .\.venv\Scripts\python.exe -c "import numpy as np; d=np.load('data/demo/SYNTHETIC_numeric.npz'); print('X shape',d['X_numerical'].shape,'mask shape',d['X_mask'].shape)"
Write-Host 'Local smoke check complete.'
