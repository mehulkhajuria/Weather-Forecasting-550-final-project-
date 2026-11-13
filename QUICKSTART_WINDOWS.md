# Windows 11: environment, tests, then first GitHub commit

These are the exact first steps. You do not need VS Code or Codex to run the tests. Commands are for **PowerShell** (not Command Prompt, WSL or Python's `>>>` shell).

## A. Extract the zip and open PowerShell in the project folder

1. Download `CS550_Weather_Data_Starter_v0_2_1_RepoReady.zip` from ChatGPT.
2. Open File Explorer, right-click it, choose **Extract All**, and open the resulting **CS550_Weather_Data_Starter** folder.
3. Click the address bar of that folder, type `powershell`, hit Enter. This guarantees commands are run from the project root (where `pyproject.toml` is).
4. Run:

```powershell
Get-Location
Get-ChildItem pyproject.toml
py -0p
```

Your previously reported Windows installation had Python 3.13. Use 3.13 for this initial pipeline. If `py -3.13` cannot be found, install an up-to-date x64 Python via python.org and reopen PowerShell, or use an existing supported `py -3.12` / `py -3.11`.

## B. Isolate dependencies

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

Use `-e` (editable installation) so later changes to `src/` immediately affect the command. The `.[all]` extra includes the GOES NetCDF dependencies and pytest. No PowerShell `Activate.ps1` needed; using `.venv\Scripts\python.exe` avoids execution-policy problems.

## C. Verify tests before any network requests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\weather550.exe demo --out data/demo
.\.venv\Scripts\python.exe -c "import numpy as np; d=np.load('data/demo/SYNTHETIC_numeric.npz'); print('X:',d['X_numerical'].shape,'mask:',d['X_mask'].shape,'features:',d['feature_names'].tolist())"
```

Expected demo shape: `(46, 3, 5, 32, 32)`. The fixture is intentionally fake, clearly marked **SYNTHETIC**, and must not be used for reported experiments.

If a command fails, copy the **first error traceback** and the preceding command. Don't repeatedly reinstall Python globally.

## D. Download a *small* real numerical sample

```powershell
.\.venv\Scripts\weather550.exe fetch-asos --stations TTN EWR --start 2026-10-20T00:00:00Z --end 2026-10-22T00:00:00Z
.\.venv\Scripts\weather550.exe prepare-numeric --stations data/processed/stations_utc.csv --start 2026-10-20T00:00:00Z --end 2026-10-22T00:00:00Z
```

Inspect `data/processed/stations.summary.json` and `data/processed/model_numeric.json`. The latter shows number of issuance windows, valid fraction and mask shape. Confirm it contains actual station records. If IEM has no data, there is **no hidden fallback to synthetic values**.

## E. Check a single GOES scene (optional next step)

```powershell
.\.venv\Scripts\weather550.exe fetch-goes --at 2026-10-20T18:00:00Z --list-only
.\.venv\Scripts\weather550.exe fetch-goes --at 2026-10-20T18:00:00Z
```

Expect a public NetCDF download and a model-grid reprojected `.npz`. This step may take longer or fail for network or source-data reasons; it is isolated from station processing.

## F. Create a clean LOCAL Git history

```powershell
git --version
git init -b main
git status
git add .gitignore README.md QUICKSTART_WINDOWS.md pyproject.toml src tests docs scripts data/raw/.gitkeep data/processed/.gitkeep
git commit -m "Bootstrap CS550 weather data ingestion and tensor contract"
git status
```

If Git is missing, install Git for Windows from https://git-scm.com/downloads/win, then reopen PowerShell. If Git asks for your identity, set it in your local repository:

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR GITHUB EMAIL"
```

Run the commit again. Don't commit `data/raw/`, `data/processed/`, `.venv/`, `.env` or API tokens; `.gitignore` is already configured.

## G. Publish to GitHub, **only after coordinating with your teammates**

**If the team already has a GitHub repository:** use their shared repo, not a parallel official team repo. Clone it and bring the starter files in via a feature branch/PR after checking for conflicts. The local bootstrap can still be used to test the package independently.

**If nobody has created the shared repository:** install GitHub CLI (optional) and create one privately:

```powershell
winget install --id GitHub.cli -e
# Reopen PowerShell after installation
gh auth login
gh repo create cs550-weather-fusion --private --source . --remote origin --push
```

The command creates the repository **under the GitHub account you're logged into**. Your teammates can then be granted access in GitHub Repository Settings > Collaborators. Don't enable public visibility until everyone agrees about course rules and code attribution.

## H. The one thing to send your teammate

Send the schema from `docs/DATA_CONTRACT.md` and the first real `.npz` through an approved data-sharing method (it is ignored by Git). Ask them to accept `X_numerical`, `X_mask`, feature names, timestamps and lat/lon grid. **There is no precipitation `y` yet.** The authoritative +1h/+3h targets will be added in a future version.

## Recovery / troubleshooting

| Symptom | Check |
|---|---|
| `py -3.13` not recognized | `py -0p`, choose 3.11+ or install Python x64 |
| `No module named weather550` | Verify you're in the folder with `pyproject.toml`; rerun `-m pip install -e ".[all]"` |
| Execution policy error | Don't use `Activate.ps1`; call `.venv\Scripts\python.exe` directly |
| Can't find `weather550.exe` | Use `.venv\Scripts\python.exe -m weather550.cli demo` |
| `403`/network timeout | Source/network issue; offline demo still works; don't change the dataset dates silently |
| `No valid ASOS station records` | Inspect raw file or try one station/time window; don't fabricate missing data |
| `GOES scene ... not found` | Check UTC date/year and product; use `--list-only` before download |
| `h5netcdf` missing | `.venv\Scripts\python.exe -m pip install -e ".[all]"` |
| `git commit` identity error | Set repository-local `git config user.name/email` as above |

## Completion gate for this session

1. Unit tests pass.
2. Offline demo tensor shape is verified.
3. `git status` is clean after the initial commit.
4. For a first real-data pass, confirm a nonempty IEM CSV and matching sample tensor metadata.
5. **Do not begin training the team models until actual precipitation targets and split design are agreed.**


## Linux/HPC note

For Rutgers iLab or Amarel, follow [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) and `scripts/setup_linux.sh`, not the Windows commands.
