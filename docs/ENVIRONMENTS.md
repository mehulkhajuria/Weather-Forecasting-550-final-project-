# CS550 weather pipeline — environment setup across Windows, iLab and Amarel

## Key rule
Commit code, `pyproject.toml`, scripts and tests; never commit `.venv/`, downloaded datasets, credentials, or local wheels. **Each machine must create its own `.venv`**, since Windows, Ubuntu, and RHEL use different compiled binaries. Installed version snapshots are written to `.venv/requirements-installed.txt` after setup and are local diagnostic records (not a cross-platform lockfile).

## 1. Windows 11 (primary development)

Open PowerShell in the unzipped project directory (where `pyproject.toml` lives).

```powershell
py -0p
# No activation or PowerShell execution-policy change required
& .\scripts\setup_windows.ps1
# If PowerShell disallows scripts, use the commands from QUICKSTART_WINDOWS.md.
& .\scripts\verify_windows.ps1
```

If script execution policy blocks the `.ps1`, follow the direct commands from `QUICKSTART_WINDOWS.md` (which use `.venv\Scripts\python.exe` directly) rather than changing system-wide execution policy. Setup uses the newest installed Python among 3.13, 3.12, and 3.11. The project declares Python >=3.11.

## 2. Rutgers iLab (Linux reproducibility)

Rutgers iLab machines use Ubuntu 24.04. Connect through SSH or WebLogin. Choose a specific `ilab1`/`ilab2` host for editor Remote SSH; `ilab.cs.rutgers.edu` can move you between hosts. Use Git to clone the project after creating a remote repository; do not copy `.venv` from Windows.

```bash
ssh YOUR_NETID@ilab.cs.rutgers.edu
# Once connected, clone only after your GitHub repo has been created:
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
# iLab has Python 3.13 provided by Rutgers (Aug 2026); create isolated project venv:
PYTHON_BIN=/usr/bin/python3.13 bash scripts/setup_linux.sh
bash scripts/verify_linux.sh
```

If `/usr/bin/python3.13` does not exist on the selected host, use `python3 --version` and `ls /usr/bin/python3.*`, then set `PYTHON_BIN` to an available supported Python. Avoid installing system packages with `sudo`. Rutgers documents an optional preset interpreter environment `/common/system/venv/python313/bin/python` if you need a fallback, but do not share its site-packages with the project.

On iLab/rLab, GPU work requires the Rutgers SLURM scheduler; the above test is CPU-only. Respect shared-machine resource limits. iLab storage is separate from Amarel.

## 3. Rutgers Amarel (later; Slurm processing and training)

Rutgers' August 2026 migration maps `amarel.rutgers.edu` to RHEL9 login nodes. Connect from the Rutgers network or Rutgers VPN. Amarel is independent of iLab.

```bash
# From PowerShell on Windows *after connecting to Rutgers VPN*:
ssh YOUR_NETID@amarel.rutgers.edu
# On the Amarel login node:
module spider python
python3 --version
# If Python <3.11, load an *actually listed* compatible module, then inspect python3 --version.
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
PYTHON_BIN=python3 bash scripts/setup_linux.sh --install-only
sbatch scripts/amarel_verify.sbatch
squeue -u "$USER"
```

`PYTHON_BIN=python3` is an example, not a claim that the cluster default is new enough. If module environment is required, `module load ...` using the **actual** listed name/version before creating the environment. The same module may also need loading in your batch script. Do not run pytest, downloads, numerical computation, or model training on an Amarel login node. Install/inspect there as permitted; run tests on a scheduled compute node. A SLURM account or partition may need to be supplied depending on your allocation. The provided job file omits both instead of guessing.

## Why local → iLab → Amarel?

- Windows is fastest for coding/debugging and convenient for data downloaded to your workstation.
- iLab independently validates that code works under Ubuntu, useful for class collaborators.
- Amarel provides scheduled compute resources for heavier datasets and training. It doesn't require iLab to access it.

## Dependency troubleshooting

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze
```

On Windows the equivalent begins `.\ .venv` with no space: `.\.venv\Scripts\python.exe -m pip check`.

After verifying your primary Windows install and an iLab install, preserve `pyproject.toml` and compare `.venv/requirements-installed.txt` snapshots. If we find a version conflict, tighten the dependency ranges or generate **separate platform-specific lock files** from verified environments; don't pin arbitrary untested versions. Nothing here installs PyTorch/CUDA or a global Python environment. That's a separate stage when the model is ready.

## GitHub and data

Git synchronizes code, not huge raw satellite/radar files. Keep raw downloads in `data/raw` (ignored), processed files in `data/processed` (ignored), and published sample manifests/checksums in `docs` or a designated manifests folder. Never copy `.venv` or use it as a backup. Use approved transfers (`scp`/SFTP or object storage) for large data when moving to Amarel.
