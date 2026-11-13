# Update from v0.2 to v0.2.1

- Added `scipy>=1.15,<2` to the satellite and all extras: `test_reproject_synthetic_netcdf` explicitly uses `engine="scipy"`.
- GitHub Actions verifies the test suite on Windows and Ubuntu (Python 3.13).
- No changes to dataset processing behavior or the 15 tests.

**Existing installation (no need to recreate `.venv`):** replace your existing `pyproject.toml` with this version, then run `\.venv\Scripts\python.exe -m pip install -e ".[all]"` (Windows PowerShell, with prefix `.` before the backslash) and `\.venv\Scripts\python.exe -m pytest -q`. The old `DeprecationWarning` messages may still appear and will be addressed separately.

**Git:** avoid committing `.venv`, raw data, `*.npz`, credentials; use the included `.gitignore`. Do not force-push over a teammate's branch.
