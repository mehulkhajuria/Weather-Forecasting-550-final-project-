> Updated November 13, 2026: [the verified real-data handoff](docs/REAL_DATA_HANDOFF.md) has actual results and current commands. Statements below about work not yet demonstrated describe the original ZIP before local verification.

# CS550 v0.3 — Windows-first: complete the data-engineering contribution

**What is already done**: 22 offline tests, GOES Band 13 downloader/reprojector, IEM ASOS numerical-grid prep, public-archive MRMS QPE 01H fetcher and GRIB2 decoder, strict causal supervised join, and chronological split embargo option.

**What is NOT yet demonstrated**: a real Windows MRMS ecCodes decode or a real NOAA+IEM+MRMS joined training sample. Those need the commands below. No claim of real training performance is made.

## Step 1: download and extract the v0.3 ZIP

Extract the folder `CS550_Weather_Data_v0_3_Class_Dataset` into a convenient Windows folder such as `Documents\\Rutgers\\CS550` (NOT inside `.venv`). Open PowerShell IN that extracted folder.

```powershell
py -0p
py -3.13 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -e ".[all]"
.\\.venv\\Scripts\\python.exe -m pip check
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\weather550.exe demo --out data/demo
```

`pip install -e ".[all]"` includes `eccodes` for GRIB2 (published Windows 3.13 wheel). If it fails, preserve the error and stop: do not remove the radar dependency and falsely treat tests as a live-data pass.

## Step 2: publish to your currently empty GitHub repository

```powershell
git init -b main
git add .
git status
git commit -m "Add tested weather550 data pipeline v0.3"
git remote add origin https://github.com/mehulkhajuria/Weather-Forecasting-550-final-project-.git
git push -u origin main
```

If you already initialized a repo, inspect `git remote -v` and omit `git init` / `git remote add`. Commit only code/docs/tests. `.gitignore` excludes `.venv` and all downloaded weather files. If `git push` rejects because the remote gained commits, STOP and compare histories rather than force-pushing.

## Step 3: first reproducible real-data experiment — ONE issuance

Time `2026-10-20T18:00:00Z` is a **test time**, not a verified storm event. NOAA scene and MRMS source metadata must actually be available for this date before any claim of a real-data pass.

```powershell
# Check GOES availability and the two historical radar URLs without downloading radar
.\\.venv\\Scripts\\weather550.exe fetch-sample --at 2026-10-20T18:00:00Z --list-only

# Download two full days of IEM ASOS/METAR (date-level API requires midnight boundaries)
.\\.venv\\Scripts\\weather550.exe fetch-asos --stations TTN EWR --start 2026-10-20T00:00:00Z --end 2026-10-22T00:00:00Z

# Build 32x32 numerical grids with past-only observations; this includes issuance 18:00
.\\.venv\\Scripts\\weather550.exe prepare-numeric --stations data/processed/stations_utc.csv --start 2026-10-20T15:00:00Z --end 2026-10-20T20:00:00Z

# Real GOES C13 scans (16,17,18 UTC) plus MRMS QPE 1-hour targets ending 19 and 21 UTC
.\\.venv\\Scripts\\weather550.exe fetch-sample --at 2026-10-20T18:00:00Z

# Join strict, matching coordinates, target horizons and masks
.\\.venv\\Scripts\\weather550.exe build-supervised --numeric data/processed/model_numeric.npz --out data/processed/supervised.npz
```

If the join says there are no complete samples, inspect `data/processed/goes/*.json`, `data/processed/mrms/*.json`, station summary, and the error details. Do not silently skip all gaps or invent data. Note that exact source URLs and raw filenames will be printed and preserved.

## Step 4: bounded expansion, after the single case works

```powershell
# A maximum of 12 hourly issuances from historical files (safety cap)
.\\.venv\\Scripts\\weather550.exe fetch-batch --start 2026-10-20T00:00:00Z --end 2026-10-21T00:00:00Z --max-issuances 12 --continue-on-error

# Report each completed/missing source and cache failures
Get-Content data/processed/fetch_audit.json
```

Also increase ASOS/numeric window to cover corresponding issuance times and past history. For more than 8 days, IEM downloader requires separate chunks; the starter deliberately enforces a small-request cap. For a final report, document *actual* date coverage, rainy cases and the percentage of missing data.

## Team model contract

`data/processed/supervised.npz` contains:

- `X_goes`: (N, 3, 1, H, W), C13 brightness temperatures in Kelvin (unscaled)
- `X_goes_mask`: identical shape, Boolean pixel validity
- `X_numerical`: (N, 3, F, H, W), sparse ASOS readings mapped by inverse-distance weights; NOT true full-resolution analyses
- `X_num_mask`: identical numerical shape, Boolean validity
- `Y_rain_mm`: (N, 2, H, W), MRMS rainfall total in **mm** for hours ending +1h and +3h
- `Y_rain_mask`: identical rain shape, Boolean validity
- `issuance_times_utc`, `forecast_horizons_hours` = [1,3], `feature_names`, `lat_grid`, `lon_grid`

No normalization applied. **Fit normalizers only on chronological training data.** To pre-label train/val/test, pass `--train-before <UTC>` and `--val-before <UTC>` to `build-supervised`. The `split` array includes `exclude` windows touching split boundaries. Stage-1 numerical model overlays for stage-2 training MUST be generated with forward-chaining out-of-fold stage-1 models. This data package does not train those models.

## Assignment scope and TA requirements

- Frozen proposed source plan: NJ + surroundings bbox `[39.2,-75.3,41.2,-73.0]`, GOES-19 ABI-L2-CMIPC C13, IEM ASOS (TTN/EWR initially), IEM MRMS MultiSensor_QPE_01H_Pass2, preliminary August–October 2026 study.
- Output: future **hourly accumulation** ending t+1h and t+3h, not accumulated 3h rainfall and not radar VIL.
- Public data does not reliably include historical solar irradiance from these ASOS stations: missing explicitly, not fabricated.
- The classroom model comparison is persistence, satellite-only, direct-numeric concatenation, and numerical-predicted overlay fusion, with identical chronological splits and target horizons.
- Actual matched source coverage and units are **not yet verified** until Step 3 succeeds. If MRMS archive is unavailable, request a documented adjustment/SEVIR fallback from the course team, rather than silently changing the research target.

## Source URLs

- GOES NODD: https://registry.opendata.aws/noaa-goes/
- IEM ASOS: https://mesonet.agron.iastate.edu/request/download.phtml
- IEM MRMS archive sample: https://mtarchive.geol.iastate.edu/2026/06/09/mrms/ncep/MultiSensor_QPE_01H_Pass2/
- ECMWF ecCodes Python: https://pypi.org/project/eccodes/

**Known limitation:** full CONUS GRIB latitude/longitude decoding can use >1 GB RAM per file, and full archive downloads can be significant. This design prioritizes correct spatial alignment and bounded sample download over speed until a real live decode has been verified.
