# Verified data handoff — November 13, 2026

`data/processed/supervised.npz` contains **12 real synchronized samples: six rainy and six dry**. All 12 selected issuances passed; none were rejected. Downloads and derived arrays are ignored by Git. The small verification manifests, code and regeneration scripts are committed.

Validation on Windows/Python 3.13: the original 22 tests passed; the expanded **37-test suite passed with warnings treated as errors**, and `pip check` reported no broken requirements. The regeneration script completed successfully using cached raw sources. Independent checks passed at 48 MRMS and 48 GOES grid locations.

## Actual coverage

| Component | Verified result |
|---|---|
| Satellite | GOES-19 ABI-L2-CMIPC C13, brightness temperature K; 16 NetCDFs |
| Numerics | IEM ASOS/METAR TTN and EWR; 556 parsed station records |
| Targets | MRMS MultiSensor_QPE_01H_Pass2; 16 hourly GRIB2 files |
| Region | `[39.2,-75.3,41.2,-73.0]` south/west/north/east; north-first 32×32 grid |
| Rainy issuances | October 11, 2026, 14:00–19:00 UTC, hourly |
| Dry issuances | October 20, 2026, 15:00–20:00 UTC, hourly |
| Rain labels | October 11 15:00–22:00 UTC; October 20 16:00–23:00 UTC |
| Station coverage | October 10–17 and October 20–21, 2026 |
| Raw missingness | Pressure missing in 76/556 records; other four features complete |
| Tensor validity | GOES 100%; rain 100%; numerical 76.813% |
| Numerical feature validity | First four features 80.273% each; pressure 62.972% |
| Physical ranges | GOES 220.371–299.731 K; rain 0–26.7 mm |

Rainy means at least one valid target pixel exceeds zero at either horizon. Dry requires every target cell valid and zero. A 110-km station-support radius leaves some pixels unsupported. Missing observations are never fabricated. The initial one-case join retained one of two numerical candidates; its other candidate lacked satellite history. The final subset selected exactly 12 candidates and rejected none.

**Only these matched periods are verified.** August–October remains a candidate study period, not an exhaustively checked archive. This integration subset contains one main rain event and is insufficient for model evaluation or held-out-storm claims.

## NPZ interface

Load with `np.load(path, allow_pickle=False)`.

| Key | Shape / definition |
|---|---|
| `X_goes` | `(12,3,1,32,32)` float32, C13 K |
| `X_goes_mask` | Same shape, bool |
| `X_numerical` | `(12,3,5,32,32)` float32, physical station-conditioned features |
| `X_num_mask` | Same shape, bool |
| `Y_rain_mm` | `(12,2,32,32)` float32, hourly accumulation mm |
| `Y_rain_mask` | Same shape, bool |
| `lat_grid`, `lon_grid` | `(32,32)`, degrees north/east, first row north |
| `issuance_times_utc` | Chronological UTC strings |
| `forecast_horizons_hours` | `[1,3]` |
| `feature_names` | temperature_c, dewpoint_c, humidity_pct, wind_speed_mps, pressure_hpa |
| `goes_scan_times_utc`, `goes_available_times_utc` | `(12,3)` source scan starts and S3 LastModified |
| `latest_station_observation_times_utc` | `(12,3)` latest selected station timestamps |
| `rain_valid_end_times_utc` | `(12,2)` exact label end times |
| `source_metadata_json` | Per-sample URLs, filenames and timestamps as a JSON string |
| `source_station_files` | Two station CSV paths |
| `observation_latency_minutes` | 10, an explicit assumption |
| `max_observation_age_minutes` | 90 |

History ticks are `[t−120min,t−60min,t]`. The +1h target covers `(t,t+1h]`; the +3h target covers `(t+2h,t+3h]`. **The second target is not a three-hour total.** Numerical grids are inverse-distance-weighted station proxies, not NWP analyses. Invalid cells are zero placeholders with masks false; use mask-aware processing. No normalization or model training is performed.

## Availability and scientific limits

GOES scans must be published by each history tick and no more than 20 minutes old. The downloader uses S3 LastModified, checks NetCDF versus filename time, preserves coverage end, and requires all bilinear contributors to have DQF=0.

**IEM lacks historical receipt timestamps in this response.** Only observations at least 10 minutes before each input tick are used, with a 90-minute age limit. This is a disclosed retrospective reporting-latency assumption, not proof of actual receipt by that time. Strict operational availability claims require receipt-time archives or exclusion of uncertain reports.

MRMS Pass2 is delayed quality-controlled data used only as a target. ecCodes reports NOAA local units as `unknown`. The decoder validates centre 161 / discipline 209 / category 6 / parameter 37 and derives mm and one-hour duration from the [NOAA MRMS operational table](https://www.nssl.noaa.gov/projects/mrms/operational/tables.php). Other products, rates and mismatched valid times are rejected. Gzip bytes are decompressed before decoding; grid endpoints, increments and scanning flags are validated.

Independent raw checks compare processed MRMS values with ecCodes geographic nearest neighbors. Half-cell ties at bbox edges allow a 1-metre distance tolerance for float32 exported coordinates. GOES checks use the raw CF projection and four source pixels. Evidence and raw-file SHA-256 hashes are in `RAW_ALIGNMENT_VERIFICATION.json` and `REAL_DATA_VERIFICATION.json`.

## Exact regeneration commands

Run in the inner extracted project directory containing `pyproject.toml`. On a new Windows Python 3.13 machine:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py313.txt
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -W error
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/regenerate_real_subset.ps1
```

The lock file records verified Windows Python 3.13 dependencies; use declared extras on other platforms. The regeneration script contains the exact CLI commands: initial listing, two bounded station requests, one complete test case, then two six-issuance batches. It builds the joined dataset, verifies raw geography/values and runs the loading example. It stops on errors. Cached ASOS requests are reused only when saved station/date request metadata match and parsed timestamps are valid. Fresh retrievals remain subject to provider rate limits.

To rebuild from existing files without network access:

```powershell
.\.venv\Scripts\python.exe scripts/build_delivery.py
.\.venv\Scripts\python.exe scripts/verify_real_alignment.py
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

The delivery builder intentionally fixes the two verified integration periods and asserts 12 complete samples. Use the general CLI for further bounded events.

## Teammate usage and evaluation

```python
import numpy as np
with np.load('data/processed/supervised.npz', allow_pickle=False) as d:
    X = d['X_numerical']
    valid = d['X_num_mask']
    y = d['Y_rain_mm']
    y_valid = d['Y_rain_mask']
    times = d['issuance_times_utc']
```

The runnable example is `scripts/load_dataset.py`. The ignored `data/processed/teammate_handoff.zip` includes the NPZ, companion JSON, this contract and loading example.

Professor Tang's satellite-only, direct numerical concatenation and learned precipitation-overlay comparisons must use common targets and chronological splits. Stage-2 training overlays must come from **forward-chaining out-of-fold** numerical predictions. Fit scalers and imputation only within training folds. Add independent storms before evaluation and hold out whole storms where feasible. Do not randomly split overlapping sequences. CLI split boundaries embargo the entire target interval and history, including the 90-minute station lookback. This tiny subset is left unsplit.

No SEVIR substitution, VIL target or model training was introduced.

## Errors fixed and recovery

- Original 22 tests passed with timestamp deprecation warnings. Standard-library timedeltas fix these without suppressing warnings.
- Added h5py to the declared HDF5 satellite dependencies.
- Replaced future/nearest GOES selection with publication-aware past selection.
- Fixed Windows MRMS `No GRIB message in MRMS file` by decoding decompressed bytes; validated the NOAA local parameter identity.
- A repeated ASOS request returned HTTP 429. Validated cached data were preserved and reused. On a fresh machine, stop and retry after the provider's requested delay; never substitute fabricated values.

Remaining research work: broader event coverage and confirmation of the availability assumption, followed by teammate model integration.
