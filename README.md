# CS550 Weather Forecasting

This repository handles the data side of our CS550 project, *Fourier-Based Weather Prediction by Combining Satellite Imagery and Numerical Observations*. It collects satellite imagery, station observations, and precipitation measurements, then processes them into matching grids and time windows for the models. Keeping this separate from the model lets us try the satellite-only, direct numerical-feature, and precipitation-overlay approaches using the same inputs and targets.

Right now we have 12 real samples that have been downloaded, aligned, and verified: six rainy samples from October 11, 2026, and six dry samples from October 20, 2026. These are enough to start checking model inputs and integrating the models. They are still a pilot dataset, so we need more independent weather events before using this for training and evaluation. The data pipeline already downloads and decodes all three sources, builds the numerical grids, and joins the satellite history with future precipitation targets. Model training is separate.

## Data and preprocessing

| Source | What we use |
|---|---|
| [NOAA GOES-19 public archive](https://registry.opendata.aws/noaa-goes/) | ABI-L2-CMIPC Band 13 imagery from `noaa-goes19.s3.amazonaws.com`, stored as brightness temperature in Kelvin |
| [Iowa Environmental Mesonet ASOS/METAR](https://mesonet.agron.iastate.edu/request/download.phtml) | Observations from TTN and EWR, downloaded through IEM's ASOS CSV service |
| [IEM historical MRMS archive](https://mtarchive.geol.iastate.edu/2026/10/11/mrms/ncep/MultiSensor_QPE_01H_Pass2/) | NOAA `MultiSensor_QPE_01H_Pass2` hourly precipitation accumulations in mm, decoded from GRIB2 with ecCodes |

The grid covers New Jersey and nearby areas, from 39.2 to 41.2 degrees north and -75.3 to -73.0 degrees longitude. All three sources use the same 32x32 latitude/longitude grid, with north in the first row. The verified forecast issuance times are October 11 at 14:00 through 19:00 UTC and October 20 at 15:00 through 20:00 UTC, in hourly steps. August through October 2026 is the candidate study period; we have not verified or downloaded that entire period.

The numerical features, in order, are `temperature_c`, `dewpoint_c`, `humidity_pct`, `wind_speed_mps`, and `pressure_hpa`. The ASOS parser converts Fahrenheit to Celsius and knots to m/s, preserves missing measurements, and applies basic range checks. These are station observations mapped onto the grid using inverse-distance weights within 110 km, rather than a full numerical weather analysis. Solar irradiance is not included because these station records do not provide it reliably.

Each sample uses history ticks at `t-2h`, `t-1h`, and `t`. At each tick, GOES selection checks that the scan was published by then and is no more than 20 minutes old. Imagery is reprojected using the NetCDF projection metadata, and every pixel contributing to bilinear interpolation must have a valid DQF flag. Station selection uses the latest reading at least 10 minutes before the tick and no more than 90 minutes old. IEM does not include historical receipt timestamps here, so the 10-minute reporting delay is an assumption, not proof that every report had arrived by then.

MRMS is resampled to that same grid with nearest-neighbor sampling. The decoder checks the grid geometry, valid time, and NOAA product identity. ecCodes reports the local parameter's units as `unknown`, so the exact centre 161 / discipline 209 / category 6 / parameter 37 identity is checked against the [NOAA MRMS table](https://www.nssl.noaa.gov/projects/mrms/operational/tables.php), which defines this product as one-hour accumulation in mm. Pass2 is delayed quality-controlled data and is used only for targets.

## Setup and tests

Clone the repository and run these commands from its root in PowerShell. Python 3.13 is the verified Windows environment; the package supports Python 3.11 or newer. Each machine needs its own virtual environment.

```powershell
git clone https://github.com/mehulkhajuria/Weather-Forecasting-550-final-project-.git
cd Weather-Forecasting-550-final-project-
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py313.txt
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The lock file records the verified Windows Python 3.13 dependencies, including SciPy, the HDF5 backend, and ecCodes. On Linux, create a local environment with `python3 -m venv .venv`, install `.[all]` with `.venv/bin/python`, and use that interpreter for the same checks. The existing `scripts/setup_linux.sh` and `scripts/verify_linux.sh` are also available.

The offline suite has 37 tests covering parsing, units, missing values, geometry, temporal alignment, future-label placement, leakage checks, and tensor shapes. The real-data verification also checked 48 GOES and 48 MRMS locations against the raw files. [The dataset manifest](docs/REAL_DATA_VERIFICATION.json) records coverage and checksums, and [the alignment results](docs/RAW_ALIGNMENT_VERIFICATION.json) record those raw-file comparisons.

## Getting the data

Raw files, processed arrays, and the handoff ZIP are excluded from Git, so cloning the repo does not download the dataset. If you already have `teammate_handoff.zip`, place it under `data/processed`, extract it there, and run the loading example:

```powershell
Expand-Archive -LiteralPath data/processed/teammate_handoff.zip -DestinationPath data/processed -Force
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

To obtain the original data and reproduce the pilot instead, start by listing the source files for the initial test case. The regeneration script then runs the bounded station requests and two six-issuance batches, builds the tensors, checks raw alignment, and packages the handoff.

```powershell
.\.venv\Scripts\weather550.exe fetch-sample --at 2026-10-20T18:00:00Z --list-only
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/regenerate_real_subset.ps1
```

The exact source requests are in `scripts/regenerate_real_subset.ps1`. ASOS requests cover October 10 through 17 and October 20 through 21, with exclusive midnight end dates. GOES history and MRMS targets are downloaded only for the selected forecast windows. Raw data go under `data/raw`; station CSVs, processed scenes, batch audit logs, and final arrays go under `data/processed`. Downloads are cached. If IEM returns HTTP 429, keep the existing files and retry after the provider's requested delay.

If the station CSVs and processed source scenes are already present, rebuilding and loading need no network access:

```powershell
.\.venv\Scripts\python.exe scripts/build_delivery.py
.\.venv\Scripts\python.exe scripts/verify_real_alignment.py
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

The delivery builder is intentionally set to the two pilot periods and expects 12 complete samples. It produces `supervised.npz`, its source-summary JSON, and `teammate_handoff.zip`. The ZIP includes the dataset, metadata, this README, and a standalone loading example. The general CLI also has `fetch-asos`, `fetch-goes`, `fetch-mrms`, `prepare-numeric`, `fetch-batch`, and `build-supervised` commands; use `--help` for their arguments.

## Using the tensors

`data/processed/supervised.npz` contains the following arrays, with `N=12` for the pilot:

| Array | Shape and contents |
|---|---|
| `X_goes` | `(N,3,1,32,32)`, float32 C13 brightness temperatures in K |
| `X_numerical` | `(N,3,5,32,32)`, float32 features in the order listed above |
| `Y_rain_mm` | `(N,2,32,32)`, float32 hourly precipitation accumulations |
| `X_goes_mask`, `X_num_mask`, `Y_rain_mask` | Boolean validity masks matching their corresponding arrays |
| `lat_grid`, `lon_grid` | `(32,32)`, grid coordinates in degrees north/east |
| `issuance_times_utc`, `forecast_horizons_hours`, `feature_names` | Sample times, horizons `[1,3]`, and numerical feature order |
| `goes_scan_times_utc`, `goes_available_times_utc` | `(N,3)`, scan starts and S3 publication timestamps |
| `latest_station_observation_times_utc`, `rain_valid_end_times_utc` | `(N,3)` latest selected station times and `(N,2)` target end times |
| `source_metadata_json`, `source_station_files` | Per-sample source URLs and paths, plus station CSV paths |
| `observation_latency_minutes`, `max_observation_age_minutes` | Reporting-delay assumption of 10 minutes and age limit of 90 minutes |

Target index 0 is the rainfall during `(t,t+1h]`. Target index 1 is rainfall during `(t+2h,t+3h]`, so it is the one-hour amount ending at +3h, not the cumulative rainfall over three hours.

Some numerical measurements and grid locations have no support, so invalid cells are stored as zero with the mask set to false. Use the masks so those placeholders are not treated as measured zero values. In the pilot, satellite and target validity are 100%, while numerical validity is about 76.8%. All 12 selected samples passed the join. A rainy sample has at least one positive target pixel; a dry sample has all target cells valid and zero.

```python
import numpy as np

with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
    X_goes = data['X_goes']
    X_numerical = data['X_numerical']
    Y_rain_mm = data['Y_rain_mm']
    goes_valid = data['X_goes_mask']
    numerical_valid = data['X_num_mask']
    rain_valid = data['Y_rain_mask']
    times = data['issuance_times_utc']
```

Normalization has not been applied because it needs to be fitted on the training split. Keep the splits chronological and avoid randomly splitting overlapping windows. `build-supervised` supports `--train-before` and `--val-before` boundaries, excluding windows whose history or target intervals cross them, including the station lookback. The pilot is left unsplit. For the learned precipitation-overlay model, training overlays need chronological out-of-fold numerical predictions so the next model does not train on overfit predictions. The satellite-only and direct numerical-feature baselines should use the same samples, targets, and splits.
