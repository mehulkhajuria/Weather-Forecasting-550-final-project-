# CS550 Weather Forecasting

This repository handles the data side of our CS550 project, *Fourier-Based Weather Prediction by Combining Satellite Imagery and Numerical Observations*. It collects satellite imagery, station observations, and precipitation measurements, then processes them into matching grids and time windows for the models. Keeping this separate from the model lets us try the satellite-only, direct numerical-feature, and precipitation-overlay approaches using the same inputs and targets.

The current handoff has 72 real samples from August through October 2026: 52 samples with rain in at least one target map and 20 with both target maps dry. They cover eight separated rainy event groups, three fully dry periods, and one mixed period. Each group has six hourly samples, so those hours are not counted as separate storms. The data pipeline downloads and decodes all three sources, builds the numerical grids, and joins the satellite history with future precipitation targets. The dataset is ready for model integration, but event count by itself does not establish that it is enough for reliable evaluation. Model implementation and training are separate.

## Data and preprocessing

| Source | What we use |
|---|---|
| [NOAA GOES-19 public archive](https://registry.opendata.aws/noaa-goes/) | ABI-L2-CMIPC Band 13 imagery from `noaa-goes19.s3.amazonaws.com`, stored as brightness temperature in Kelvin |
| [Iowa Environmental Mesonet ASOS/METAR](https://mesonet.agron.iastate.edu/request/download.phtml) | Observations from TTN and EWR, downloaded through IEM's ASOS CSV service |
| [IEM historical MRMS archive](https://mtarchive.geol.iastate.edu/2026/10/11/mrms/ncep/MultiSensor_QPE_01H_Pass2/) | NOAA `MultiSensor_QPE_01H_Pass2` hourly precipitation accumulations in mm, decoded from GRIB2 with ecCodes |

The grid covers New Jersey and nearby areas, from 39.2 to 41.2 degrees north and -75.3 to -73.0 degrees longitude. All three sources use the same 32x32 latitude/longitude grid, with north in the first row. The verified issuance windows are listed below; all times are UTC, sampled hourly, and the end hour shown is inclusive. August through October 2026 is the candidate study period, not continuous verified coverage.

| Group | Issuance dates and hours |
|---|---|
| Rainy | August 3, 07:00 to 12:00; August 7, 20:00 to August 8, 01:00 |
| Rainy | August 17, 01:00 to 06:00; August 20, 19:00 to August 21, 00:00 |
| Rainy | September 1, 22:00 to September 2, 03:00; September 20, 23:00 to September 21, 04:00 |
| Rainy | October 4, 12:00 to 17:00; October 11, 14:00 to 19:00 |
| Dry | September 7, 12:00 to 17:00; October 20, 15:00 to 20:00; October 28, 12:00 to 17:00 |
| Mixed | August 1, 12:00 to 17:00 |

Rainy groups were screened with TTN/EWR rain reports, confirmed against MRMS targets, and separated by at least 48 hours between their full input/target windows, including the station lookback. This gives us identifiable event groups without treating adjacent hours as independent storms; it does not prove statistical independence. August 1 was initially a dry candidate at the two stations, but MRMS showed rain elsewhere in the grid. Those valid samples were kept as mixed rather than counted as a dry period or an additional independent rainy event. All 72 configured issuances passed source collection and the join, with no failures or rejected samples.

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

The offline suite has 43 tests covering parsing, units, missing values, geometry, temporal alignment, future-label placement, leakage checks, and tensor shapes. Independent raw-file checks passed at 288 GOES and 288 MRMS locations. [The dataset manifest](docs/REAL_DATA_VERIFICATION.json) records coverage and checksums, and [the alignment results](docs/RAW_ALIGNMENT_VERIFICATION.json) record those raw-file comparisons.

## Getting the data

Raw files, processed arrays, and the handoff ZIP are excluded from Git, so cloning the repo does not download the dataset. If you already have `teammate_handoff.zip`, place it under `data/processed`, extract it there, and run the loading example:

```powershell
Expand-Archive -LiteralPath data/processed/teammate_handoff.zip -DestinationPath data/processed -Force
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

To obtain the original data and reproduce the selected periods instead, start by listing the source files for the initial test case. The regeneration script reads the selected events from `scripts/collection_periods.json`, checks source availability, runs the bounded station requests and hourly batches, builds the tensors, checks raw alignment, and packages the handoff.

```powershell
.\.venv\Scripts\weather550.exe fetch-sample --at 2026-10-20T18:00:00Z --list-only
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/regenerate_real_subset.ps1
```

The exact event windows and station requests are in `scripts/collection_periods.json`. Each entry has an event ID, UTC issuance bounds, and an ASOS request of at most eight days with an exclusive midnight end. `scripts/regenerate_real_subset.ps1` runs that configuration; pass `-Periods path/to/periods.json` to use another configuration. GOES history and MRMS targets are downloaded only for the selected forecast windows. Raw data go under `data/raw`; station CSVs, processed scenes, batch audit logs, and final arrays go under `data/processed`. Downloads are cached. If IEM returns HTTP 429, keep the existing files and retry after the provider's requested delay.

If the station CSVs and processed source scenes are already present, rebuilding and loading need no network access:

```powershell
.\.venv\Scripts\python.exe scripts/build_delivery.py
.\.venv\Scripts\python.exe scripts/verify_real_alignment.py
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

The delivery builder reads the configuration and keeps the samples that pass the existing alignment and quality checks. The sample count comes from the records that actually join, rather than a fixed count. `python scripts/build_delivery.py --periods scripts/collection_periods.json` rebuilds from cached data; adding `--fetch` checks availability and collects the configured periods first. It produces `supervised.npz`, its source-summary JSON, and `teammate_handoff.zip`. The ZIP includes the dataset, metadata, this README, and a standalone loading example. From an extracted bundle folder, run `python load_dataset.py supervised.npz` using the project environment. The general CLI also has `fetch-asos`, `fetch-goes`, `fetch-mrms`, `prepare-numeric`, `fetch-batch`, and `build-supervised` commands; use `--help` for their arguments.

## Using the tensors

`data/processed/supervised.npz` contains the following arrays, with `N=72` for the current handoff:

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
| `event_ids` | `(N,)`, shared event/period ID for every hourly sample in the same group |
| `observation_latency_minutes`, `max_observation_age_minutes` | Reporting-delay assumption of 10 minutes and age limit of 90 minutes |

Target index 0 is the rainfall during `(t,t+1h]`. Target index 1 is rainfall during `(t+2h,t+3h]`, so it is the one-hour amount ending at +3h, not the cumulative rainfall over three hours.

Some numerical measurements and grid locations have no support, so invalid cells are stored as zero with the mask set to false. Use the masks so those placeholders are not treated as measured zero values. Satellite and target validity are 100%, while numerical validity is about 76.70% (23.30% missing). Temperature, dew point, humidity, and wind each have 80.27% validity; pressure has 62.40% validity. This includes both missing station readings and pixels outside the station support radius. A rainy sample has at least one positive target pixel; a dry sample has all target cells valid and zero.

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
    event_ids = data['event_ids']
```

Normalization has not been applied because it needs to be fitted on the training split. Keep the splits chronological and avoid randomly splitting overlapping windows. `build-supervised` supports `--train-before` and `--val-before` boundaries, excluding windows whose history or target intervals cross them, including the station lookback. The delivered data are left unsplit. Keep all hours from an event together when choosing held-out storms. For the learned precipitation-overlay model, training overlays need chronological out-of-fold numerical predictions so the next model does not train on overfit predictions. The satellite-only and direct numerical-feature baselines should use the same samples, targets, and splits.

The original 12-sample pilot is preserved locally as `data/processed/pilot_supervised.npz`, with SHA-256 `26408cb7d8b4188c47bbb8daa9d292d1c227373d8888f296be43b282d4952d0c`. Its original results are kept in `docs/PILOT_DATA_VERIFICATION.json`, which is also in the handoff ZIP.

The numerical inputs were checked with the input loader and dataset class on the `models` branch at `b4fa3fc`. Its 3D arrangement is a transpose to `(N,5,3,32,32)`; a 2D model can flatten the stored history and feature axes to `(N,15,32,32)`. These are model-side changes, so the saved arrays keep the same format. The branch currently has an `exit(0)` in `get_rain_data_ndarray` before it returns targets. That prevents its training entry point from proceeding and needs a model-side fix; those files were not changed here. PyTorch input loading was tested using the existing global installation, without adding it to the data pipeline dependencies.
