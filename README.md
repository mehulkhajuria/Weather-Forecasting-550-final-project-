# CS550 Weather Forecasting

This repository handles the data side of our CS550 project, *Fourier-Based Weather Prediction by Combining Satellite Imagery and Numerical Observations*. It collects satellite imagery, station observations, and precipitation measurements, then processes them into matching grids and time windows for the models. Keeping this separate from the model lets us try the satellite-only, direct numerical-feature, and precipitation-overlay approaches using the same inputs and targets.

The current handoff covers the requested study interval, August 1, 2026 at 00:00 UTC through November 1 at 00:00 UTC, exclusive. All 2,208 possible hourly issuances were attempted. We kept 2,123 real samples: 923 with rain in at least one target map and 1,200 with both target maps dry. The other 85 hours are accounted for below. We stopped at the end of the requested interval, rather than at a sample-count target. Model implementation, normalization, and training remain separate from this repository.

## Data and preprocessing

| Source | What we use |
|---|---|
| [NOAA GOES-19 public archive](https://registry.opendata.aws/noaa-goes/) | ABI-L2-CMIPC Band 13 imagery from `noaa-goes19.s3.amazonaws.com`, stored as brightness temperature in Kelvin |
| [Iowa Environmental Mesonet ASOS/METAR](https://mesonet.agron.iastate.edu/request/download.phtml) | Observations from TTN and EWR, downloaded through IEM's ASOS CSV service |
| [IEM historical MRMS archive](https://mtarchive.geol.iastate.edu/2026/10/11/mrms/ncep/MultiSensor_QPE_01H_Pass2/) | NOAA `MultiSensor_QPE_01H_Pass2` hourly precipitation accumulations in mm, decoded from GRIB2 with ecCodes |

The grid covers New Jersey and nearby areas, from 39.2 to 41.2 degrees north and -75.3 to -73.0 degrees longitude. All three sources use the same 32x32 latitude/longitude grid, with north in the first row. The retained issuance times run from August 1 at 00:00 UTC through October 31 at 23:00 UTC, with the following gaps:

| Reason | Unusable issuance hours | Affected dates |
|---|---|---|
| GOES quality below the existing 80% threshold | 24 | August 22; August 29 through September 4 |
| No GOES scan published by the input cutoff within the 20-minute freshness limit | 8 | September 22 |
| Required MRMS targets returned HTTP 404 | 53 | October 6 through October 11 |

The counts reconcile as `2,208 attempted = 2,123 usable + 85 rejected`, with zero unattempted hours. There are 16 missing intervals. Their exact inclusive starts and exclusive ends, the reason for each rejected hour, and the source failures are in the dataset manifest's `coverage` field and the ZIP's `study_coverage.json`. These are gaps in the data, so they have not been replaced with adjacent scans or invented measurements. Historical inputs extend into July 31, and future targets extend through November 1 at 02:00 UTC where needed. That keeps valid issuances at the study boundaries.

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

The offline suite has 57 tests covering parsing, units, missing values, geometry, temporal alignment, future-label placement, leakage checks, and tensor shapes. The independent verifier checks three locations in every raw source file referenced by the final dataset. It uses native ecCodes neighbors for MRMS and the declared grid nodes with an independent CF projection for GOES, checking exported coordinates and DQF masks. [The dataset manifest](docs/REAL_DATA_VERIFICATION.json) records coverage and checksums, and [the alignment results](docs/RAW_ALIGNMENT_VERIFICATION.json) record those raw-file comparisons.

## Getting the data

Raw files, processed arrays, and the handoff ZIP are excluded from Git, so cloning the repo does not download the dataset. If you already have `teammate_handoff.zip`, place it under `data/processed`, extract it there, and run the loading example:

```powershell
Expand-Archive -LiteralPath data/processed/teammate_handoff.zip -DestinationPath data/processed -Force
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

To obtain the original data and reproduce the full study interval instead, start by listing the source files for the initial test case. The regeneration script reads the continuous `study` interval from `scripts/collection_periods.json`, runs bounded source requests and daily assembly, checks raw alignment, and packages the handoff.

```powershell
.\.venv\Scripts\weather550.exe fetch-sample --at 2026-10-20T18:00:00Z --list-only
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/regenerate_real_subset.ps1
```

The `study` entry defines the full interval without listing every day. The older 12 selected periods are kept in the same file as regression references. ASOS is requested in chunks of at most eight days, including the historical-input boundary. GOES and MRMS sources are scheduled in daily batches with up to four worker processes, and each needed source hour is downloaded and processed once. A free-space check uses an approximately 20 GB working estimate for this interval.

Raw data go under `data/raw`; station CSVs, processed scenes, progress journals, coverage records, and arrays go under `data/processed`. The collector validates cached scenes before reuse. `study_sources.jsonl` retains source attempts across interruptions, while `study_coverage.json` accounts for every issuance without double-counting retries. The default rerun reuses successful work and leaves recorded failures alone. Pass `-RetryFailed` to the PowerShell script to retry failed sources, or use `--retry-failed` with the Python builder's `--fetch` option. Requests use bounded retries, backoff, and server Retry-After instructions for rate limits. One missing file does not stop the remaining hours.

If the station CSVs and processed source scenes are already present, rebuilding and loading need no network access:

```powershell
.\.venv\Scripts\python.exe scripts/build_delivery.py
.\.venv\Scripts\python.exe scripts/verify_real_alignment.py --dataset data/processed/supervised.npz --workers 4
.\.venv\Scripts\python.exe scripts/load_dataset.py data/processed/supervised.npz
```

The delivery builder reads the configuration and keeps the samples that pass the existing alignment and quality checks. The sample count comes from the records that actually join, rather than a fixed count. `python scripts/build_delivery.py --periods scripts/collection_periods.json` rebuilds from cached data; adding `--fetch` checks availability and collects the configured periods first. It produces `supervised.npz`, its source-summary JSON, and `teammate_handoff.zip`. The ZIP includes the dataset, metadata, this README, and a standalone loading example. From an extracted bundle folder, run `python load_dataset.py supervised.npz` using the project environment. The general CLI also has `fetch-asos`, `fetch-goes`, `fetch-mrms`, `prepare-numeric`, `fetch-batch`, and `build-supervised` commands; use `--help` for their arguments.

## Using the tensors

`data/processed/supervised.npz` contains the following arrays, with `N=2123` for the current handoff:

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
| `event_ids` | `(N,)`, original period IDs for the 72 reference samples, episode/dry-group IDs elsewhere |
| `precip_episode_ids` | `(N,)`, continuous-study episode labels for evaluation grouping across legacy period boundaries |
| `legacy_period_ids` | `(N,)`, original 12 period identifiers where applicable, otherwise empty strings |
| `observation_latency_minutes`, `max_observation_age_minutes` | Reporting-delay assumption of 10 minutes and age limit of 90 minutes |

Target index 0 is the rainfall during `(t,t+1h]`. Target index 1 is rainfall during `(t+2h,t+3h]`, so it is the one-hour amount ending at +3h, not the cumulative rainfall over three hours.

Some numerical measurements and grid locations have no support, so invalid cells are stored as zero with the mask set to false. Use the masks so those placeholders are not treated as measured zero values. Satellite and target validity are 100%, while numerical validity is about 79.06% (20.94% missing). Temperature, dew point, and humidity each have 80.24% validity; wind has 80.23%, and pressure has 74.33%. This includes both missing station readings and pixels outside the station support radius. A rainy sample has at least one positive target pixel; a dry sample has all target cells valid and zero.

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
    episode_ids = data['precip_episode_ids']
```

Normalization has not been applied because it needs to be fitted on the training split. Keep the splits chronological and avoid randomly splitting overlapping windows. `build-supervised` supports `--train-before` and `--val-before` boundaries, excluding windows whose history or target intervals cross them, including the station lookback. The delivered data are left unsplit. Use `precip_episode_ids` to keep whole precipitation episodes together when choosing held-out events, including hours inside the old selected periods. For the learned precipitation-overlay model, training overlays need chronological out-of-fold numerical predictions so the next model does not train on overfit predictions. The satellite-only and direct numerical-feature baselines should use the same samples, targets, and splits.

Episode labels are retrospective evaluation metadata derived from the target maps, never numerical predictor features. A positive valid target pixel starts or continues an episode. Only 24 consecutive hourly samples with fully valid zero targets end one. Missing hours cannot establish dry separation, so an episode can bridge a coverage gap. This produces 13 precipitation episode groups and 39 calendar-based dry labels; neither count proves statistical independence. The 39 labels are not 39 independently verified dry events. The original 12 event/period IDs are still available, but they are not the only grouping of the longer dataset.

The original 72-sample dataset and its reports are preserved locally as `data/processed/reference72_supervised.npz`, `reference72_supervised.json`, `reference72_teammate_handoff.zip`, and the corresponding archived verification JSON files. Its NPZ SHA-256 remains `169a7a3968fd784f3bba76c133ed6d9c9b136cc655d50a4d95772112a53e31b4`. All 72 samples remain in the full dataset unchanged, including values, masks, source/target times, feature order, and original period IDs. The manifest records those comparisons.

The original 12-sample pilot is preserved locally as `data/processed/pilot_supervised.npz`, with SHA-256 `26408cb7d8b4188c47bbb8daa9d292d1c227373d8888f296be43b282d4952d0c`. Its original results are kept in `docs/PILOT_DATA_VERIFICATION.json`, which is also in the handoff ZIP.

The numerical inputs were checked with the input loader and dataset class on the `models` branch at `b4fa3fc`. Its 3D arrangement is a transpose to `(N,5,3,32,32)`; a 2D model can flatten the stored history and feature axes to `(N,15,32,32)`. These are model-side changes, so the saved arrays keep the same format. The branch currently has an `exit(0)` in `get_rain_data_ndarray` before it returns targets. That prevents its training entry point from proceeding and needs a model-side fix; those files were not changed here. PyTorch input loading was tested using the existing global installation, without adding it to the data pipeline dependencies.
