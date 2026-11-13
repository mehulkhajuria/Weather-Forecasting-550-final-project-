# CS550 Model Input Contract — v0.1 (provisional; teammate sign-off needed)

## Why this exists

Our CS550 proposal requires (A) temporally aligned GOES imagery, (B) numerical weather observations, and (C) future precipitation fields for **+1 hour and +3 hours**, with comparable satellite-only / direct-feature / learned-overlay models. This first implementation establishes real numerical and single-scene GOES interfaces, **not** the final joint training sample.

## Numerical tensor now delivered

`data/processed/model_numeric.npz` contains:

| Field | dtype / shape | Semantics |
|---|---|---|
| `X_numerical` | float32, `[N,T,F,H,W]` | Causal physical-unit station-derived features |
| `X_mask` | bool, `[N,T,F,H,W]` | True only where there is a valid station reading with grid coverage |
| `nearest_station_km` | float32, `[N,T,H,W]` | Distance to nearest *observed* station; infinity indicates none |
| `lat_grid`, `lon_grid` | float32, `[H,W]` | Geographic cell centers; first row is north |
| `issuance_times_utc` | UTF-8 strings, `[N]` | Last permitted observation time for sample |
| `feature_names` | strings, `[F]` | Column order |

Features in order: `temperature_c`, `dewpoint_c`, `humidity_pct`, `wind_speed_mps`, `pressure_hpa`. **No solar irradiance** has been invented from sparse ASOS stations. The IDW spatial maps are proxy input features, not continuous numerical-weather analyses. The boolean mask is essential, because missing features are stored as zero only for simple numeric tensor compatibility.

Default `T=3`, hourly steps, `H=W=32`, NJ demonstration bounding box `[39.2,-75.3,41.2,-73.0]`. These are **engineering defaults**, not a finalized meteorological training domain. Source data are METAR/ASOS; their original reporting interval is irregular. For each issuance tick, the builder chooses the most recent observation **not later than the tick** and no older than 90 minutes (configurable). These values may be repeated in adjacent ticks and that behavior must be documented during evaluation.

## Satellite single-frame format now delivered

`data/processed/goes_C13.npz` contains:

| Field | dtype / shape | Semantics |
|---|---|---|
| `CMI` | float32, `[H,W]` | Real ABI band 13 Cloud and Moisture imagery; usually brightness temperature (K), see units |
| `quality_mask` | bool, `[H,W]` | Source DQF=0 and valid geolocation |
| `latitude`, `longitude` | float32, `[H,W]` | Same grid definition as numerical output |
| `scan_time_utc` | string | Scene start, NOT model issuance time |
| `source_url` | string | Original NOAA file link |
| `channel`, `units` | scalar | Actual channel/physical units |

GOES projection resampling uses the official per-file geostationary projection attributes. This step deliberately handles **one scan**, not a temporally aligned image sequence.

## Requested teammate interface for the next phase

Suggested contract:

```python
X_num: np.ndarray  # (N,T,F,H,W)
M_num: np.ndarray  # (N,T,F,H,W), bool
X_sat: np.ndarray  # (N,T,C,H,W)  <-- NOT produced yet
M_sat: np.ndarray  # (N,T,C,H,W)  <-- NOT produced yet
Y_rain: np.ndarray # (N,2,H,W)    <-- NOT produced yet
M_rain: np.ndarray # (N,2,H,W)    <-- NOT produced yet
forecast_horizons_minutes = (60,180)
```

The numerical model must output a **spatial precipitation forecast map** at each lead time, not just two scalar precipitation predictions, if following the proposal as written. Teammate may prefer sparse station tokens rather than gridded proxies; decide this together before building a deep FNN.

## Not yet implemented (explicit)

- MRMS `MultiSensor_QPE_01H` GRIB2 decoding and resampling, with proper interval timing and units.
- Actual `Y_rain` for future +60/+180 minutes, target masks, and complete GOES image history.
- Per-sample temporal alignment with known valid intervals and tolerance limits.
- Chronological train/validation/test splits with target-window embargo; fit normalization only on training.
- Forward-chaining OOF numerical overlays required for second-stage training.
- Dataset version manifest with source checksums and train/test reproducibility locks.
- Live AirGradient/Ecowitt/HA/LP02/skycam adapter (separate OpenSky later).

## Strict data-leakage rule

At issuance `t`, all numerical and GOES inputs must have observation/scan availability time `<= t`; future radar targets are used **only** as labels during training/evaluation. Scanner publication delays and MRMS accumulation time semantics must be accounted for in next phase. Never fit scaling parameters using any validation/test observations.

## Data access / class reproducibility

No secrets are necessary for public IEM or NOAA buckets. All raw/processed large data are `.gitignore`d; share only curated samples through course-approved storage. Keep `*.json` source metadata with the dataset outside Git.
