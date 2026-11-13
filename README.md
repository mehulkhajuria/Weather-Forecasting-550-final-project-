> **Verified v0.3 handoff:** Start with [real dataset results and regeneration commands](docs/REAL_DATA_HANDOFF.md). Twelve real rainy/dry samples have been produced on Windows/Python 3.13. The v0.2 material below describes the original numerical-only starter; use the handoff for the current supervised contract and availability limitations.
>
> **Environment setup:** Read [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) for Windows, Rutgers iLab, and Amarel setup. Each computer creates its own `.venv`.

# CS550 Weather Data — Starter (v0.2.1)

**Purpose:** a small, auditable Rutgers CS550 data pipeline for the proposal *Fourier-Based Weather Prediction by Combining Satellite Imagery and Numerical Observations*. It is also structured so observation-processing code can later be reused by **OpenSky GroundNet** in a separate integration.

**Current scope:** download and clean historical station measurements from IEM ASOS; prepare a numerical-model input tensor with UTC, physical units, masks and geographic grid coordinates; optionally download **one** NOAA GOES-19 ABI Band 13 (infrared) NetCDF file and resample it onto the same geographic grid. The included demo is fully offline and **explicitly synthetic**. It does **not** yet build radar precipitation targets or a complete synchronized satellite sequence: this is a real first implementation step, not a trained forecast model.

[Quickstart for Windows / Git / GitHub](QUICKSTART_WINDOWS.md) · [Tensor contract](docs/DATA_CONTRACT.md) · [Phased roadmap](docs/ROADMAP.md)

## One-screen start (PowerShell in this folder)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\weather550.exe demo --out data/demo
```

If `py -3.13` is not found, use `py -3.12` or `py -3.11`; a supported Python **3.11 or newer** is required. No activation or PowerShell execution-policy modification is required.

Expected result: tests pass and `data/demo/SYNTHETIC_numeric.npz` contains **46 samples × 3 time steps × 5 numerical variables × 32 × 32 pixels**, plus a per-element validity mask. The demo is not research data.

### Fetch a small *real* ASOS station sample (Internet required)

```powershell
.\.venv\Scripts\weather550.exe fetch-asos --stations TTN EWR --start 2026-10-20T00:00:00Z --end 2026-10-22T00:00:00Z
.\.venv\Scripts\weather550.exe prepare-numeric --stations data/processed/stations_utc.csv --start 2026-10-20T00:00:00Z --end 2026-10-22T00:00:00Z
```

Output: `data/raw/iem_asos.csv`, `data/processed/stations_utc.csv`, and `data/processed/model_numeric.npz`. Their metadata JSON files show station coverage and validity. ASOS observations are **sparse** and do not necessarily include direct solar radiation. Failed/empty retrievals exit with an error instead of fabricating data. If the IEM returns no coverage for one station/date, adjust the station/date and inspect the `.request.json` metadata.

### Fetch a small *real* GOES sample (Internet required)

```powershell
# First list a candidate GOES-19 C13 CONUS scene WITHOUT downloading:
.\.venv\Scripts\weather550.exe fetch-goes --at 2026-10-20T18:00:00Z --list-only

# Then download and reproject just ONE scene:
.\.venv\Scripts\weather550.exe fetch-goes --at 2026-10-20T18:00:00Z
```

Files: one raw `.nc` under `data/raw/goes/` (capped at 120 MiB), and a 32×32 scene under `data/processed/goes_C13.npz`. No AWS credentials required; anonymous NOAA public S3. It uses actual GOES geostationary projection metadata, not a guessed latitude/longitude mapping. Cloud/quality mask is preserved; invalid values are zero *with mask false*. One scene is **not yet** a satellite sequence, and brightness temperature is **not** precipitation.

If you get a network or remote-bucket error, leave the raw data unchanged, record the console output, and continue with `demo` and local tests. Live network retrieval was **not run by the authoring environment**.

## Commands

```text
weather550 demo            Offline synthetic end-to-end numerical pipeline
weather550 fetch-asos      Real IEM historical station download and conversion
weather550 convert-asos    Offline re-parse an already downloaded IEM CSV
weather550 prepare-numeric Causal historical station-window tensor generation
weather550 fetch-goes      Public NOAA GOES Band-13 single-scene retrieval / reprojection
```

Run `weather550 <command> --help` for every argument.

## Numerical model contract

```python
import numpy as np
batch = np.load('data/processed/model_numeric.npz')
X = batch['X_numerical']        # [N,T,F,H,W], float32, physical units
mask = batch['X_mask']         # [N,T,F,H,W], bool; True = has data support
lat = batch['lat_grid']        # [H,W], degrees north
lon = batch['lon_grid']        # [H,W], degrees east
names = batch['feature_names'] # [F]
times = batch['issuance_times_utc']
```

Feature order: `temperature_c`, `dewpoint_c`, `humidity_pct`, `wind_speed_mps`, `pressure_hpa`. Zeros in invalid places are placeholders; **never treat them as genuine zero measurements**. Pass the mask to the teammate's network, or use explicitly mask-aware imputation.

**The output is not model-supervised precipitation data yet.** The authoritative proposal requires future radar-based precipitation fields at **+60 and +180 min**, GOES image sequences, matched windows, chronological splits and forward-chaining out-of-fold overlays. We will build those after the first real observation and one-scene satellite test pass.

## Design safeguards

- All times UTC. Issuance-history logic only takes station readings `<= issuance_time`, with a maximum observation-age gate.
- Bounding box is `[south,west,north,east]`, shared by numerical tensor and GOES crop. First row is the **northern** row.
- Station variables are distance-weighted to form a **station-conditioned input representation**, not a physical weather analysis. `X_mask` and `nearest_station_km` preserve coverage limits.
- Data download caps prevent accidentally acquiring hundreds of gigs. For research training, collect more events in *bounded batches*, not one unlimited scrape.
- No train/val/test scaler is fitted in this first step; that belongs in the later chronological-split pipeline and must be fit on training only.
- No credentials, HA data, access tokens or arbitrary raw downloads should be committed to the class repository.

## Structure

```text
src/weather550/
  asos.py        IEM ASOS request/parse/units/QC
  goes.py        GOES public S3 listing/download + projection-correct sample
  numerical.py   UTC-causal station windows, IDW pseudo-fields and masks
  demo.py        SYNTHETIC offline fixture
  cli.py         Command line
  timeutil.py    timezone-aware utilities
tests/             No-network unit and integration tests
scripts/           Optional Windows setup/test scripts
docs/              Source contract, next steps, collaboration notes
data/raw/          Ignored downloaded source files
data/processed/    Ignored derived files
```

## Data providers and attribution

- [Iowa Environmental Mesonet ASOS archive](https://mesonet.agron.iastate.edu/request/download.phtml): raw airport observations, with sparse coverage and minimal provider QC.
- [NOAA GOES NODD public distribution](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data.php): anonymous GOES ABI NetCDF imagery; GOES-19 is the appropriate current GOES-East satellite for the 2026 example.
- [NOAA MRMS public archive](https://registry.opendata.aws/noaa-mrms-pds/): planned **future** radar precipitation target source, not yet downloaded/decoded by this version.

This project is independent coursework scaffolding and does not claim NOAA endorsement.


## Repository integrity

The `.[all]` install now declares SciPy, used by the synthetic GOES NetCDF test. GitHub Actions runs the offline tests on Windows and Ubuntu using Python 3.13. For an existing virtual environment, run `python -m pip install -e ".[all]"` again after updating pyproject.toml.

## v0.3 — MRMS + supervised joins

Start with [START_HERE_v0_3.md](START_HERE_v0_3.md) for the complete Windows-to-GitHub-to-one-real-sample procedure. `weather550 fetch-sample` retrieves a bounded 3-GOES + 2-MRMS case; `weather550 fetch-batch` processes up to 36 hourly issuances with caching and an audit log; `weather550 build-supervised` joins numerics and future rain. These are new and have offline tests, but still need a real-source Windows verification. Read [Team Handoff](docs/TEAM_HANDOFF.md) and [Narrow Codex Integration](docs/CODEX_SINGLE_PASS_INTEGRATION.md).
