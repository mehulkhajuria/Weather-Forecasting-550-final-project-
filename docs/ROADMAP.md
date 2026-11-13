# Build sequence — milestones, not a Codex black box

## Milestone 0 — done and offline tested in this package
- [x] Python package, pyproject, console CLI, `pytest` tests
- [x] Fully offline synthetic fixture, parse -> causal numerical tensor end to end
- [x] IEM ASOS live downloader, CSV parse, UTC normalization, unit conversions, QC, masks
- [x] NOAA GOES-19 C13 public S3 listing / single scene downloader and geostationary reprojection
- [x] Clear numerical tensor interface for teammate
- [ ] Live IEM endpoint test **on user's machine**
- [ ] Live GOES bucket test **on user's machine**
- [ ] GitHub repo creation and teammate integration **on user's machine**

## Milestone 1 — next after real sample verified
- [ ] Audit actual IEM station coverage, missingness, and geographic sample counts
- [ ] Verify 1–3 sample GOES scans (C13 IR first, perhaps C02 visible later)
- [ ] Choose model domain and time intervals with teammate
- [ ] Choose target product: NOAA MRMS MultiSensor_QPE_01H historical, not radar VIL/reflectivity
- [ ] Decode MRMS GRIB2 on Windows in a reversible optional environment; validate units, expected valid interval, reproject onto common grid, and compare against a known public display
- [ ] Create a data manifest for 5–10 **real** forecast-issuance cases including sample availability and target provenance

## Milestone 2 — trainable sample generation
- [ ] Use sliding history windows: X_sat, X_num, masks, and future Y (+60/+180)
- [ ] Match actual scan valid / publication times; reject cases with missing required labels
- [ ] Distinguish numerically observed stations from non-observed interpolation with masks
- [ ] Add data-version hashes, shape consistency tests and dataloader
- [ ] Chronological train/val/test with 3h horizon embargo; train-only normalization

## Milestone 3 — support CS550 team's experiment
- [ ] Agree on tensor shape, units and precipitation target semantics
- [ ] Numerical-only FNN and persistence baseline
- [ ] Satellite-only Fourier backbone
- [ ] Direct numerical-concatenation baseline
- [ ] Learned numerical-forecast overlay pipeline, **forward-chaining out-of-fold**; no in-sample overlay leakage
- [ ] Matched RMSE, MAE, CSI / rain event metrics and a simple held-out case study

## Milestone 4 — OpenSky personal project, separate code boundary
- [ ] Safe Home Assistant read-only exporter for live AirGradient/Ecowitt/soil data
- [ ] Preserve source entity IDs, unit, UTC, calibration, QC and missing-data flags
- [ ] Store numeric time-series in PostgreSQL (not vector database)
- [ ] Add LP02 solar thermopile after correctly calibrated ADS1115 acquisition
- [ ] Add upward-facing camera images/optical flow, lightning detector, GOES receiver metadata
- [ ] Evaluate whether your local readings materially improve held-out forecast skill over the public-data baseline

## Session handoff convention
After each milestone record: last executed command, test count/result, files changed, source timestamps, sample counts, known limitations, the **exact** next command. This avoids the opaque partial state that previously caused trouble with Codex.
