"""Causal, sparse-station-to-grid observation representation for a teammate's numerical model.

WARNING: This is an inverse-distance-weighted *station-conditioned feature representation*,
not a calibrated meteorological gridded analysis. We export coverage masks and source ages.
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .asos import FEATURES
from .timeutil import parse_utc, iso_utc


def grid_coordinates(bbox, height: int, width: int):
    """bbox=(south,west,north,east), north is first image row."""
    south, west, north, east = [float(v) for v in bbox]
    if not (-90 < south < north < 90 and -180 < west < east < 180):
        raise ValueError("Invalid bounding box [south,west,north,east]")
    if min(height, width) < 2:
        raise ValueError("Grid must be at least 2x2")
    lat = np.linspace(north, south, height, dtype=np.float64)
    lon = np.linspace(west, east, width, dtype=np.float64)
    return np.meshgrid(lat, lon, indexing="ij")


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat/2)**2 + np.cos(lat1*r)*np.cos(lat2*r)*np.sin(dlon/2)**2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def station_map(frame: pd.DataFrame, lat_grid: np.ndarray, lon_grid: np.ndarray,
                radius_km=110.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = len(FEATURES)
    h, w = lat_grid.shape
    values = np.zeros((n_features,h,w), dtype=np.float32)
    mask = np.zeros((n_features,h,w), dtype=np.bool_)
    closest = np.full((h,w), np.inf, dtype=np.float32)
    if frame.empty:
        return values, mask, closest
    stations_lat = frame.latitude.to_numpy(dtype=float)
    stations_lon = frame.longitude.to_numpy(dtype=float)
    distances = haversine_km(stations_lat[:,None,None], stations_lon[:,None,None],
                             lat_grid[None], lon_grid[None])
    closest = distances.min(axis=0).astype(np.float32)
    for i, f in enumerate(FEATURES):
        obs = pd.to_numeric(frame[f], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(obs)
        weights = np.where(valid[:,None,None] & (distances <= radius_km),
                           1.0 / np.maximum(distances, 1.0)**2, 0.0)
        weight_sum = weights.sum(axis=0)
        good = weight_sum > 0
        if np.any(good):
            numerator = (weights * np.nan_to_num(obs)[:,None,None]).sum(axis=0)
            values[i,good] = (numerator[good] / weight_sum[good]).astype(np.float32)
            mask[i,good] = True
    return values, mask, closest


def latest_causal(df, timestamp, max_age_minutes, latency_minutes=0):
    """No future observation is permitted, even within interpolation windows."""
    lo = timestamp - timedelta(minutes=int(max_age_minutes))
    cutoff = timestamp - timedelta(minutes=int(latency_minutes))
    frame = df[(df.timestamp_utc <= cutoff) & (df.timestamp_utc >= lo)]
    if frame.empty:
        return frame
    return frame.sort_values("timestamp_utc").drop_duplicates("station", keep="last")


def build_numerical(
    station_csv: Path, outfile: Path, start: str, end: str, *,
    bbox=(39.2,-75.3,41.2,-73.0), height=32, width=32,
    history_steps=3, step_minutes=60, max_age_minutes=90, radius_km=110., latency_minutes=10,
):
    if history_steps < 1 or step_minutes < 1 or max_age_minutes < 1 or latency_minutes < 0:
        raise ValueError("History, step and max_age must be positive")
    df = pd.read_csv(station_csv)
    required = {"station", "timestamp_utc", "latitude", "longitude", *FEATURES}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing columns: {sorted(required-set(df.columns))}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="raise")
    start_dt, end_dt = parse_utc(start), parse_utc(end)
    if not end_dt > start_dt:
        raise ValueError("end must be strictly after start")
    if (end_dt - start_dt) > timedelta(days=31):
        raise ValueError("Numerical demo builder limited to 31 days per run")
    lat_grid, lon_grid = grid_coordinates(bbox,height,width)
    times = pd.date_range(start=start_dt, end=end_dt, freq=f"{step_minutes}min", inclusive="left")
    xs, masks, times_out, nearests, latest_times = [], [], [], [], []
    # Each issuance time's history includes the current tick and strictly earlier ticks.
    for issuance in times:
        issuance = issuance.to_pydatetime()
        if issuance - timedelta(minutes=int((history_steps-1)*step_minutes)) < start_dt:
            continue
        sample_x, sample_mask, sample_dist, sample_latest = [], [], [], []
        for lag in range(history_steps-1, -1, -1):
            t = issuance - timedelta(minutes=int(lag*step_minutes))
            frame = latest_causal(df, t, max_age_minutes, latency_minutes)
            sample_latest.append('' if frame.empty else iso_utc(frame.timestamp_utc.max().to_pydatetime()))
            val, mask, dist = station_map(frame,lat_grid,lon_grid,radius_km)
            sample_x.append(val)
            sample_mask.append(mask)
            sample_dist.append(dist)
        xs.append(np.stack(sample_x))
        masks.append(np.stack(sample_mask))
        nearests.append(np.stack(sample_dist))
        latest_times.append(sample_latest)
        times_out.append(iso_utc(issuance))
    if not xs:
        raise ValueError("No windows generated: choose a longer time interval")
    X = np.stack(xs).astype("float32")
    M = np.stack(masks)
    D = np.stack(nearests).astype("float32")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outfile,
        X_numerical=X, X_mask=M, nearest_station_km=D,
        issuance_times_utc=np.array(times_out),
        feature_names=np.array(FEATURES),
        lat_grid=lat_grid.astype("float32"), lon_grid=lon_grid.astype("float32"),
        history_offsets_minutes=np.arange(-(history_steps-1),1)*step_minutes,
        latest_observation_times_utc=np.array(latest_times),
        observation_latency_minutes=np.array(latency_minutes),
        max_observation_age_minutes=np.array(max_age_minutes),
        source_station_files=np.array([str(station_csv)]))
    summary={
        "contract": "CS550-NUMERIC-v0.1", "file":str(outfile),
        "shape_X_numerical":list(X.shape), "shape_X_mask":list(M.shape),
        "N":len(times_out),"observed_fraction":float(M.mean()),
        "forecast_horizons_minutes_proposed":[60,180],
        "targets_present":False,
        "values_are_normalized":False,
        "feature_names":list(FEATURES),
        "time_utc_start":times_out[0],"time_utc_end":times_out[-1],
        "source_csv":str(station_csv),
        "bbox_south_west_north_east":list(bbox),
        "radius_km":radius_km,"max_observation_age_minutes":max_age_minutes,
        "observation_latency_minutes":latency_minutes,
        "availability_limitation":"IEM archive lacks receipt times; observation time plus latency is an explicit retrospective assumption, not verified receipt.",
        "warning":"Spatially gridded station-conditioned features are IDW proxies, not NWP analyses. No solar/radar targets.",
        "normalization_policy":"Fit scaler only on chronological training subset after targets assembled.",
    }
    outfile.with_suffix(".json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary
