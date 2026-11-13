"""Small, auditable retrieval of IEM ASOS observations.

Source: https://mesonet.agron.iastate.edu/request/download.phtml
Important: ASOS is sparse station data, and does NOT reliably contain solar irradiance.
"""
from __future__ import annotations

from datetime import timedelta
from io import StringIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import requests

from .timeutil import iso_utc, parse_utc

ASOS_ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# We explicitly request latlon=yes. The provider may still return gaps; never invent coordinates.
FEATURES = ("temperature_c", "dewpoint_c", "humidity_pct", "wind_speed_mps", "pressure_hpa")


def build_params(stations: list[str], start, end) -> list[tuple[str, str]]:
    if not stations:
        raise ValueError("At least one ASOS station is required")
    if end <= start or end - start > timedelta(days=8):
        raise ValueError("Choose a positive window of at most 8 days per request")
    if start.time() != start.replace(hour=0, minute=0, second=0, microsecond=0).time():
        raise ValueError("ASOS's date-level API requires a start at 00:00 UTC")
    if end.time() != end.replace(hour=0, minute=0, second=0, microsecond=0).time():
        raise ValueError("ASOS's date-level API requires an exclusive end at 00:00 UTC")
    p = [
        ("data", "all"),
        ("year1", str(start.year)), ("month1", str(start.month)), ("day1", str(start.day)),
        ("year2", str(end.year)), ("month2", str(end.month)), ("day2", str(end.day)),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"),
        ("elev", "no"), ("missing", "M"), ("trace", "T"),
        ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
    ]
    p.extend(("station", s.upper()) for s in stations)
    return p


def download_asos(stations: list[str], start, end, outfile: Path, session=None) -> Path:
    p = build_params(stations, start, end)
    request_file = outfile.with_suffix('.request.json')
    if outfile.exists() and request_file.exists():
        meta = json.loads(request_file.read_text(encoding='utf-8'))
        if (meta.get('source') == ASOS_ENDPOINT and
            sorted(s.upper() for s in meta.get('stations', [])) == sorted(s.upper() for s in stations) and
            meta.get('start_utc') == iso_utc(start) and
            meta.get('end_exclusive_utc') == iso_utc(end)):
            cached = parse_asos_csv(outfile)
            if not cached.empty and cached.timestamp_utc.min() >= start and cached.timestamp_utc.max() < end:
                return outfile
    session = session or requests.Session()
    response = session.get(ASOS_ENDPOINT, params=p, timeout=90,
                           headers={"User-Agent": "cs550-weather-data/0.1 (research sample)"})
    response.raise_for_status()
    payload = response.text
    header_line = next((x for x in payload.splitlines() if x.strip() and not x.startswith("#")), "")
    if "station" not in header_line.lower() or "valid" not in header_line.lower():
        raise ValueError("IEM response did not contain a station/valid CSV header; raw data not written")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(payload, encoding="utf-8")
    outfile.with_suffix(".request.json").write_text(
        json.dumps({"source": ASOS_ENDPOINT, "stations": stations,
                    "start_utc": iso_utc(start), "end_exclusive_utc": iso_utc(end),
                    "latlon_requested": True, "raw_file": outfile.name}, indent=2),
        encoding="utf-8",
    )
    return outfile


def _numeric(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.replace({"M": np.nan, "T": np.nan, "": np.nan}), errors="coerce")


def parse_asos_csv(raw_csv: Path) -> pd.DataFrame:
    content = raw_csv.read_text(encoding="utf-8-sig")
    stripped = "\n".join(x for x in content.splitlines() if x.strip() and not x.startswith("#"))
    if not stripped:
        raise ValueError("ASOS file is empty")
    df = pd.read_csv(StringIO(stripped), low_memory=False, dtype={"station": "string"})
    df.columns = [x.strip().lower() for x in df.columns]
    if "station" not in df.columns or "valid" not in df.columns:
        raise ValueError("Expected 'station' and 'valid' columns")
    lat_key = next((c for c in ("lat", "latitude") if c in df), None)
    lon_key = next((c for c in ("lon", "longitude") if c in df), None)
    if lat_key is None or lon_key is None:
        raise ValueError("Station coordinates missing; fetch again with latlon=yes")

    out = pd.DataFrame()
    out["station"] = df["station"].astype("string").str.upper()
    out["timestamp_utc"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    out["latitude"] = _numeric(df[lat_key])
    out["longitude"] = _numeric(df[lon_key])
    def c(name):
        return _numeric(df[name]) if name in df else pd.Series(np.nan, index=df.index, dtype=float)
    out["temperature_c"] = (c("tmpf") - 32.0) * 5.0 / 9.0
    out["dewpoint_c"] = (c("dwpf") - 32.0) * 5.0 / 9.0
    out["humidity_pct"] = c("relh")
    out["wind_speed_mps"] = c("sknt") * 0.514444
    out["pressure_hpa"] = c("mslp")
    # ASOS is not a reliable irradiance source. This is intentionally NOT synthesized.
    out = out.dropna(subset=["timestamp_utc", "latitude", "longitude"])
    out = out[(out.latitude.between(-90,90)) & (out.longitude.between(-180,180))]
    out.loc[~out.humidity_pct.between(0,100), "humidity_pct"] = np.nan
    out.loc[~out.temperature_c.between(-90,65), "temperature_c"] = np.nan
    out.loc[~out.wind_speed_mps.between(0,125), "wind_speed_mps"] = np.nan
    out.loc[~out.pressure_hpa.between(850,1100), "pressure_hpa"] = np.nan
    # Special and routine METAR can duplicate a station/time. Choose row with most fields.
    out["_count"] = out[list(FEATURES)].notna().sum(axis=1)
    out = out.sort_values(["station", "timestamp_utc", "_count"])
    out = out.drop_duplicates(subset=["station", "timestamp_utc"], keep="last")
    return out.drop(columns="_count").sort_values(["timestamp_utc", "station"]).reset_index(drop=True)


def convert_asos(raw: Path, outfile: Path) -> dict:
    df = parse_asos_csv(raw)
    if df.empty:
        raise ValueError("No valid ASOS station records in the input")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "source": str(raw), "output": str(outfile), "records": int(len(df)),
        "stations": sorted(str(x) for x in df.station.unique()),
        "time_min_utc": df.timestamp_utc.min().isoformat(),
        "time_max_utc": df.timestamp_utc.max().isoformat(),
        "feature_valid_counts": {f: int(df[f].notna().sum()) for f in FEATURES},
        "solar_irradiance_available": False,
    }
    outfile.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
