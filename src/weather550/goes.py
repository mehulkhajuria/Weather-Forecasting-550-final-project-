"""Anonymous GOES ABI Band 13 sample retrieval and lat/lon reprojection.

Sources: NOAA NODD public S3 buckets, specifically ABI-L2-CMIPC (CONUS Band C13).
Outputs single real, geolocated CMI raster with DQF quality mask; does NOT create precipitation labels.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import requests

from .numerical import grid_coordinates
from .timeutil import iso_utc

RE_START = re.compile(r"_s(\d{4}\d{3}\d{2}\d{2}\d{2}\d)")


def scan_start_from_key(key: str) -> datetime:
    m = RE_START.search(key)
    if not m:
        raise ValueError(f"Cannot parse GOES scan start from {key}")
    digits = m.group(1)
    return datetime.strptime(digits[:13], "%Y%j%H%M%S").replace(
        tzinfo=timezone.utc) + timedelta(microseconds=int(digits[13]) * 100000)


def list_public_s3(bucket: str, prefix: str, session=None, max_pages: int=3,
                   with_availability=False):
    session = session or requests.Session()
    keys: list[str] = []
    availability = {}
    token = None
    for _ in range(max_pages):
        params = {"list-type":"2", "prefix":prefix, "max-keys":"1000"}
        if token:
            params["continuation-token"] = token
        resp = session.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=40)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
        keys.extend(n.text for n in root.findall(".//s:Contents/s:Key", ns) if n.text)
        for entry in root.findall("s:Contents", ns):
            key = entry.findtext("s:Key", namespaces=ns)
            modified = entry.findtext("s:LastModified", namespaces=ns)
            if key and modified:
                availability[key] = datetime.fromisoformat(modified.replace('Z', '+00:00'))
        is_truncated = root.findtext("s:IsTruncated", default="false", namespaces=ns)
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if is_truncated.lower() != "true":
            return availability if with_availability else keys
        if not token:
            raise RuntimeError("S3 listing truncated without continuation token")
    raise RuntimeError("S3 listing exceeded max_pages; narrow to a single hour")


def select_goes_c13(bucket: str, when: datetime, session=None, tolerance_minutes=20,
                    return_availability=False):
    if when.tzinfo is None:
        raise ValueError("--at must include timezone")
    when = when.astimezone(timezone.utc)
    candidates = {}
    for hour in [when - timedelta(hours=1),when]:
        prefix = f"ABI-L2-CMIPC/{hour.year}/{hour.timetuple().tm_yday:03d}/{hour.hour:02d}/"
        candidates.update(list_public_s3(bucket, prefix, session=session, with_availability=True))
    availability = candidates
    candidates = [k for k, available in availability.items()
                  if 'C13_' in k and k.endswith('.nc')
                  and scan_start_from_key(k) <= when and available <= when]
    if not candidates:
        raise ValueError("No GOES Band 13 CONUS scene available by requested cutoff")
    best = min(candidates,key=lambda k: abs((scan_start_from_key(k)-when).total_seconds()))
    delta = abs((scan_start_from_key(best)-when).total_seconds())
    if delta > tolerance_minutes*60:
        raise ValueError(f"Closest GOES scene is {delta/60:.1f} minutes away; adjust time")
    result = (best, scan_start_from_key(best))
    return (*result, availability[best]) if return_availability else result


def download_goes(bucket: str, key: str, outfile: Path, session=None, max_megabytes=120):
    session = session or requests.Session()
    outfile.parent.mkdir(parents=True,exist_ok=True)
    tmp = outfile.with_suffix(outfile.suffix + ".part")
    total = 0
    try:
        with session.get(f"https://{bucket}.s3.amazonaws.com/{key}",stream=True,timeout=120) as r:
            r.raise_for_status()
            with open(tmp,"wb") as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk:
                        total += len(chunk)
                        if total > max_megabytes*1024*1024:
                            raise ValueError("Download exceeded safety cap; choose smaller data")
                        f.write(chunk)
        tmp.replace(outfile)
    finally:
        if tmp.exists():
            tmp.unlink()
    return outfile


def _fractional_index(coords: np.ndarray, target: np.ndarray):
    if coords[0] > coords[-1]:
        return np.interp(target,coords[::-1],np.arange(len(coords))[::-1],left=np.nan,right=np.nan)
    return np.interp(target,coords,np.arange(len(coords)),left=np.nan,right=np.nan)


def _bilinear(data, rows,cols):
    """Bilinear sample with finite-mask propagation and strict bounds."""
    h,w = data.shape
    inside = np.isfinite(rows) & np.isfinite(cols)
    r = np.nan_to_num(rows, nan=-1.)
    c = np.nan_to_num(cols, nan=-1.)
    inside &= (r>=0)&(c>=0)&(r<=h-1)&(c<=w-1)
    r0=np.clip(np.floor(r).astype(int),0,h-1)
    c0=np.clip(np.floor(c).astype(int),0,w-1)
    r1=np.clip(r0+1,0,h-1)
    c1=np.clip(c0+1,0,w-1)
    wr=r-r0
    wc=c-c0
    v=(data[r0,c0]*(1-wr)*(1-wc)+data[r0,c1]*(1-wr)*wc
       +data[r1,c0]*wr*(1-wc)+data[r1,c1]*wr*wc)
    return v, inside & np.isfinite(v)


def reproject_goes_netcdf(nc_path: Path, out_npz: Path, *, bbox, height=32,width=32,
                          source_url="", scan_time_utc="", available_time_utc="") -> dict:
    try:
        import xarray as xr
        from pyproj import CRS, Transformer
    except ImportError as e:
        raise RuntimeError('Missing satellite dependencies: pip install -e ".[all]"') from e
    lat,lon = grid_coordinates(bbox,height,width)
    with xr.open_dataset(nc_path, mask_and_scale=True) as ds:
        if not all(v in ds for v in ("CMI","DQF","goes_imager_projection")):
            raise ValueError("Expected NOAA ABI-L2-CMIP CMI/DQF/geostationary projection")
        a=ds["goes_imager_projection"].attrs
        required=("perspective_point_height","longitude_of_projection_origin",
                  "semi_major_axis","semi_minor_axis")
        if not all(k in a for k in required):
            raise ValueError("Incomplete GOES projection metadata")
        height_m=float(a["perspective_point_height"])
        geos_crs = CRS.from_proj4(
            f'+proj=geos +h={height_m} +lon_0={a["longitude_of_projection_origin"]} '
            f'+a={a["semi_major_axis"]} +b={a["semi_minor_axis"]} '
            f'+sweep={a.get("sweep_angle_axis","x")} +units=m +no_defs')
        transformer = Transformer.from_crs("EPSG:4326",geos_crs,always_xy=True)
        x_m,y_m=transformer.transform(lon,lat)
        cols=_fractional_index(ds["x"].values, x_m/height_m)
        rows=_fractional_index(ds["y"].values, y_m/height_m)
        cmi=np.asarray(ds["CMI"].values,dtype="float32")
        quality=np.asarray(ds["DQF"].values)
        # Every contributor to bilinear interpolation must have DQF=0.
        cmi = np.where(quality == 0, cmi, np.nan)
        sampled,inside=_bilinear(cmi,rows,cols)
        # An output cell is considered high quality only when the closest input cell has DQF=0.
        ri=np.clip(np.rint(np.nan_to_num(rows,nan=0)).astype(int),0,quality.shape[0]-1)
        ci=np.clip(np.rint(np.nan_to_num(cols,nan=0)).astype(int),0,quality.shape[1]-1)
        good = inside & (quality[ri,ci] == 0)
        units = ds["CMI"].attrs.get("units", "unknown")
        if str(units) != 'K':
            raise ValueError(f'Expected C13 brightness temperature in K, got {units}')
        coverage_start = ds.attrs.get('time_coverage_start', '')
        coverage_end = ds.attrs.get('time_coverage_end', '')
        if coverage_start and scan_time_utc:
            from .timeutil import parse_utc
            if abs((parse_utc(coverage_start) - parse_utc(scan_time_utc)).total_seconds()) > 1:
                raise ValueError('GOES NetCDF time differs from filename scan time')
        band = int(np.asarray(ds["band_id"].values).item()) if "band_id" in ds else None
        if band is not None and band != 13:
            raise ValueError(f"Expected channel 13, but file band_id={band}")
    out_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_npz,
        CMI=np.where(good,sampled,0).astype("float32"),
        quality_mask=good, latitude=lat.astype("float32"),
        longitude=lon.astype("float32"), scan_time_utc=np.array(scan_time_utc),
        channel=np.array(13),units=np.array(str(units)),source_url=np.array(source_url),
        available_time_utc=np.array(available_time_utc),
        coverage_end_utc=np.array(coverage_end))
    info={"source_nc":str(nc_path), "source_url":source_url, "channel":13,
          "scan_time_utc":scan_time_utc,"shape":[height,width],
          "available_time_utc":available_time_utc,"coverage_start_utc":coverage_start,
          "coverage_end_utc":coverage_end,
          "units":str(units),"valid_fraction":float(good.mean()),
          "bbox_south_west_north_east":list(bbox),
          "label":"GOES C13 brightness temperature / imagery; NOT precipitation"}
    out_npz.with_suffix(".json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    return info
