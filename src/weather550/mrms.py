"""MRMS MultiSensor_QPE_01H_Pass2: archived, *future* one-hour rain totals.

Uses the publicly browseable IEM historical MRMS archive.  A GRIB2 file
ending at t+60 minutes is the label for the hour (t, t+60]; similarly
one ending at t+180 is for (t+120, t+180].  These are NOT 3-hour sums.

For precise geolocation, regular latitude/longitude grid metadata and scanning
flags are validated. No CONUS origin or direction is hardcoded. Process one
file at a time; the global value array uses roughly 200 MB.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import gzip
import json
import os

import numpy as np
import requests
from .numerical import grid_coordinates
from .timeutil import parse_utc, iso_utc

PRODUCT = 'MultiSensor_QPE_01H_Pass2'
BASE = 'https://mtarchive.geol.iastate.edu'


def mrms_archive_url(moment: datetime) -> str:
    """Exact on-the-hour UTC archive file; fail closed otherwise."""
    if moment.tzinfo is None:
        raise ValueError('MRMS request timestamp requires UTC timezone')
    t = moment.astimezone(timezone.utc)
    if t.minute or t.second or t.microsecond:
        raise ValueError('MRMS 1-hour QPE samples must end at exact UTC hour')
    return (f'{BASE}/{t:%Y/%m/%d}/mrms/ncep/{PRODUCT}/'
            f'{PRODUCT}_00.00_{t:%Y%m%d-%H%M%S}.grib2.gz')


def download_mrms(moment: datetime, dest: Path, session=None, max_mb=20) -> dict:
    url = mrms_archive_url(moment)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 30:
        with dest.open('rb') as f:
            if f.read(2) == b'\x1f\x8b':
                return {'file':str(dest),'url':url,'cached':True}
        raise ValueError(f'Cached MRMS file is not gzip: {dest}')
    session = session or requests.Session()
    part = dest.with_suffix(dest.suffix+'.part')
    count = 0
    try:
        with session.get(url,stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            with part.open('wb') as f:
                for chunk in response.iter_content(1024*512):
                    if chunk:
                        count += len(chunk)
                        if count > max_mb*1024*1024:
                            raise ValueError('MRMS download safety cap exceeded')
                        f.write(chunk)
        with part.open('rb') as f:
            if f.read(2) != b'\x1f\x8b':
                raise ValueError('MRMS archive responded with non-gzip content')
        os.replace(part,dest)
    finally:
        if part.exists(): part.unlink()
    return {'file':str(dest),'url':url,'cached':False}


def _fractional(source_axis, target):
    axis = np.asarray(source_axis, dtype=np.float64)
    if len(axis) < 2 or not np.all(np.isfinite(axis)):
        raise ValueError('Invalid GRIB coordinate axis')
    if not (np.all(np.diff(axis)>0) or np.all(np.diff(axis)<0)):
        raise ValueError('MRMS grid is not regular in expected direction')
    idx = np.arange(len(axis), dtype=float)
    if axis[0] > axis[-1]: axis,idx = axis[::-1],idx[::-1]
    return np.interp(target,axis,idx,left=np.nan,right=np.nan)


def sample_regular_grid(values, source_lat, source_lon, target_lat, target_lon):
    """Nearest-neighbor sample on validated rectilinear lat/lon grid.

    Returns zeros and False masks for out-of-bounds and missing values. No
    precipitation value may be invented for a missing source pixel.
    """
    values=np.asarray(values, dtype=float)
    slat=np.asarray(source_lat,dtype=float)
    slon=np.asarray(source_lon,dtype=float)
    if values.shape != (len(slat),len(slon)):
        raise ValueError('MRMS values and coordinate axes disagree')
    r=_fractional(slat,target_lat)
    c=_fractional(slon,target_lon)
    # Longitude is sometimes stored 0..360 in GRIB.
    if np.nanmax(slon)>180:
        c=_fractional(slon, np.mod(target_lon,360))
    good=np.isfinite(r)&np.isfinite(c)
    ri=np.clip(np.rint(np.nan_to_num(r,nan=-1)).astype(int),0,values.shape[0]-1)
    ci=np.clip(np.rint(np.nan_to_num(c,nan=-1)).astype(int),0,values.shape[1]-1)
    sampled=values[ri,ci]
    good &= np.isfinite(sampled) & (sampled>=0)
    return np.where(good,sampled,0).astype(np.float32),good


def decode_mrms_grib(gz_file: Path, out_npz: Path, *, valid_end_utc: str,
                     bbox=(39.2,-75.3,41.2,-73.0), height=64,width=64,
                     source_url='') -> dict:
    """Decode actual GRIB2 via ecCodes, reject unexpected units and geometry.

    GRIB geometry avoids hardcoded grid orientation; decode one hour at a time.
    """
    try:
        import eccodes as ec
    except ImportError as e:
        raise RuntimeError('Install radar extra: python -m pip install -e ".[all]"') from e
    valid = parse_utc(valid_end_utc)
    target_lat,target_lon=grid_coordinates(bbox,height,width)
    # ecCodes' file API reads the underlying descriptor, bypassing gzip decoding.
    with gzip.open(gz_file, 'rb') as handle:
        message = handle.read(256 * 1024 * 1024 + 1)
    if len(message) > 256 * 1024 * 1024:
        raise ValueError('Decompressed GRIB exceeds 256 MiB safety cap')
    if not message.startswith(b'GRIB'):
        raise ValueError('No GRIB message in MRMS file')
    gid = ec.codes_new_from_message(message)
    try:
        identity = tuple(int(ec.codes_get(gid, k)) for k in
                         ('discipline', 'parameterCategory', 'parameterNumber'))
        if identity != (209, 6, 37) or int(ec.codes_get(gid, 'centre')) != 161:
            raise ValueError(f'Expected NOAA MRMS {PRODUCT} (209/6/37), got {identity}')
        # NOAA local parameter table is authoritative when ecCodes has no local definition.
        # https://www.nssl.noaa.gov/projects/mrms/operational/tables.php
        units = str(ec.codes_get(gid, 'units'))
        canon = units.lower().replace(' ', '').replace('**', '^')
        if canon not in {'unknown', 'mm', 'kgm^-2', 'kgm-2', 'kg/m^2', 'kg/m2'}:
            raise ValueError(f'Unknown MRMS rainfall accumulation units: {units}')
        actual_valid = datetime.strptime(
            f"{int(ec.codes_get(gid, 'validityDate')):08d}"
            f"{int(ec.codes_get(gid, 'validityTime')):04d}", '%Y%m%d%H%M'
        ).replace(tzinfo=timezone.utc)
        if actual_valid != valid:
            raise ValueError(f'GRIB valid time {actual_valid} differs from requested {valid}')
        if str(ec.codes_get(gid, 'gridType')) != 'regular_ll':
            raise ValueError('MRMS grid must be regular latitude/longitude')
        for key in ('jPointsAreConsecutive', 'alternativeRowScanning'):
            if int(ec.codes_get(gid, key)):
                raise ValueError(f'Unsupported GRIB scanning flag {key}')
        ni, nj = (int(ec.codes_get(gid, k)) for k in ('Ni', 'Nj'))
        lat0 = float(ec.codes_get(gid, 'latitudeOfFirstGridPointInDegrees'))
        lon0 = float(ec.codes_get(gid, 'longitudeOfFirstGridPointInDegrees'))
        dy = float(ec.codes_get(gid, 'jDirectionIncrementInDegrees'))
        dx = float(ec.codes_get(gid, 'iDirectionIncrementInDegrees'))
        dy *= 1 if int(ec.codes_get(gid, 'jScansPositively')) else -1
        dx *= -1 if int(ec.codes_get(gid, 'iScansNegatively')) else 1
        slat = lat0 + np.arange(nj) * dy
        slon = lon0 + np.arange(ni) * dx
        if (not np.isclose(slat[-1], ec.codes_get(gid, 'latitudeOfLastGridPointInDegrees'), atol=1e-4, rtol=0)
                or not np.isclose(slon[-1], ec.codes_get(gid, 'longitudeOfLastGridPointInDegrees'), atol=1e-4, rtol=0)):
            raise ValueError('GRIB coordinate endpoints disagree with increments/scanning flags')
        arr = np.asarray(ec.codes_get_values(gid), dtype=float).reshape(nj, ni)
        arr[arr == float(ec.codes_get(gid, 'missingValue'))] = np.nan
        y, mask = sample_regular_grid(arr, slat, slon, target_lat, target_lon)
    finally:
        ec.codes_release(gid)
    out_npz=Path(out_npz)
    out_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_npz,precip_mm=y,quality_mask=mask,
       latitude=target_lat.astype('float32'),longitude=target_lon.astype('float32'),
       valid_end_utc=np.array(iso_utc(valid)),units=np.array('mm'),
       source_url=np.array(source_url))
    info={'valid_end_utc':iso_utc(valid),'file':str(out_npz),'shape':list(y.shape),
          'valid_fraction':float(mask.mean()),'units':'mm',
          'product':PRODUCT, 'source_url':source_url,
          'grib_parameter_identity':list(identity),'grib_reported_units':units,
          'units_authority':'https://www.nssl.noaa.gov/projects/mrms/operational/tables.php',
          'grid_geometry':'regular_ll endpoints/increments/scanning flags validated',
          'interpretation':'hourly QPE accumulation ending at valid_end_utc'}
    out_npz.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


def fetch_mrms(moment: datetime, raw_dir: Path, processed_dir: Path, *,
              bbox=(39.2,-75.3,41.2,-73.0),height=64,width=64) -> dict:
    name=f'MRMS_QPE01H_{moment:%Y%m%dT%H%M%SZ}'
    raw=Path(raw_dir)/(name+'.grib2.gz')
    processed=Path(processed_dir)/(name+'.npz')
    record=download_mrms(moment,raw)
    if processed.exists():
        return {'processed':str(processed),'cached':True,'source_url':record['url']}
    data=decode_mrms_grib(raw,processed,valid_end_utc=iso_utc(moment),
          bbox=bbox,height=height,width=width,source_url=record['url'])
    return data
