"""Small safe orchestration of aligned GOES+MRMS samples; no huge surprise downloads."""
from datetime import timedelta
from pathlib import Path
import json
from .goes import select_goes_c13,download_goes,reproject_goes_netcdf
from .mrms import mrms_archive_url,fetch_mrms
from .timeutil import iso_utc,parse_utc


def fetch_one(moment, *,raw_dir=Path('data/raw'),processed_dir=Path('data/processed'),
              bbox=(39.2,-75.3,41.2,-73.0),height=32,width=32,list_only=False):
    if moment.minute or moment.second or moment.microsecond:
        raise ValueError('Issuance must be an exact UTC hour')
    raw_dir,processed_dir=Path(raw_dir),Path(processed_dir)
    result={'issuance_utc':iso_utc(moment),'goes':[],'mrms':[]}
    for lag in (120,60,0):
        key,scan,available=select_goes_c13('noaa-goes19',moment-timedelta(minutes=lag),
                                         return_availability=True)
        url=f'https://noaa-goes19.s3.amazonaws.com/{key}'
        item={'scan_time_utc':iso_utc(scan),'available_time_utc':iso_utc(available),'url':url}
        if not list_only:
            raw=raw_dir/'goes'/Path(key).name
            if not raw.exists():download_goes('noaa-goes19',key,raw)
            out=processed_dir/'goes'/('GOES19_C13_'+scan.strftime('%Y%m%dT%H%M%SZ')+'.npz')
            reproject_goes_netcdf(raw,out,bbox=bbox,height=height,width=width,
                   source_url=url,scan_time_utc=iso_utc(scan),available_time_utc=iso_utc(available))
            item['processed_file']=str(out)
        result['goes'].append(item)
    for lead in (1,3):
        target=moment+timedelta(hours=lead)
        item={'valid_end_utc':iso_utc(target),'url':mrms_archive_url(target)}
        if not list_only:
            result_m=fetch_mrms(target,raw_dir/'mrms',processed_dir/'mrms',
               bbox=bbox,height=height,width=width)
            item['processed_file']=str(result_m.get('processed',result_m.get('file')))
        result['mrms'].append(item)
    return result


def fetch_batch(start,end, *,step_hours=1,max_issuances=12,
                continue_on_error=False,**kwargs):
    s,e=parse_utc(start),parse_utc(end)
    if s>=e or step_hours<1 or max_issuances<1 or max_issuances>36:
        raise ValueError('Invalid range or batch cap. At most 36 hourly issuances per run.')
    current=s
    attempted=0
    records=[]
    while current<e and attempted<max_issuances:
        attempted+=1
        try:
            res=fetch_one(current,**kwargs)
            records.append({'time':iso_utc(current),'ok':True,'result':res})
        except Exception as exc:
            records.append({'time':iso_utc(current),'ok':False,'error':str(exc)})
            if not continue_on_error:
                break
        current+=timedelta(hours=step_hours)
    return {'records':records,'attempted':attempted,'ok':sum(r['ok'] for r in records),
            'failures':sum(not r['ok'] for r in records),
            'remaining_unattempted':current<e and attempted>=max_issuances}
