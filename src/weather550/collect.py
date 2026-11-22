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


# Continuous collection keeps the existing source decoders and their safeguards.
# Only converted 32x32 scenes and status records cross the process boundary.
_STUDY_SESSION = None


def retry_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(
        total=3,backoff_factor=2,status_forcelist=[429,500,502,503,504],
        allowed_methods=['GET','HEAD'],respect_retry_after_header=True)))
    return session


def _valid_scene(path,kind,cutoff):
    import numpy as np
    from .dataset import load_grid
    from .numerical import grid_coordinates
    field,timefield=('CMI','scan_time_utc') if kind=='goes' else ('precip_mm','valid_end_utc')
    item=load_grid(Path(path),field,timefield)
    lat,lon=grid_coordinates((39.2,-75.3,41.2,-73.0),32,32)
    if item['values'].shape!=(32,32) or item['mask'].shape!=(32,32):return False
    if not np.allclose(item['lat'],lat,atol=1e-4,rtol=0) or not np.allclose(item['lon'],lon,atol=1e-4,rtol=0):return False
    if kind=='mrms':return item['time']==cutoff
    return item['available'] is not None and item['time']<=item['available']<=cutoff and cutoff-item['time']<=timedelta(minutes=20)


def _study_source(job):
    global _STUDY_SESSION
    kind,text,cached=job
    moment=parse_utc(text)
    record={'kind':kind,'time':text}
    try:
        if cached:
            try:
                if _valid_scene(cached,kind,moment):
                    return {**record,'ok':True,'file':str(cached),'cached':True}
            except (ValueError,OSError,KeyError):pass
        if _STUDY_SESSION is None:_STUDY_SESSION=retry_session()
        if kind=='goes':
            key,scan,available=select_goes_c13('noaa-goes19',moment,session=_STUDY_SESSION,
                return_availability=True,cache_dir=Path('data/raw/goes_listings'))
            raw=Path('data/raw/goes')/Path(key).name
            if not raw.exists():download_goes('noaa-goes19',key,raw,session=_STUDY_SESSION)
            out=Path('data/processed/goes')/('GOES19_C13_'+scan.strftime('%Y%m%dT%H%M%SZ')+'.npz')
            if not out.exists() or not _valid_scene(out,kind,moment):
                reproject_goes_netcdf(raw,out,bbox=(39.2,-75.3,41.2,-73.0),height=32,width=32,
                    source_url=f'https://noaa-goes19.s3.amazonaws.com/{key}',
                    scan_time_utc=iso_utc(scan),available_time_utc=iso_utc(available))
        else:
            result=fetch_mrms(moment,Path('data/raw/mrms'),Path('data/processed/mrms'),
                              height=32,width=32,session=_STUDY_SESSION)
            out=Path(result.get('processed',result.get('file')))
        if not _valid_scene(out,kind,moment):raise ValueError('Converted scene failed validation')
        return {**record,'ok':True,'file':str(out),'cached':False}
    except Exception as exc:
        status=getattr(getattr(exc,'response',None),'status_code',None)
        return {**record,'ok':False,'reason':f'http_{status}' if status else type(exc).__name__,'error':str(exc)}


def source_records(path=Path('data/processed/study_sources.jsonl')):
    records={}
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try:
                item=json.loads(line);records[(item['kind'],item['time'])]=item
            except (ValueError,KeyError):continue # interrupted final append is retried
    return records


def study_hours(start,end):
    start,end=parse_utc(start),parse_utc(end)
    if start>=end or any((t.minute or t.second or t.microsecond) for t in (start,end)):
        raise ValueError('Study bounds must be increasing exact UTC hours')
    hours=[]
    while start<end:hours.append(start);start+=timedelta(hours=1)
    return hours


def fetch_study(start,end,*,workers=4,retry_failed=False):
    import shutil
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from .dataset import load_grid
    if workers not in range(1,5):raise ValueError('Use 1-4 bounded source workers')
    hours=study_hours(start,end)
    estimated_bytes=int(len(hours)*8*1024**2+2*1024**3)
    if shutil.disk_usage('.').free<estimated_bytes:raise RuntimeError(f'Need approximately {estimated_bytes} free bytes before collection')
    journal=Path('data/processed/study_sources.jsonl');journal.parent.mkdir(parents=True,exist_ok=True)
    records=source_records(journal)
    # Catalog existing verified scenes once, preserving the original selected scans.
    cached_goes=[]
    for path in sorted(Path('data/processed/goes').glob('*.npz')):
        try:
            item=load_grid(path,'CMI','scan_time_utc');cached_goes.append(item)
        except (ValueError,OSError,KeyError):continue
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(hours),24):
            day=hours[offset:offset+24]
            needed={('goes',iso_utc(t-timedelta(hours=h))) for t in day for h in (0,1,2)}
            needed.update(('mrms',iso_utc(t+timedelta(hours=h))) for t in day for h in (1,3))
            jobs=[]
            for kind,text in sorted(needed):
                old=records.get((kind,text))
                if old and not old['ok'] and not retry_failed:continue
                cached=old.get('file') if old and old['ok'] else None
                if not cached and kind=='mrms':
                    p=Path('data/processed/mrms')/f'MRMS_QPE01H_{parse_utc(text):%Y%m%dT%H%M%SZ}.npz'
                    if p.exists():cached=str(p)
                elif not cached:
                    t=parse_utc(text)
                    eligible=[q for q in cached_goes if q['available'] is not None and q['time']<=q['available']<=t and t-q['time']<=timedelta(minutes=20)]
                    if eligible:cached=max(eligible,key=lambda q:q['time'])['file']
                if old and old['ok'] and cached:
                    try:
                        if _valid_scene(cached,kind,parse_utc(text)):continue
                    except (ValueError,OSError,KeyError):pass
                jobs.append((kind,text,cached))
            futures=[pool.submit(_study_source,job) for job in jobs]
            for future in as_completed(futures):
                result=future.result();records[(result['kind'],result['time'])]=result
                with journal.open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(result)+'\n');handle.flush()
            print(json.dumps({'day':iso_utc(day[0]),'source_records':len(records),
                  'source_failures':sum(not r['ok'] for r in records.values())}),flush=True)
            if shutil.disk_usage('.').free<1024**3:raise RuntimeError('Less than 1 GiB remains; progress retained')
    return records



def collect_study_asos(start,end):
    import pandas as pd
    import time
    from .asos import download_asos,convert_asos
    start=parse_utc(start)-timedelta(hours=3.5)
    start=start.replace(hour=0,minute=0,second=0,microsecond=0)
    end=parse_utc(end)
    if end.hour or end.minute:end=end.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
    results=[];frames=[];session=retry_session()
    while start<end:
        stop=min(start+timedelta(days=8),end)
        raw=Path(f'data/raw/study_asos_{start:%Y%m%d}.csv')
        out=Path(f'data/processed/study_stations_{start:%Y%m%d}.csv')
        cached=raw.exists()
        record={'start':start.isoformat(),'end':stop.isoformat(),'file':str(out)}
        try:
            download_asos(['TTN','EWR'],start,stop,raw,session=session)
            summary=convert_asos(raw,out);frames.append(pd.read_csv(out))
            record.update(ok=True,records=summary['records'])
        except Exception as exc:record.update(ok=False,error=f'{type(exc).__name__}: {exc}')
        results.append(record)
        Path('data/processed/study_asos_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        start=stop
        if not cached:time.sleep(3)
    if not frames:raise ValueError('No study ASOS records available; see study_asos_audit.json')
    frame=pd.concat(frames,ignore_index=True).sort_values(['timestamp_utc','station']).drop_duplicates(['timestamp_utc','station'],keep='last')
    frame.to_csv('data/processed/stations_study.csv',index=False)
    return results
