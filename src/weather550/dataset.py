"""Join physically observed weather products into causal supervised samples.

Assumes producer modules have already converted raw sources to *matching*
lat/lon grids. Uses exact/nearest-previous GOES scan semantics and exact MRMS
valid-hour targets; rejects missing data, no silent filling or leakage.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import numpy as np
from .timeutil import parse_utc, iso_utc


def load_grid(npz:Path, value_key:str, time_key:str):
    with np.load(npz,allow_pickle=False) as d:
        expected_units = 'K' if value_key == 'CMI' else 'mm'
        real_source = 'source_url' in d and bool(str(d['source_url'].item()))
        if real_source and ('units' not in d or (value_key == 'CMI' and
                ('available_time_utc' not in d or not str(d['available_time_utc'].item())))):
            raise ValueError(f'Missing units/publication metadata in real source {npz}')
        if 'units' in d and str(d['units'].item()) != expected_units:
            raise ValueError(f'Unexpected source units in {npz}; expected {expected_units}')
        values = np.asarray(d[value_key],dtype=np.float32)
        mask = np.asarray(d['quality_mask'],dtype=bool)
        if np.any(mask & (~np.isfinite(values) | ((values < 0) if value_key == 'precip_mm' else False))):
            raise ValueError(f'Invalid values marked valid in {npz}')
        return {
          'time': parse_utc(str(d[time_key].item())),
          'values': np.where(mask, values, 0),
          'mask': mask,
          'available': parse_utc(str(d['available_time_utc'].item()))
                       if 'available_time_utc' in d and str(d['available_time_utc'].item()) else None,
          'url': str(d['source_url'].item()) if 'source_url' in d else '',
          'lat': np.asarray(d['latitude'],dtype=np.float32),
          'lon': np.asarray(d['longitude'],dtype=np.float32),
          'file':str(npz),
        }


def _catalog(directory:Path, field:str, time_field:str):
    result={}
    for path in sorted(Path(directory).glob('*.npz')):
        item=load_grid(path,field,time_field)
        if item['time'] in result: raise ValueError(f'Duplicate timestamp in {directory}: {item["time"]}')
        result[item['time']]=item
    return result


def split_by_target_bounds(issuance_times, *,train_before, val_before, history_hours=2,
                           target_hours=3):
    """Split with embargo: train/validation/test *entire target intervals* must stay within split.
    Values: train, val, test or exclude. History also must not cross split start.
    """
    train_cut=parse_utc(train_before)
    val_cut=parse_utc(val_before)
    if not train_cut < val_cut: raise ValueError('train_before must precede val_before')
    labels=[]
    for t in issuance_times:
        t=parse_utc(t) if isinstance(t,str) else t
        s=t-timedelta(hours=history_hours)
        e=t+timedelta(hours=target_hours)
        if e < train_cut: label='train'
        elif s>=train_cut and e<val_cut: label='val'
        elif s>=val_cut: label='test'
        else: label='exclude'
        labels.append(label)
    return np.array(labels)


def build_supervised(numerical_file:Path, goes_dir:Path, mrms_dir:Path, out_file:Path,
        *, goes_history=3,goes_step_minutes=60, tolerance_minutes=20,
        min_goes_fraction=.8,min_mrms_fraction=.8,
        train_before=None,val_before=None) -> dict:
    """Produce N x T x C x H x W GOES, numeric N x T x F x H x W, 2 QPE targets.

    No interpolation across missing satellite scenes: this is intentionally
    strict for a research audit.  Missing samples are counted and excluded.
    """
    with np.load(numerical_file,allow_pickle=False) as n:
        numX=np.asarray(n['X_numerical'],dtype=np.float32)
        numM=np.asarray(n['X_mask'],dtype=bool)
        times=[parse_utc(str(x)) for x in n['issuance_times_utc']]
        lat=np.asarray(n['lat_grid'],dtype=np.float32)
        lon=np.asarray(n['lon_grid'],dtype=np.float32)
        names=np.asarray(n['feature_names'])
        offsets=np.asarray(n['history_offsets_minutes']) if 'history_offsets_minutes' in n else None
        latest=np.asarray(n['latest_observation_times_utc']) if 'latest_observation_times_utc' in n else None
        latency=int(n['observation_latency_minutes'].item()) if 'observation_latency_minutes' in n else 0
        station_sources=np.asarray(n['source_station_files']) if 'source_station_files' in n else np.array([])
        max_age=int(n['max_observation_age_minutes'].item()) if 'max_observation_age_minutes' in n else 90
    if numX.shape!=numM.shape or numX.shape[0]!=len(times):
        raise ValueError('Numeric arrays and timestamps misaligned')
    if times != sorted(set(times)):
        raise ValueError('Numerical issuance timestamps must be unique and chronological')
    if numX.ndim != 5 or numX.shape[1:3] != (goes_history,len(names)) or numX.shape[-2:] != lat.shape:
        raise ValueError('Numeric tensor dimensions differ from history/features/grid')
    if np.any(numM & ~np.isfinite(numX)):
        raise ValueError('Nonfinite numerical values marked valid')
    if offsets is not None and not np.array_equal(offsets,np.arange(-(goes_history-1),1)*goes_step_minutes):
        raise ValueError('Numerical and GOES history cadence differ')
    if latest is not None:
        for i,t in enumerate(times):
            for j,offset in enumerate(offsets):
                if latest[i,j] and parse_utc(str(latest[i,j])) + timedelta(minutes=latency) > t + timedelta(minutes=int(offset)):
                    raise ValueError('Numerical observation availability exceeds input cutoff')
    numX=np.where(numM,numX,0)
    goes=_catalog(goes_dir,'CMI','scan_time_utc')
    mrms=_catalog(mrms_dir,'precip_mm','valid_end_utc')
    if not goes or not mrms: raise ValueError('Missing GOES or MRMS converted scenes')
    shapes=lat.shape
    for item in [*goes.values(),*mrms.values()]:
        if item['values'].shape != shapes or item['mask'].shape != shapes:
            raise ValueError('Source grids differ in shape')
        if (item['lat'].shape != lat.shape or item['lon'].shape != lon.shape or
            not np.allclose(item['lat'],lat,atol=1e-4,rtol=0) or not np.allclose(item['lon'],lon,atol=1e-4,rtol=0)):
            raise ValueError('Source grids differ in physical coordinates')
    gtimes=sorted(goes)
    G,GM,N,NM,Y,YM,ts,trace=[],[],[],[],[],[],[],[]
    missing={'goes':0,'mrms':0,'quality':0,'numerical':0}
    from bisect import bisect_right
    for i,t in enumerate(times):
        scenes=[]
        for lag in range(goes_history-1,-1,-1):
            cutoff=t-timedelta(minutes=lag*goes_step_minutes)
            pos=bisect_right(gtimes,cutoff)-1
            while pos >= 0 and goes[gtimes[pos]]['available'] is not None and goes[gtimes[pos]]['available'] > cutoff:
                pos -= 1
            if pos<0 or (cutoff-gtimes[pos]).total_seconds()>tolerance_minutes*60:
                break
            scenes.append(goes[gtimes[pos]])
        if len(scenes)!=goes_history:
            missing['goes']+=1
            continue
        keys=[t+timedelta(hours=h) for h in (1,3)]
        if any(k not in mrms for k in keys):
            missing['mrms']+=1
            continue
        labels=[mrms[k] for k in keys]
        if not numM[i].any():
            missing['numerical']+=1
            continue
        if (min(np.mean(q['mask']) for q in scenes)<min_goes_fraction or
            min(np.mean(q['mask']) for q in labels)<min_mrms_fraction):
            missing['quality']+=1
            continue
        G.append(np.stack([q['values'][None] for q in scenes]))
        GM.append(np.stack([q['mask'][None] for q in scenes]))
        N.append(numX[i]); NM.append(numM[i]);
        Y.append(np.stack([q['values'] for q in labels]))
        YM.append(np.stack([q['mask'] for q in labels]))
        ts.append(iso_utc(t))
        trace.append({'issuance_utc':iso_utc(t),'goes_files':[q['file'] for q in scenes],
                      'rain_files':[q['file'] for q in labels],
                      'goes_scan_times_utc':[iso_utc(q['time']) for q in scenes],
                      'goes_available_times_utc':[iso_utc(q['available']) if q['available'] else '' for q in scenes],
                      'source_urls':[q['url'] for q in [*scenes,*labels]],
                      'rain_valid_end_times_utc':[iso_utc(k) for k in keys],
                      'latest_station_observation_times_utc':latest[i].tolist() if latest is not None else []})
    if not ts: raise ValueError(f'No complete samples: missing counts {missing}')
    G=np.stack(G).astype('float32'); GM=np.stack(GM)
    N=np.stack(N).astype('float32'); NM=np.stack(NM)
    Y=np.stack(Y).astype('float32'); YM=np.stack(YM)
    if np.all(~NM): raise ValueError('Every numerical feature missing; no usable numerical supervision')
    out_file=Path(out_file);out_file.parent.mkdir(parents=True,exist_ok=True)
    payload={'X_goes':G,'X_goes_mask':GM,'X_numerical':N,'X_num_mask':NM,
       'Y_rain_mm':Y,'Y_rain_mask':YM,'issuance_times_utc':np.array(ts),
       'forecast_horizons_hours':np.array([1,3]),'feature_names':names,
       'lat_grid':lat,'lon_grid':lon,
       'goes_scan_times_utc':np.array([q['goes_scan_times_utc'] for q in trace]),
       'goes_available_times_utc':np.array([q['goes_available_times_utc'] for q in trace]),
       'rain_valid_end_times_utc':np.array([q['rain_valid_end_times_utc'] for q in trace]),
       'latest_station_observation_times_utc':np.array([q['latest_station_observation_times_utc'] for q in trace]),
       'source_metadata_json':np.array(json.dumps(trace)),
       'source_station_files':station_sources,
       'max_observation_age_minutes':np.array(max_age),
       'observation_latency_minutes':np.array(latency)}
    if (train_before is None)!=(val_before is None):
        raise ValueError('Both split cutoffs are required when splitting')
    if train_before:
        payload['split']=split_by_target_bounds(ts,train_before=train_before,
          val_before=val_before,history_hours=(max(goes_history-1,numX.shape[1]-1)*goes_step_minutes+max_age)/60)
    np.savez_compressed(out_file,**payload)
    summary={'contract':'CS550-SUPERVISED-v0.3','file':str(out_file),
          'samples':len(ts),'shape_X_goes':list(G.shape),'shape_X_num':list(N.shape),
          'shape_Y_rain_mm':list(Y.shape),'targets':'hourly QPE mm ending at t+1h and t+3h',
          'skipped':missing,'total_numeric_issuance_times':len(times),
          'valid_fraction_goes':float(GM.mean()),'valid_fraction_numerical':float(NM.mean()),
          'valid_fraction_rain':float(YM.mean()),
          'rainy_samples':int(np.any((Y>0)&YM,axis=(1,2,3)).sum()),
          'dry_samples':int(np.all((Y==0)&YM,axis=(1,2,3)).sum()),
          'observation_latency_minutes':latency,
          'train_normalization_done':False,'chronological_splitting':'pass cutoff timestamps or do separately',
          'sources':trace,'notes':'MRMS Pass2 may be quality controlled after event time; targets only, never predictors'}
    out_file.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return {k:v for k,v in summary.items() if k!='sources'}
