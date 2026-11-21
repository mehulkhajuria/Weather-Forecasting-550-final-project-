"""Assemble configured bounded weather periods without normalization or model training."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from weather550.numerical import build_numerical
from weather550.dataset import build_supervised
from weather550.timeutil import parse_utc
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]

def configured_periods(path):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    periods=config['periods']
    ids=set(); ticks=set()
    for period in periods:
        if period['event_id'] in ids or period['kind'] not in ('rainy','dry'):
            raise ValueError('Event IDs must be unique and kind must be rainy or dry')
        ids.add(period['event_id'])
        start,end=parse_utc(period['start']),parse_utc(period['end'])
        if start.minute or start.second or end.minute or end.second or not timedelta(0)<end-start<=timedelta(hours=36):
            raise ValueError('Each period must span 1-36 exact UTC hours')
        while start<end:
            if start in ticks: raise ValueError('Overlapping issuance periods')
            ticks.add(start);start+=timedelta(hours=1)
    if not periods: raise ValueError('No periods configured')
    rainy=sorted((parse_utc(p['start'])-timedelta(hours=3.5),parse_utc(p['end'])+timedelta(hours=2))
                 for p in periods if p['kind']=='rainy')
    if any(b[0]-a[1]<timedelta(hours=48) for a,b in zip(rainy,rainy[1:])):
        raise ValueError('Rainy event groups need at least 48 hours between complete windows')
    return config


def collect_periods(periods):
    from weather550.asos import download_asos, convert_asos
    from weather550.collect import fetch_one, fetch_batch
    from weather550.mrms import mrms_archive_url
    import requests
    audit=[]
    for period in periods:
        record={'event_id':period['event_id']}
        try:
            start,end=parse_utc(period['start']),parse_utc(period['end'])
            record['preflight']=fetch_one(start,list_only=True)
            targets=set()
            tick=start
            while tick<end:
                targets.update(tick+timedelta(hours=h) for h in (1,3))
                tick+=timedelta(hours=1)
            for target in sorted(targets):
                response=requests.head(mrms_archive_url(target),timeout=40)
                response.raise_for_status()
            raw=Path(period['station_raw'])
            download_asos(['TTN','EWR'],parse_utc(period['station_start']+'T00:00:00Z'),
                          parse_utc(period['station_end']+'T00:00:00Z'),raw)
            record['station_summary']=convert_asos(raw,Path(period['station_file']))
            record['batch']=fetch_batch(period['start'],period['end'],max_issuances=36,continue_on_error=True)
            record['status']='complete' if record['batch']['failures']==0 else 'partial'
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        audit.append(record)
        Path('data/processed/fetch_audit.json').write_text(json.dumps({'events':audit},indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k not in ('preflight','station_summary','batch')}),flush=True)
    return audit


def main(argv=None):
    import os
    os.chdir(ROOT)
    out = Path('data/processed')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--periods',type=Path,default=Path('scripts/collection_periods.json'))
    parser.add_argument('--fetch',action='store_true',help='Check availability and download bounded configured periods first')
    args=parser.parse_args(argv)
    config=configured_periods(args.periods)
    periods=config['periods']
    if args.fetch: collect_periods(periods)
    parts=[]
    assembly_errors=[]
    for period in periods:
        path=out/f"numeric_{period['event_id']}.npz"
        start=parse_utc(period['start'])-timedelta(hours=2)
        try:
            build_numerical(Path(period['station_file']),path,start.isoformat(),period['end'],latency_minutes=10)
            with np.load(path,allow_pickle=False) as d: parts.append({k:d[k] for k in d.files})
        except (ValueError,FileNotFoundError) as exc:
            assembly_errors.append({'event_id':period['event_id'],'error':str(exc)})
    if not parts: raise ValueError(f'No numerical periods available: {assembly_errors}')
    merged=parts[0].copy()
    for key in ['X_numerical','X_mask','nearest_station_km','issuance_times_utc','latest_observation_times_utc']:
        merged[key]=np.concatenate([part[key] for part in parts])
    for key in ['feature_names','lat_grid','lon_grid','history_offsets_minutes','observation_latency_minutes']:
        assert np.array_equal(parts[0][key],parts[1][key]), key
    order=np.argsort(merged['issuance_times_utc'])
    for key in ['X_numerical','X_mask','nearest_station_km','issuance_times_utc','latest_observation_times_utc']:
        merged[key]=merged[key][order]
    merged['source_station_files']=np.array(sorted({p['station_file'] for p in periods}))
    np.savez_compressed(out/'model_numeric.npz',**merged)
    (out/'model_numeric.json').write_text(json.dumps({
        'shape_X_numerical':list(merged['X_numerical'].shape),
        'source_station_files':merged['source_station_files'].tolist(),
        'history_offsets_minutes':merged['history_offsets_minutes'].tolist(),
        'observation_latency_minutes':10,'normalization_done':False},indent=2),encoding='utf-8')
    summary=build_supervised(out/'model_numeric.npz',out/'goes',out/'mrms',out/'supervised.npz')
    with np.load(out/'supervised.npz',allow_pickle=False) as d:
        count=len(d['issuance_times_utc'])
        assert d['X_goes'].shape == (count,3,1,32,32)
        assert d['X_numerical'].shape == (count,3,5,32,32)
        assert d['Y_rain_mm'].shape == (count,2,32,32)
        assert len(set(d['issuance_times_utc']))==count
        assert list(d['issuance_times_utc'])==sorted(d['issuance_times_utc'])
        assert d['feature_names'].tolist()==['temperature_c','dewpoint_c','humidity_pct','wind_speed_mps','pressure_hpa']
        assert d['forecast_horizons_hours'].tolist()==[1,3]
        for value,mask in [('X_goes','X_goes_mask'),('X_numerical','X_num_mask'),('Y_rain_mm','Y_rain_mask')]:
            assert np.isfinite(d[value]).all()
            assert (d[value][~d[mask]] == 0).all()
        assert (d['Y_rain_mm'][d['Y_rain_mask']]>=0).all()
        for i,text in enumerate(d['issuance_times_utc']):
            t=parse_utc(str(text))
            for j,lag in enumerate([120,60,0]):
                cutoff=t-timedelta(minutes=lag)
                scan=parse_utc(str(d['goes_scan_times_utc'][i,j]))
                published=parse_utc(str(d['goes_available_times_utc'][i,j]))
                assert scan <= published <= cutoff
                text=str(d['latest_station_observation_times_utc'][i,j])
                if text:
                    obs=parse_utc(text)
                    assert cutoff-timedelta(minutes=90)<=obs<=cutoff-timedelta(minutes=10)
            for j,lead in enumerate([1,3]):
                assert parse_utc(str(d['rain_valid_end_times_utc'][i,j])) == t+timedelta(hours=lead)
        summary['issuance_times_utc']=d['issuance_times_utc'].tolist()
        summary['valid_fraction_per_numerical_feature']={str(f):float(d['X_num_mask'][:,:,j].mean()) for j,f in enumerate(d['feature_names'])}
        summary['goes_valid_range_kelvin']=[float(d['X_goes'][d['X_goes_mask']].min()),float(d['X_goes'][d['X_goes_mask']].max())]
        summary['rain_valid_range_mm']=[float(d['Y_rain_mm'][d['Y_rain_mask']].min()),float(d['Y_rain_mm'][d['Y_rain_mask']].max())]
        summary['rainy_definition']='Any valid target cell > 0 mm across either horizon; dry requires all cells valid and zero.'
    summary['availability_limitation']='IEM lacks historical receipt timestamps. Ten-minute reporting latency is an explicit retrospective assumption, not measured receipt. GOES uses S3 LastModified.'
    with np.load(out/'supervised.npz',allow_pickle=False) as d:
        payload={k:d[k] for k in d.files}
    labels=[];event_summary=[]
    for text in payload['issuance_times_utc']:
        moment=parse_utc(str(text))
        matches=[p for p in periods if parse_utc(p['start'])<=moment<parse_utc(p['end'])]
        if len(matches)!=1: raise ValueError('Issuance has ambiguous event membership')
        labels.append(matches[0]['event_id'])
    payload['event_ids']=np.array(labels)
    for period in periods:
        keep=payload['event_ids']==period['event_id']
        rain=payload['Y_rain_mm'][keep];mask=payload['Y_rain_mask'][keep]
        rainy=bool(keep.any() and np.any((rain>0)&mask))
        dry=bool(keep.any() and np.all(mask) and np.all(rain==0))
        event_summary.append({**period,'samples':int(keep.sum()),'verified_rainy':rainy,
                              'verified_dry':dry,'status':'verified' if keep.any() else 'no_complete_samples'})
    np.savez_compressed(out/'supervised.npz',**payload)
    summary['events']=event_summary
    summary['assembly_errors']=assembly_errors
    if Path('data/processed/fetch_audit.json').exists():
        summary['collection_audit']=json.loads(Path('data/processed/fetch_audit.json').read_text(encoding='utf-8'))
    summary['event_definition']=config['event_definition']
    summary['verified_rainy_events']=sum(e['kind']=='rainy' and e['verified_rainy'] for e in event_summary)
    summary['verified_dry_periods']=sum(e['kind']=='dry' and e['verified_dry'] for e in event_summary)
    summary['study_coverage']='Configured verified periods within August-October 2026; not continuous seasonal coverage.'
    summary['pilot_reference']={'file':'data/processed/pilot_supervised.npz','sha256':'26408cb7d8b4188c47bbb8daa9d292d1c227373d8888f296be43b282d4952d0c','manifest':'docs/PILOT_DATA_VERIFICATION.json'}
    companion=json.loads((out/'supervised.json').read_text(encoding='utf-8'))
    companion.update({k:summary[k] for k in ['events','event_definition','verified_rainy_events','verified_dry_periods','pilot_reference']})
    (out/'supervised.json').write_text(json.dumps(companion,indent=2),encoding='utf-8')
    summary['evaluation_limitation']='Selected event subset; event count alone does not establish evaluation reliability. No evaluation or held-out-storm performance claimed. No scaler fitted or model trained.'
    files=sorted(Path('data/raw').rglob('*.nc'))+sorted(Path('data/raw').rglob('*.gz'))+sorted(Path('data/raw').glob('*.csv'))
    def digest(path):
        with path.open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()
    summary['raw_files']=[{'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    summary['dataset_bytes']=(out/'supervised.npz').stat().st_size
    summary['dataset_sha256']=hashlib.sha256((out/'supervised.npz').read_bytes()).hexdigest()
    Path('docs/REAL_DATA_VERIFICATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(out/'teammate_handoff.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for path in [out/'supervised.npz',out/'supervised.json',
                     Path('README.md'),Path('docs/REAL_DATA_VERIFICATION.json'),
                     Path('docs/RAW_ALIGNMENT_VERIFICATION.json'),Path('docs/PILOT_DATA_VERIFICATION.json'),
                     args.periods,Path('data/processed/fetch_audit.json'),
                     Path('scripts/load_dataset.py')]:
            if path.exists(): bundle.write(path,path.as_posix() if path.parent == Path('docs') else path.name)
    print(json.dumps({k:v for k,v in summary.items() if k!='raw_files'},indent=2))

if __name__=='__main__':main()
