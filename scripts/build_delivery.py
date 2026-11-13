"""Assemble only the two bounded, verified October periods; no downloads or fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
from weather550.numerical import build_numerical
from weather550.dataset import build_supervised
from weather550.timeutil import parse_utc
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]

def main():
    import os
    os.chdir(ROOT)
    out = Path('data/processed')
    periods = [('rainy','stations_oct10_18.csv','2026-10-11T12:00:00Z','2026-10-11T20:00:00Z'),
               ('dry','stations_utc.csv','2026-10-20T13:00:00Z','2026-10-20T21:00:00Z')]
    parts=[]
    for name,stations,start,end in periods:
        path=out/f'numeric_{name}.npz'
        build_numerical(out/stations,path,start,end,latency_minutes=10)
        with np.load(path,allow_pickle=False) as d: parts.append({k:d[k] for k in d.files})
    merged=parts[0].copy()
    for key in ['X_numerical','X_mask','nearest_station_km','issuance_times_utc','latest_observation_times_utc']:
        merged[key]=np.concatenate([part[key] for part in parts])
    for key in ['feature_names','lat_grid','lon_grid','history_offsets_minutes','observation_latency_minutes']:
        assert np.array_equal(parts[0][key],parts[1][key]), key
    merged['source_station_files']=np.array([str(out/stations) for _,stations,_,_ in periods])
    np.savez_compressed(out/'model_numeric.npz',**merged)
    (out/'model_numeric.json').write_text(json.dumps({
        'shape_X_numerical':list(merged['X_numerical'].shape),
        'source_station_files':merged['source_station_files'].tolist(),
        'history_offsets_minutes':merged['history_offsets_minutes'].tolist(),
        'observation_latency_minutes':10,'normalization_done':False},indent=2),encoding='utf-8')
    summary=build_supervised(out/'model_numeric.npz',out/'goes',out/'mrms',out/'supervised.npz')
    with np.load(out/'supervised.npz',allow_pickle=False) as d:
        assert d['X_goes'].shape == (12,3,1,32,32)
        assert d['X_numerical'].shape == (12,3,5,32,32)
        assert d['Y_rain_mm'].shape == (12,2,32,32)
        for value,mask in [('X_goes','X_goes_mask'),('X_numerical','X_num_mask'),('Y_rain_mm','Y_rain_mask')]:
            assert np.isfinite(d[value]).all()
            assert (d[value][~d[mask]] == 0).all()
        assert d['Y_rain_mask'].all() and d['X_goes_mask'].all()
        for i,text in enumerate(d['issuance_times_utc']):
            t=parse_utc(str(text))
            for j,lag in enumerate([120,60,0]):
                cutoff=t-timedelta(minutes=lag)
                scan=parse_utc(str(d['goes_scan_times_utc'][i,j]))
                published=parse_utc(str(d['goes_available_times_utc'][i,j]))
                assert scan <= published <= cutoff
                obs=parse_utc(str(d['latest_station_observation_times_utc'][i,j]))
                assert obs+timedelta(minutes=10) <= cutoff
            for j,lead in enumerate([1,3]):
                assert parse_utc(str(d['rain_valid_end_times_utc'][i,j])) == t+timedelta(hours=lead)
        summary['issuance_times_utc']=d['issuance_times_utc'].tolist()
        summary['valid_fraction_per_numerical_feature']={str(f):float(d['X_num_mask'][:,:,j].mean()) for j,f in enumerate(d['feature_names'])}
        summary['goes_valid_range_kelvin']=[float(d['X_goes'][d['X_goes_mask']].min()),float(d['X_goes'][d['X_goes_mask']].max())]
        summary['rain_valid_range_mm']=[float(d['Y_rain_mm'][d['Y_rain_mask']].min()),float(d['Y_rain_mm'][d['Y_rain_mask']].max())]
        summary['rainy_definition']='Any valid target cell > 0 mm across either horizon; dry requires all cells valid and zero.'
    summary['availability_limitation']='IEM lacks historical receipt timestamps. Ten-minute reporting latency is an explicit retrospective assumption, not measured receipt. GOES uses S3 LastModified.'
    summary['study_coverage']='Only October 11 and 20, 2026 matched; August-October season not exhaustively verified.'
    summary['evaluation_limitation']='Integration subset with one main rain event; insufficient for model evaluation or held-out-storm claims. No scaler fitted or model trained.'
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
                     Path('docs/REAL_DATA_HANDOFF.md'),Path('docs/REAL_DATA_VERIFICATION.json'),
                     Path('scripts/load_dataset.py')]:
            bundle.write(path,path.name)
    print(json.dumps({k:v for k,v in summary.items() if k!='raw_files'},indent=2))

if __name__=='__main__':main()
