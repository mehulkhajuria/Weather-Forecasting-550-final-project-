"""Check three grid locations per retained source against original GRIB/NetCDF data."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np

_GEOMETRY = {}


def _verify_source(job):
    kind,text,raw_text,raw_hash,processed_hash=job
    path,raw_path=Path(text),Path(raw_text)
    checks=[]
    with np.load(path,allow_pickle=False) as d:
        height,width=d['quality_mask'].shape
        pixels=[(0,0),(height//2-1,width//2-1),(height-1,width-1)]
        if kind=='mrms':
            import eccodes as ec
            with gzip.open(raw_path,'rb') as handle:message=handle.read(256*1024*1024+1)
            if len(message)>256*1024*1024:raise ValueError('GRIB safety cap exceeded')
            g=ec.codes_new_from_message(message)
            try:
                assert tuple(int(ec.codes_get(g,k)) for k in ['centre','discipline','parameterCategory','parameterNumber'])==(161,209,6,37)
                assert d['units'].item()=='mm' and ec.codes_get(g,'gridType')=='regular_ll'
                keys=['Ni','Nj','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees',
                      'latitudeOfLastGridPointInDegrees','longitudeOfLastGridPointInDegrees',
                      'iDirectionIncrementInDegrees','jDirectionIncrementInDegrees',
                      'iScansNegatively','jScansPositively','jPointsAreConsecutive','alternativeRowScanning']
                targets=tuple((float(d['latitude'][r,c]),float(d['longitude'][r,c])) for r,c in pixels)
                geometry=tuple(ec.codes_get(g,k) for k in keys)+targets
                if geometry not in _GEOMETRY:
                    candidates=[]
                    for lat,lon in targets:
                        neighbors=ec.codes_grib_find_nearest(g,lat,lon,npoints=4)
                        minimum=min(q['distance'] for q in neighbors)
                        candidates.append([(dict(q),minimum) for q in neighbors if q['distance']<=minimum+.001])
                    _GEOMETRY[geometry]=candidates
                candidates=_GEOMETRY[geometry]
                indexes=sorted({int(q['index']) for group in candidates for q,_ in group})
                raw_values=dict(zip(indexes,ec.codes_get_double_elements(g,'values',indexes)))
                for (r,c),group in zip(pixels,candidates):
                    if d['quality_mask'][r,c]:
                        matches=[(q,m) for q,m in group if np.isclose(d['precip_mm'][r,c],raw_values[int(q['index'])],atol=1e-4)]
                        assert matches,(path,r,c,'raw precipitation mismatch')
                    else:
                        assert d['precip_mm'][r,c]==0
                        matches=[(q,m) for q,m in group if raw_values[int(q['index'])]<0 or raw_values[int(q['index'])]==ec.codes_get(g,'missingValue')]
                        assert matches,(path,r,c,'invalid mask inconsistent with raw source')
                    nearest,minimum=matches[0];value=raw_values[int(nearest['index'])]
                    checks.append({'file':text,'pixel':[r,c],'target_lat_lon':[float(d['latitude'][r,c]),float(d['longitude'][r,c])],
                        'raw_nearest_lat_lon':[nearest['lat'],nearest['lon']], 'raw_mm':float(value),
                        'processed_mm':float(d['precip_mm'][r,c]),'valid':bool(d['quality_mask'][r,c]),
                        'distance_above_minimum_m':float((nearest['distance']-minimum)*1000)})
            finally:ec.codes_release(g)
        else:
            import xarray as xr
            from pyproj import CRS,Transformer
            info=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            south,west,north,east=info['bbox_south_west_north_east']
            declared_lat=np.linspace(north,south,height)
            declared_lon=np.linspace(west,east,width)
            with xr.open_dataset(raw_path,engine='h5netcdf') as raw:
                assert raw['CMI'].attrs['units']=='K' and int(raw.band_id.item())==13
                tr=Transformer.from_crs('EPSG:4326',CRS.from_cf(raw.goes_imager_projection.attrs),always_xy=True)
                height_m=raw.goes_imager_projection.attrs['perspective_point_height']
                for r,c in pixels:
                    assert d['latitude'][r,c]==np.float32(declared_lat[r])
                    assert d['longitude'][r,c]==np.float32(declared_lon[c])
                    x,y=tr.transform(declared_lon[c],declared_lat[r])
                    col=float(np.interp(x/height_m,raw.x.values,np.arange(raw.sizes['x'])))
                    row=float(np.interp(y/height_m,raw.y.values[::-1],np.arange(raw.sizes['y'])[::-1]))
                    rr,cc=int(np.floor(row)),int(np.floor(col))
                    assert 0<=rr<raw.sizes['y']-1 and 0<=cc<raw.sizes['x']-1
                    block=raw.CMI.isel(y=slice(rr,rr+2),x=slice(cc,cc+2)).values
                    quality=raw.DQF.isel(y=slice(rr,rr+2),x=slice(cc,cc+2)).values
                    good=bool((quality==0).all() and np.isfinite(block).all())
                    assert bool(d['quality_mask'][r,c])==good,(path,r,c,'GOES mask mismatch')
                    if good:
                        wr,wc=row-rr,col-cc
                        expected=block[0,0]*(1-wr)*(1-wc)+block[0,1]*(1-wr)*wc+block[1,0]*wr*(1-wc)+block[1,1]*wr*wc
                        assert np.isclose(d['CMI'][r,c],expected,atol=.002,rtol=0),(path,r,c,'GOES value mismatch')
                    else:
                        expected=0.;assert d['CMI'][r,c]==0
                    checks.append({'file':text,'pixel':[r,c],'raw_bilinear_kelvin':float(expected),
                                   'processed_kelvin':float(d['CMI'][r,c]),'all_four_DQF_zero':bool((quality==0).all()),'valid':good})
    return {'kind':kind,'file':text,'raw_file':raw_text,'raw_sha256':raw_hash,
            'processed_sha256':processed_hash,'verification_version':3 if kind=='goes' else 2,'checks':checks}


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def verify(root=Path('data'),dataset=None,workers=4):
    from concurrent.futures import ProcessPoolExecutor
    from weather550.collect import append_record
    root=Path(root)
    if workers not in range(1,5):raise ValueError('Use 1-4 workers')
    if dataset:
        with np.load(dataset,allow_pickle=False) as d:trace=json.loads(d['source_metadata_json'].item())
        files={'goes':sorted({p for sample in trace for p in sample['goes_files']}),
               'mrms':sorted({p for sample in trace for p in sample['rain_files']})}
    else:files={kind:[str(p) for p in sorted((root/'processed'/kind).glob('*.npz'))] for kind in ['goes','mrms']}
    journal=root/'processed/raw_alignment_progress.jsonl';cached={}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:
                item=json.loads(line);cached[(item['kind'],item['file'])]=item
            except (ValueError,KeyError):continue
    jobs=[];results=[]
    for kind,paths in files.items():
        for text in paths:
            path=Path(text)
            raw=(root/'raw/mrms'/path.with_suffix('.grib2.gz').name if kind=='mrms' else
                 Path(json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['source_nc']))
            raw_hash,processed_hash=digest(raw),digest(path)
            old=cached.get((kind,text))
            if old and old.get('verification_version')==(3 if kind=='goes' else 2) and old['raw_sha256']==raw_hash and old['processed_sha256']==processed_hash:results.append(old)
            else:jobs.append((kind,text,str(raw),raw_hash,processed_hash))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(jobs),64):
            for item in pool.map(_verify_source,jobs[offset:offset+64]):
                results.append(item)
                append_record(journal,item)
            print(f'Raw alignment files checked: {len(results)}/{sum(map(len,files.values()))}',flush=True)
    checks=[c for result in results if result['kind']=='mrms' for c in result['checks']]
    goes=[c for result in results if result['kind']=='goes' for c in result['checks']]
    report={'method':'Three locations per referenced raw source. MRMS: ecCodes four nearest cells, 1m tie tolerance; neighbor indices reused only for identical validated grid geometry and target coordinates. GOES: independent CF projection at declared unrounded grid nodes, exported float32 coordinates checked, four-pixel interpolation and DQF checked; arithmetic tolerance 0.002 K.',
            'source_scope':'dataset-referenced files' if dataset else 'all converted files',
            'dataset_sha256':digest(dataset) if dataset else None,'checks':checks,'count':len(checks),
            'goes_checks':goes,'goes_count':len(goes),
            'sources':[{k:v for k,v in r.items() if k!='checks'} for r in results]}
    Path('docs/RAW_ALIGNMENT_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Passed {len(goes)} GOES and {len(checks)} MRMS raw-point checks across {len(results)} files",flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();verify(dataset=args.dataset,workers=args.workers)
