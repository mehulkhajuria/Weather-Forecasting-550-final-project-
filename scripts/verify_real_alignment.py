"""Independent ecCodes geolocation check against every downloaded target map."""
from pathlib import Path
import gzip
import json
import numpy as np
import eccodes as ec


def verify(root=Path('data')):
    checks=[]
    for path in sorted((root/'processed/mrms').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            raw=root/'raw/mrms'/path.with_suffix('.grib2.gz').name
            with gzip.open(raw,'rb') as f: g=ec.codes_new_from_message(f.read())
            try:
                for r,c in [(0,0),(15,15),(31,31)]:
                    lat,lon=float(d['latitude'][r,c]),float(d['longitude'][r,c])
                    neighbors=ec.codes_grib_find_nearest(g,lat,lon,npoints=4)
                    minimum=min(q['distance'] for q in neighbors)
                    # Bbox endpoints fall exactly between native pixel centers.
                    # float32 exported coords can break a tie by <1m. Accept only
                    # raw neighbors within 1m of the true nearest, never arbitrary cells.
                    candidates=[q for q in neighbors if q['distance'] <= minimum + .001]
                    matching=[q for q in candidates if np.isclose(d['precip_mm'][r,c],q['value'],atol=1e-4)]
                    nearest=matching[0] if matching else min(neighbors,key=lambda q:q['distance'])
                    expected=nearest['value']
                    if d['quality_mask'][r,c]:
                        assert matching, (path,r,c,neighbors)
                    else:
                        assert expected < 0 or expected == ec.codes_get(g,'missingValue')
                    checks.append({'file':str(path),'pixel':[r,c],'target_lat_lon':[lat,lon],
                        'raw_nearest_lat_lon':[nearest['lat'],nearest['lon']],
                        'raw_mm':float(expected),'processed_mm':float(d['precip_mm'][r,c]),
                        'distance_above_minimum_m':float((nearest['distance']-minimum)*1000)})
            finally: ec.codes_release(g)
    import xarray as xr
    from pyproj import CRS, Transformer
    goes_checks=[]
    for path in sorted((root/'processed/goes').glob('*.npz')):
        info=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        with xr.open_dataset(info['source_nc'],engine='h5netcdf') as raw, np.load(path,allow_pickle=False) as d:
            assert raw['CMI'].attrs['units']=='K' and int(raw.band_id.item())==13
            projection=CRS.from_cf(raw.goes_imager_projection.attrs)
            transformer=Transformer.from_crs('EPSG:4326',projection,always_xy=True)
            height=raw.goes_imager_projection.attrs['perspective_point_height']
            for r,c in [(0,0),(15,15),(31,31)]:
                x,y=transformer.transform(float(d['longitude'][r,c]),float(d['latitude'][r,c]))
                col=float(np.interp(x/height,raw.x.values,np.arange(raw.sizes['x'])))
                row=float(np.interp(y/height,raw.y.values[::-1],np.arange(raw.sizes['y'])[::-1]))
                rr,cc=int(np.floor(row)),int(np.floor(col))
                assert 0<=rr<raw.sizes['y']-1 and 0<=cc<raw.sizes['x']-1
                block=raw.CMI.isel(y=slice(rr,rr+2),x=slice(cc,cc+2)).values
                quality=raw.DQF.isel(y=slice(rr,rr+2),x=slice(cc,cc+2)).values
                wr,wc=row-rr,col-cc
                expected=block[0,0]*(1-wr)*(1-wc)+block[0,1]*(1-wr)*wc+block[1,0]*wr*(1-wc)+block[1,1]*wr*wc
                assert (quality==0).all() and d['quality_mask'][r,c]
                assert np.isclose(d['CMI'][r,c],expected,atol=.002,rtol=0), (path,r,c,expected)
                goes_checks.append({'file':str(path),'pixel':[r,c],'raw_bilinear_kelvin':float(expected),
                                    'processed_kelvin':float(d['CMI'][r,c]),'all_four_DQF_zero':True})
    result={'goes_checks':goes_checks,'goes_count':len(goes_checks),'method':'ecCodes four nearest raw GRIB cells; value must match a nearest cell within 1m tie tolerance; independent of axis reconstruction',
            'checks':checks,'count':len(checks)}
    Path('docs/RAW_ALIGNMENT_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Raw GRIB geographic/value checks passed: {len(checks)}')
    return result

if __name__=='__main__': verify()
