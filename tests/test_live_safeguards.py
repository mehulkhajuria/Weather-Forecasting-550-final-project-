"""Offline regressions for failures found while integrating the real sources."""
from datetime import datetime, timedelta, timezone
import gzip
import numpy as np
import pandas as pd
import pytest
from weather550.goes import select_goes_c13, _bilinear
from weather550.mrms import decode_mrms_grib
from weather550.numerical import latest_causal
from weather550.dataset import build_supervised


def test_goes_requires_publication_before_cutoff(monkeypatch):
    cutoff = datetime(2026,10,20,18,tzinfo=timezone.utc)
    keys = [f'OR_ABI-L2-CMIPC-M6C13_G19_s2026293{hhmm}000_e0_c0.nc'
            for hhmm in ['1751','1756','1801']]
    published = dict(zip(keys,[cutoff-timedelta(minutes=5),cutoff+timedelta(seconds=1),cutoff+timedelta(minutes=4)]))
    monkeypatch.setattr('weather550.goes.list_public_s3',lambda *a,**kw:published)
    key, scan, available = select_goes_c13('test',cutoff,return_availability=True)
    assert key == keys[0] and scan < available <= cutoff


def test_bilinear_bad_quality_contributor_invalidates_pixel():
    v, m = _bilinear(np.array([[280.,np.nan],[280.,280.]]),np.array([.2]),np.array([.2]))
    assert not m.item()


def test_station_publication_lag_and_staleness():
    t = pd.Timestamp('2026-10-20T18:00Z')
    df = pd.DataFrame({'station':['A','B','C'], 'timestamp_utc':pd.to_datetime(
        ['2026-10-20T17:40Z','2026-10-20T17:55Z','2026-10-20T16:00Z'],utc=True)})
    assert latest_causal(df,t,90,10).station.tolist() == ['A']


@pytest.fixture
def grib(tmp_path):
    import eccodes as ec
    def make(parameter=37):
        g=ec.codes_grib_new_from_samples('regular_ll_sfc_grib2')
        try:
            for k,v in {'Ni':2,'Nj':2,'latitudeOfFirstGridPointInDegrees':41.,
                'longitudeOfFirstGridPointInDegrees':285.,'latitudeOfLastGridPointInDegrees':40.,
                'longitudeOfLastGridPointInDegrees':286.,'iDirectionIncrementInDegrees':1.,
                'jDirectionIncrementInDegrees':1.,'jScansPositively':0,'centre':161,
                'discipline':209,'parameterCategory':6,'parameterNumber':parameter,
                'dataDate':20261020,'dataTime':1900}.items(): ec.codes_set(g,k,v)
            ec.codes_set_values(g,np.array([0.,1.,-1.,2.]))
            p=tmp_path/f'fixture_{parameter}.grib2.gz'
            with gzip.open(p,'wb') as f: f.write(ec.codes_get_message(g))
            return p
        finally: ec.codes_release(g)
    return make


def test_real_grib_codec_units_geometry_and_missing(grib,tmp_path):
    out=tmp_path/'decoded.npz'
    result=decode_mrms_grib(grib(),out,valid_end_utc='2026-10-20T19:00Z',
                            bbox=[40,-75,41,-74],height=2,width=2)
    with np.load(out) as d:
        np.testing.assert_array_equal(d['precip_mm'],[[0,1],[0,2]])
        np.testing.assert_array_equal(d['quality_mask'],[[True,True],[False,True]])
        assert d['latitude'][0,0] == 41 and d['longitude'][0,0] == -75
        assert d['units'].item() == 'mm'
    assert result['grib_parameter_identity'] == [209,6,37]


def test_grib_rejects_wrong_valid_time(grib,tmp_path):
    with pytest.raises(ValueError,match='valid time'):
        decode_mrms_grib(grib(),tmp_path/'bad.npz',valid_end_utc='2026-10-20T21:00Z')


def test_grib_rejects_three_hour_product(grib,tmp_path):
    with pytest.raises(ValueError,match='209/6/37'):
        decode_mrms_grib(grib(38),tmp_path/'bad.npz',valid_end_utc='2026-10-20T19:00Z')


@pytest.fixture
def joined(tmp_path):
    base=datetime(2026,10,20,18,tzinfo=timezone.utc)
    lat=np.array([[41,41],[40,40]],dtype='float32');lon=np.array([[-75,-74],[-75,-74]],dtype='float32')
    numeric=dict(X_numerical=np.ones((2,3,5,2,2),dtype='float32'),X_mask=np.ones((2,3,5,2,2),dtype=bool),
        issuance_times_utc=np.array([base.isoformat(),(base+timedelta(hours=1)).isoformat()]),
        feature_names=np.array(['a','b','c','d','e']),lat_grid=lat,lon_grid=lon,
        history_offsets_minutes=np.array([-120,-60,0]))
    g=tmp_path/'goes';m=tmp_path/'mrms';g.mkdir();m.mkdir()
    for h in [-2,-1,0,1]:
        t=base+timedelta(hours=h)
        np.savez(g/f'{h}.npz',CMI=np.full((2,2),270+h),quality_mask=np.ones((2,2),dtype=bool),
            latitude=lat,longitude=lon,scan_time_utc=np.array(t.isoformat()),units=np.array('K'),
            available_time_utc=np.array(t.isoformat()))
    for h in [1,2,3,4]:
        t=base+timedelta(hours=h)
        np.savez(m/f'{h}.npz',precip_mm=np.full((2,2),h),quality_mask=np.ones((2,2),dtype=bool),
            latitude=lat,longitude=lon,valid_end_utc=np.array(t.isoformat()),units=np.array('mm'))
    def run():
        n=tmp_path/'numeric.npz';np.savez(n,**numeric)
        return build_supervised(n,g,m,tmp_path/'supervised.npz')
    return numeric,g,m,run,tmp_path


def test_join_rejects_each_all_missing_numerical_sample(joined):
    n,g,m,run,p=joined;n['X_mask'][0]=False
    result=run()
    assert result['samples']==1 and result['skipped']['numerical']==1


@pytest.mark.parametrize('fault',['coordinates','units','nonfinite','cadence'])
def test_join_rejects_corrupt_sources(joined,fault):
    n,g,m,run,p=joined
    if fault == 'cadence': n['history_offsets_minutes']=np.array([-60,-30,0])
    elif fault == 'nonfinite': n['X_numerical'][0,0,0,0,0]=np.nan
    else:
        path=m/'1.npz'
        with np.load(path) as d: fields={k:d[k] for k in d.files}
        if fault == 'units': fields['units']=np.array('mm/h')
        else: fields['longitude']=fields['longitude']+.0005
        np.savez(path,**fields)
    with pytest.raises(ValueError):run()


def test_join_excludes_scan_published_after_input_cutoff(joined):
    n,g,m,run,p=joined
    path=g/'0.npz'
    with np.load(path) as d: fields={k:d[k] for k in d.files}
    fields['available_time_utc']=np.array('2026-10-20T18:01Z');np.savez(path,**fields)
    with pytest.raises(ValueError,match="'goes': 2"):
        run()


def test_join_masks_and_target_placement(joined):
    n,g,m,run,p=joined;n['X_mask'][0,0,0,0,0]=False;n['X_numerical'][0,0,0,0,0]=np.nan
    run()
    with np.load(p/'supervised.npz',allow_pickle=False) as d:
        assert d['X_goes'].shape == (2,3,1,2,2)
        assert d['X_numerical'][0,0,0,0,0] == 0
        np.testing.assert_array_equal(d['Y_rain_mm'][0,:,0,0],[1,3])
        assert d['rain_valid_end_times_utc'][0,1].startswith('2026-10-20T21:00')


def test_asos_cache_reuses_only_matching_historical_request(tmp_path):
    import json
    from weather550.asos import download_asos, ASOS_ENDPOINT
    p=tmp_path/'asos.csv'
    p.write_text('station,valid,lat,lon,tmpf\nTTN,2026-10-20 12:53,40.3,-74.8,50\n')
    start=datetime(2026,10,20,tzinfo=timezone.utc);end=start+timedelta(days=1)
    p.with_suffix('.request.json').write_text(json.dumps({'source':ASOS_ENDPOINT,'stations':['TTN'],
        'start_utc':'2026-10-20T00:00:00Z','end_exclusive_utc':'2026-10-21T00:00:00Z'}))
    class NoNetwork:
        def get(self,*a,**kw): raise AssertionError('Unnecessary repeated historical request')
    assert download_asos(['TTN'],start,end,p,session=NoNetwork())==p


def test_split_embargo_includes_stale_station_lookback(joined):
    n,g,m,run,p=joined
    from weather550.dataset import split_by_target_bounds
    # t-2h lies after the test cutoff, but a station at t-3.5h can predate it.
    labels=split_by_target_bounds(['2026-10-20T18:00Z'],train_before='2026-10-19T00:00Z',
        val_before='2026-10-20T15:00Z',history_hours=3.5)
    assert labels.item()=='exclude'
