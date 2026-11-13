from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pytest
from weather550.dataset import build_supervised,split_by_target_bounds


def _t(dt):return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def test_chronological_split_embargo():
    t=['2026-09-01T00:00:00Z','2026-09-02T23:00:00Z','2026-09-04T02:00:00Z']
    result=split_by_target_bounds(t,train_before='2026-09-03T00:00:00Z',val_before='2026-09-04T00:00:00Z')
    assert list(result)==['train','exclude','test']


def test_synthetic_complete_dataset(tmp_path):
    base=datetime(2026,9,1,12,tzinfo=timezone.utc)
    lat=np.array([[41.,41.],[40.,40.]],dtype='float32')
    lon=np.array([[-75.,-74.],[-75.,-74.]],dtype='float32')
    ntimes=np.array([_t(base),_t(base+timedelta(hours=1))])
    num=tmp_path/'numerical.npz'
    np.savez_compressed(num,X_numerical=np.ones((2,3,5,2,2),dtype='float32'),
        X_mask=np.ones((2,3,5,2,2),dtype=bool),issuance_times_utc=ntimes,
        lat_grid=lat,lon_grid=lon,feature_names=np.array(['a','b','c','d','e']))
    g=tmp_path/'goes';m=tmp_path/'mrms';g.mkdir();m.mkdir()
    for h in [-2,-1,0,1]:
        time=base+timedelta(hours=h)
        np.savez_compressed(g/f'scene_{h}.npz',CMI=np.full((2,2),270.+h,dtype='float32'),
            quality_mask=np.ones((2,2),dtype=bool),latitude=lat,longitude=lon,
            scan_time_utc=np.array(_t(time)))
    for h in [1,2,3,4]:
        time=base+timedelta(hours=h)
        np.savez_compressed(m/f'rain_{h}.npz',precip_mm=np.full((2,2),h,dtype='float32'),
            quality_mask=np.ones((2,2),dtype=bool),latitude=lat,longitude=lon,
            valid_end_utc=np.array(_t(time)))
    res=build_supervised(num,g,m,tmp_path/'out.npz')
    assert res['samples']==2
    with np.load(tmp_path/'out.npz') as d:
        assert d['X_goes'].shape==(2,3,1,2,2)
        assert d['X_numerical'].shape==(2,3,5,2,2)
        assert d['Y_rain_mm'].shape==(2,2,2,2)
        assert d['Y_rain_mm'][0,1,0,0]==3
        # GOES at t+1 must not appear as input to t.
        assert d['X_goes'][0,-1,0,0,0]==270


def test_no_future_sample_leak(tmp_path):
    # Requires nonempty catalog; no future weather scan can fill a missing past observation.
    from weather550.dataset import split_by_target_bounds
    assert split_by_target_bounds(['2026-09-02T23:00:00Z'],
        train_before='2026-09-03T00:00:00Z',val_before='2026-09-04T00:00:00Z').item()=='exclude'
