import importlib.util
from pathlib import Path
import json
import pytest

spec=importlib.util.spec_from_file_location('delivery',Path(__file__).resolve().parents[1]/'scripts/build_delivery.py')
delivery=importlib.util.module_from_spec(spec);spec.loader.exec_module(delivery)


def test_period_config_accepts_variable_lengths(tmp_path):
    path=tmp_path/'periods.json'
    periods=[{'event_id':'a','kind':'rainy','start':'2026-08-03T08:00Z','end':'2026-08-03T12:00Z'},
             {'event_id':'b','kind':'mixed','start':'2026-08-10T12:00Z','end':'2026-08-10T15:00Z'}]
    path.write_text(json.dumps({'periods':periods}))
    assert len(delivery.configured_periods(path)['periods'])==2


@pytest.mark.parametrize('fault',['overlap','duplicate','fractional','same_storm'])
def test_period_config_rejects_ambiguous_events(tmp_path,fault):
    a={'event_id':'a','kind':'rainy','start':'2026-08-03T08:00Z','end':'2026-08-03T12:00Z'}
    b={**a,'event_id':'b','start':'2026-08-06T08:00Z','end':'2026-08-06T12:00Z'}
    if fault=='overlap': b.update(start=a['start'],end=a['end'])
    elif fault=='duplicate': b['event_id']='a'
    elif fault=='fractional': b['start']='2026-08-06T08:30Z'
    else: b.update(start='2026-08-04T08:00Z',end='2026-08-04T12:00Z')
    path=tmp_path/'periods.json';path.write_text(json.dumps({'periods':[a,b]}))
    with pytest.raises(ValueError):delivery.configured_periods(path)


def test_delivery_sorts_samples_and_preserves_event_ids(tmp_path,monkeypatch):
    import numpy as np
    import pandas as pd
    from datetime import datetime,timedelta,timezone
    from weather550.numerical import grid_coordinates
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(delivery,'ROOT',tmp_path)
    out=tmp_path/'data/processed';(out/'goes').mkdir(parents=True);(out/'mrms').mkdir()
    (tmp_path/'docs').mkdir()
    base=datetime(2026,10,20,tzinfo=timezone.utc)
    rows=[];lat,lon=grid_coordinates([39.2,-75.3,41.2,-73.0],32,32)
    for day in [0,1]:
        for hour in range(10):
            tick=base+timedelta(days=day,hours=hour)
            rows.append(dict(station='A',timestamp_utc=(tick-timedelta(minutes=20)).isoformat(),
                latitude=40.5,longitude=-74.5,temperature_c=10,dewpoint_c=5,
                humidity_pct=70,wind_speed_mps=2,pressure_hpa=1010))
            np.savez(out/'goes'/f'{day}_{hour}.npz',CMI=np.full((32,32),280,dtype='float32'),
                quality_mask=np.ones((32,32),dtype=bool),latitude=lat,longitude=lon,
                units=np.array('K'),scan_time_utc=np.array((tick-timedelta(minutes=5)).isoformat()),
                available_time_utc=np.array((tick-timedelta(minutes=2)).isoformat()))
            np.savez(out/'mrms'/f'{day}_{hour}.npz',precip_mm=np.full((32,32),.1 if day==0 and hour==5 else 0,dtype='float32'),
                quality_mask=np.ones((32,32),dtype=bool),latitude=lat,longitude=lon,
                units=np.array('mm'),valid_end_utc=np.array(tick.isoformat()))
    pd.DataFrame(rows).to_csv(out/'stations.csv',index=False)
    config=tmp_path/'periods.json'
    config.write_text(json.dumps({'event_definition':'Two synthetic test periods.', 'periods':[
        {'event_id':'later','kind':'dry','start':'2026-10-21T04:00Z','end':'2026-10-21T07:00Z','station_file':str(out/'stations.csv')},
        {'event_id':'earlier','kind':'mixed','start':'2026-10-20T04:00Z','end':'2026-10-20T06:00Z','station_file':str(out/'stations.csv')}]}))
    delivery.main(['--periods',str(config)])
    with np.load(out/'supervised.npz',allow_pickle=False) as d:
        assert d['X_numerical'].shape==(5,3,5,32,32)
        assert d['event_ids'].tolist()==['earlier','earlier','later','later','later']
        assert d['issuance_times_utc'].tolist()==sorted(d['issuance_times_utc'].tolist())
    report=json.loads((tmp_path/'docs/REAL_DATA_VERIFICATION.json').read_text())
    assert report['samples']==5 and report['verified_dry_periods']==1
    assert report['verified_mixed_periods']==1 and report['verified_rainy_events']==0
