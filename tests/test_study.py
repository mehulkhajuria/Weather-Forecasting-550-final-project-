from datetime import timedelta
from pathlib import Path
import json
import numpy as np
import pytest
from weather550.collect import study_hours,source_records,_study_source
from weather550.timeutil import iso_utc


def test_full_interval_has_all_issuances_and_boundary_sources():
    hours=study_hours('2026-08-01T00:00Z','2026-11-01T00:00Z')
    assert len(hours)==2208 and iso_utc(hours[-1])=='2026-10-31T23:00:00Z'
    assert iso_utc(hours[0]-timedelta(hours=2))=='2026-07-31T22:00:00Z'
    assert iso_utc(hours[-1]+timedelta(hours=3))=='2026-11-01T02:00:00Z'


@pytest.mark.parametrize('start,end',[('2026-08-01T00:30Z','2026-08-02T00:00Z'),('2026-08-02T00:00Z','2026-08-01T00:00Z')])
def test_invalid_study_bounds(start,end):
    with pytest.raises(ValueError):study_hours(start,end)


def test_journal_resume_deduplicates_attempts_and_ignores_interrupted_append(tmp_path):
    p=tmp_path/'sources.jsonl'
    base={'kind':'mrms','time':'2026-08-01T01:00Z'}
    p.write_text(json.dumps({**base,'ok':False})+'\n'+json.dumps({**base,'ok':True,'file':'actual.npz'})+'\n'+ '{partial')
    records=source_records(p)
    assert len(records)==1 and records[('mrms',base['time'])]['ok']


def test_cached_source_validates_without_network(tmp_path,monkeypatch):
    from weather550.numerical import grid_coordinates
    lat,lon=grid_coordinates([39.2,-75.3,41.2,-73.0],32,32)
    path=tmp_path/'rain.npz'
    np.savez(path,precip_mm=np.zeros((32,32),dtype='float32'),quality_mask=np.ones((32,32),dtype=bool),
             latitude=lat,longitude=lon,valid_end_utc=np.array('2026-08-01T01:00Z'),units=np.array('mm'))
    monkeypatch.setattr('weather550.collect.retry_session',lambda:pytest.fail('Cached source must not issue requests'))
    result=_study_source(('mrms','2026-08-01T01:00Z',str(path)))
    assert result['ok'] and result['cached']



def test_daily_numeric_halo_matches_continuous_history(tmp_path):
    import pandas as pd
    from weather550.numerical import build_numerical
    start=study_hours('2026-07-31T20:00Z','2026-08-02T03:00Z')
    rows=[dict(station='A',timestamp_utc=iso_utc(t-timedelta(minutes=20)),latitude=40.5,longitude=-74.5,
               temperature_c=i,dewpoint_c=5,humidity_pct=70,wind_speed_mps=2,pressure_hpa=1010) for i,t in enumerate(start)]
    csv=tmp_path/'stations.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    paths=[]
    for name,lo,hi in [('full','2026-07-31T22:00Z','2026-08-02T03:00Z'),
                       ('first','2026-07-31T22:00Z','2026-08-02T00:00Z'),
                       ('second','2026-08-01T22:00Z','2026-08-02T03:00Z')]:
        path=tmp_path/(name+'.npz');build_numerical(csv,path,lo,hi);paths.append(path)
    with np.load(paths[0]) as full,np.load(paths[1]) as first,np.load(paths[2]) as second:
        for key in ['X_numerical','X_mask','issuance_times_utc','latest_observation_times_utc']:
            np.testing.assert_array_equal(full[key],np.concatenate([first[key],second[key]]))
        assert len(full['issuance_times_utc'])==27


def delivery_functions():
    import runpy
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_delivery.py'))


def test_episode_gaps_do_not_establish_dry_separation():
    f=delivery_functions()['precipitation_episodes']
    times=['2026-08-01T00:00Z','2026-08-01T03:00Z','2026-08-01T04:00Z','2026-08-01T05:00Z','2026-08-01T06:00Z']
    rain=np.zeros((5,2,1,1),dtype='float32');rain[0]=1;rain[1]=1;rain[4]=1
    labels,gaps=f(times,rain,np.ones_like(rain,dtype=bool),dry_gap_hours=2)
    assert labels[0]==labels[1] and labels[4]!=labels[0]
    assert len(gaps)==1 and labels[3].startswith('dry_')


def test_coverage_reconciles_unique_issuances_and_source_failures(monkeypatch):
    f=delivery_functions()['study_accounting']
    study={'start':'2026-08-01T00:00:00Z','end':'2026-08-01T03:00:00Z'}
    hours=study_hours(study['start'],study['end']);records={}
    for t in hours:
        for h in [0,1,2]:records[('goes',iso_utc(t-timedelta(hours=h)))]={'ok':True}
        for h in [1,3]:records[('mrms',iso_utc(t+timedelta(hours=h)))]={'ok':True}
    key=('goes','2026-08-01T01:00:00Z');records[key]={'kind':'goes','time':key[1],'ok':False,'reason':'http_404'}
    monkeypatch.setattr('weather550.collect.source_records',lambda:records)
    report=f(study,[iso_utc(hours[0])],[])
    assert (report['possible_issuances'],report['attempted_issuances'],report['successful_issuances'],report['rejected_issuances'],report['unattempted_issuances'])==(3,3,1,2,0)
    assert report['rejected_by_reason']=={'goes:http_404':2}
    assert report['missing_intervals'][0]['hours']==2


def test_windows_listing_cache_race_keeps_verified_listing(tmp_path,monkeypatch):
    from weather550.goes import list_public_s3
    class Response:
        content=b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated><Contents><Key>a.nc</Key><LastModified>2026-08-01T00:00:00Z</LastModified></Contents></ListBucketResult>'
        def raise_for_status(self):pass
    class Session:
        def get(self,*a,**kw):return Response()
    def locked_replace(self,target):
        Path(target).write_text(self.read_text());raise PermissionError('Concurrent Windows reader')
    monkeypatch.setattr(Path,'replace',locked_replace)
    assert list_public_s3('test','prefix',session=Session(),cache_dir=tmp_path)==['a.nc']
    assert not list(tmp_path.glob('*.part'))



def test_raw_alignment_reuses_geometry_but_reads_each_new_grib(tmp_path):
    import gzip,runpy
    import eccodes as ec
    from weather550.mrms import decode_mrms_grib
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/verify_real_alignment.py'))
    for number in [2.,7.]:
        g=ec.codes_grib_new_from_samples('regular_ll_sfc_grib2')
        try:
            for key,value in {'Ni':2,'Nj':2,'latitudeOfFirstGridPointInDegrees':41.,
                'longitudeOfFirstGridPointInDegrees':285.,'latitudeOfLastGridPointInDegrees':40.,
                'longitudeOfLastGridPointInDegrees':286.,'iDirectionIncrementInDegrees':1.,
                'jDirectionIncrementInDegrees':1.,'jScansPositively':0,'centre':161,
                'discipline':209,'parameterCategory':6,'parameterNumber':37,
                'dataDate':20260801,'dataTime':100}.items():ec.codes_set(g,key,value)
            ec.codes_set_values(g,[-1.,0.,1.,number])
            raw=tmp_path/f'{number}.gz'
            with gzip.open(raw,'wb') as handle:handle.write(ec.codes_get_message(g))
        finally:ec.codes_release(g)
        out=tmp_path/f'{number}.npz'
        decode_mrms_grib(raw,out,valid_end_utc='2026-08-01T01:00Z',bbox=[40,-75,41,-74],height=2,width=2)
        result=module['_verify_source'](('mrms',str(out),str(raw),'raw_hash','processed_hash'))
        assert len(result['checks'])==3
        assert result['checks'][0]['valid'] is False
        assert result['checks'][-1]['raw_mm']==number
