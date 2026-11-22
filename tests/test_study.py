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
