import importlib.util
from pathlib import Path
import json
import pytest

spec=importlib.util.spec_from_file_location('delivery',Path(__file__).resolve().parents[1]/'scripts/build_delivery.py')
delivery=importlib.util.module_from_spec(spec);spec.loader.exec_module(delivery)


def test_period_config_accepts_variable_lengths(tmp_path):
    path=tmp_path/'periods.json'
    periods=[{'event_id':'a','kind':'rainy','start':'2026-08-03T08:00Z','end':'2026-08-03T12:00Z'},
             {'event_id':'b','kind':'dry','start':'2026-08-10T12:00Z','end':'2026-08-10T15:00Z'}]
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
