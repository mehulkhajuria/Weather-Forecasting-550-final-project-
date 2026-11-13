from datetime import datetime,timezone,timedelta
from pathlib import Path
import pandas as pd
import pytest

from weather550.asos import build_params, parse_asos_csv, convert_asos


def fixture(tmp_path):
    p=tmp_path/"a.csv"
    p.write_text("# generated test file\nstation,valid,lat,lon,tmpf,dwpf,relh,sknt,mslp\n"
        "TTN,2026-10-20 00:00,40.3,-74.8,50,41,61,10,1012\n"
        "TTN,2026-10-20 00:00,40.3,-74.8,M,M,M,M,M\n"
        "EWR,2026-10-20 00:00,40.7,-74.2,59,50,72,12,1016\n")
    return p


def test_asos_parse_and_units(tmp_path):
    df=parse_asos_csv(fixture(tmp_path))
    assert len(df)==2  # duplicate METAR merged by highest completeness
    ttn=df.loc[df.station=="TTN"].iloc[0]
    assert ttn.temperature_c==pytest.approx(10.)
    assert ttn.wind_speed_mps==pytest.approx(5.14444)
    assert pd.Timestamp(ttn.timestamp_utc).tzinfo is not None
    assert "solar_wm2" not in df.columns


def test_asos_metadata_and_conversion(tmp_path):
    p=fixture(tmp_path)
    result=convert_asos(p,tmp_path/"stations.csv")
    assert result["records"]==2
    assert result["solar_irradiance_available"] is False
    assert (tmp_path/"stations.summary.json").exists()


def test_asos_reject_bad_coordinates(tmp_path):
    p=tmp_path/"broken.csv"
    p.write_text("station,valid,tmpf\nTTN,2026-10-20 01:00,50\n")
    with pytest.raises(ValueError,match="coordinates"):
        parse_asos_csv(p)


def test_download_date_semantics():
    start=datetime(2026,10,20,tzinfo=timezone.utc)
    end=start+timedelta(days=2)
    params=build_params(["TTN","EWR"],start,end)
    assert ("day2","22") in params
    assert ("tz","Etc/UTC") in params
    assert ("latlon","yes") in params
    assert params.count(("station","TTN"))==1
    with pytest.raises(ValueError):
        build_params(["TTN"],start,end+timedelta(days=10))
