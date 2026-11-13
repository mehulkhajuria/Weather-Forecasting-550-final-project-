import numpy as np
import pandas as pd
import pytest
from weather550.demo import make_demo
from weather550.numerical import grid_coordinates,station_map,latest_causal,build_numerical


def test_grid_north_is_top():
    lat,lon=grid_coordinates([39,-75,41,-73],3,4)
    assert lat[0,0]==41 and lat[-1,0]==39
    assert lon[0,0]==-75 and lon[0,-1]==-73


def test_demo_end_to_end(tmp_path):
    d=make_demo(tmp_path)
    assert d["model_shape"]==[46,3,5,32,32]
    data=np.load(tmp_path/"SYNTHETIC_numeric.npz")
    assert data["X_numerical"].shape==(46,3,5,32,32)
    assert data["X_mask"].dtype==np.dtype("bool")
    assert data["X_mask"].any()
    assert len(data["issuance_times_utc"])==46
    assert data["X_numerical"].dtype==np.float32


def test_causal_no_future_values():
    data=pd.DataFrame({"station":["TTN","TTN"],
        "timestamp_utc":pd.to_datetime(["2026-10-20T01:00Z","2026-10-20T03:00Z"],utc=True),
        "temperature_c":[10,99]})
    t=pd.Timestamp("2026-10-20T02:00Z")
    observed=latest_causal(data,t,90)
    assert len(observed)==1
    assert observed.iloc[0].temperature_c==10


def test_validity_mask_not_faked_when_missing():
    lat,lon=grid_coordinates([39,-75,41,-73],3,3)
    x,mask,nearest=station_map(pd.DataFrame(),lat,lon)
    assert not mask.any()
    assert np.all(x==0)
    assert np.all(np.isinf(nearest))


def test_single_station_has_cutoff_radius():
    lat,lon=grid_coordinates([40,-75,42,-73],3,3)
    row={"latitude":40.0,"longitude":-75.0,
         "temperature_c":15,"dewpoint_c":10,"humidity_pct":65,
         "wind_speed_mps":4,"pressure_hpa":1014}
    x,mask,nearest=station_map(pd.DataFrame([row]),lat,lon,radius_km=35)
    assert mask[0,-1,0] # station at south-west corner
    assert not mask[0,0,-1]
    assert x[0,-1,0]==15
