"""Offline demo: explicit synthetic fixture, NEVER research observations."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math

from .asos import convert_asos
from .numerical import build_numerical


def make_demo(root: Path):
    root.mkdir(parents=True,exist_ok=True)
    raw=root/"SYNTHETIC_asos.csv"
    rows=["# SYNTHETIC TEST DATA - NOT OBSERVATIONS; DO NOT USE FOR RESEARCH OR TRAINING",
          "station,valid,lat,lon,tmpf,dwpf,relh,sknt,mslp"]
    start=datetime(2026,10,20,tzinfo=timezone.utc)
    for h in range(48):
        t=start+timedelta(hours=h)
        for station,lat,lon,bias in [("DEMO_A",40.30,-74.60,0),("DEMO_B",40.75,-74.20,2)]:
            temp=56+5*math.sin(h/6)+bias
            dew=temp-6
            rows.append(f'{station},{t:%Y-%m-%d %H:%M},{lat},{lon},'
                        f'{temp:.1f},{dew:.1f},78,{5+(h%4)},1014')
    raw.write_text("\n".join(rows)+"\n",encoding="utf-8")
    processed=root/"SYNTHETIC_stations_utc.csv"
    convert_asos(raw,processed)
    result=build_numerical(processed,root/"SYNTHETIC_numeric.npz",
            start="2026-10-20T00:00:00Z",end="2026-10-22T00:00:00Z")
    return {"notice":"ALL DATA IN THIS DIRECTORY ARE SYNTHETIC DEMO FIXTURES",
            "raw":str(raw),"processed":str(processed),"model_input":str(root/"SYNTHETIC_numeric.npz"),
            "model_shape":result["shape_X_numerical"]}
