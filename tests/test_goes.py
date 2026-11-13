from datetime import datetime, timezone
import numpy as np
import pytest
from weather550.goes import scan_start_from_key,select_goes_c13,_fractional_index,_bilinear,list_public_s3


def test_goes_filename_time():
    k="ABI-L2-CMIPC/2026/294/18/OR_ABI-L2-CMIPC-M6C13_G19_s20262941800123_e20262941802456_c20262941803000.nc"
    dt=scan_start_from_key(k)
    assert dt.year==2026 and dt.hour==18 and dt.minute==0
    assert dt.second==12
    assert dt.microsecond==300000


def test_fractional_index_descending():
    coords=np.array([3.,2.,1.,0.])
    v=_fractional_index(coords,np.array([2.5,1.5,-10.]))
    assert v[0]==pytest.approx(.5)
    assert v[1]==pytest.approx(1.5)
    assert np.isnan(v[2])


def test_bilinear_values():
    a=np.array([[0,10],[20,30]],dtype=float)
    v,valid=_bilinear(a,np.array([[.5,3.]]),np.array([[.5,.5]]))
    assert v[0,0]==pytest.approx(15)
    assert bool(valid[0,0])
    assert not valid[0,1]


def test_s3_xml_listing():
    xml=b'''<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated><Contents><Key>a/b.nc</Key></Contents></ListBucketResult>'''
    class FakeResp:
        content=xml
        def raise_for_status(self): pass
    class Session:
        def get(self,*args,**kwargs): return FakeResp()
    assert list_public_s3("example","prefix",session=Session())==["a/b.nc"]


def test_reproject_synthetic_netcdf(tmp_path):
    xr = pytest.importorskip("xarray")
    pyproj = pytest.importorskip("pyproj")
    from weather550.goes import reproject_goes_netcdf
    from pyproj import CRS, Transformer
    h=35786023.0
    attrs={"perspective_point_height":h,"longitude_of_projection_origin":-75.0,
           "semi_major_axis":6378137.0,"semi_minor_axis":6356752.31414,
           "sweep_angle_axis":"x"}
    crs=CRS.from_proj4(f'+proj=geos +h={h} +lon_0=-75 +a=6378137 '
                         '+b=6356752.31414 +sweep=x +units=m +no_defs')
    tr=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    xlo,ylo=tr.transform(-75.3,39.2)
    xhi,yhi=tr.transform(-73.0,41.2)
    x=np.linspace(min(xlo,xhi)/h-.005,max(xlo,xhi)/h+.005,20)
    y=np.linspace(max(ylo,yhi)/h+.005,min(ylo,yhi)/h-.005,20)
    ds=xr.Dataset(
        {"CMI":(("y","x"),np.full((20,20),280.,dtype=np.float32),{"units":"K"}),
         "DQF":(("y","x"),np.zeros((20,20),dtype=np.int16)),
         "goes_imager_projection":((),0,attrs),
         "band_id":((),13)},
        coords={"x":x,"y":y})
    raw=tmp_path/'synthetic_GOES_C13.nc'
    ds.to_netcdf(raw,engine="scipy")
    out=tmp_path/'crop.npz'
    info=reproject_goes_netcdf(raw,out,bbox=[39.2,-75.3,41.2,-73.0],height=8,width=8)
    crop=np.load(out)
    assert crop['CMI'].shape==(8,8)
    assert crop['quality_mask'].all()
    assert crop['CMI'].mean()==pytest.approx(280,abs=.01)
    assert info['valid_fraction']==1.0
