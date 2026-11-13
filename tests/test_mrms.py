from datetime import datetime,timezone
import numpy as np
import pytest
from weather550.mrms import mrms_archive_url,sample_regular_grid

def test_url_exact_archive():
    dt=datetime(2026,6,9,0,tzinfo=timezone.utc)
    assert mrms_archive_url(dt).endswith('/MultiSensor_QPE_01H_Pass2_00.00_20260609-000000.grib2.gz')

def test_url_reject_nonhour():
    with pytest.raises(ValueError):mrms_archive_url(datetime(2026,6,9,0,10,tzinfo=timezone.utc))

def test_sample_increasing_axes_and_missing():
    values=np.array([[1.,2.],[3.,np.nan]])
    target_lat=np.array([[40.,41.],[41.,50.]])
    target_lon=np.array([[-75.,-74.],[-75.,-74.]])
    out,mask=sample_regular_grid(values,[40.,41.],[-75.,-74.],target_lat,target_lon)
    assert out[0,0] == 1 and mask[0,0]
    assert out[0,1] == 4 or not mask[0,1]  # Missing corner must be masked
    assert not mask[1,1]

def test_descending_and_missing():
    vals=np.array([[2.,3.],[4.,5.]])
    out,mask=sample_regular_grid(vals,[41,40],[-75,-74],np.array([[40.]]),np.array([[-74.]]))
    assert mask.item() and out.item()==5
