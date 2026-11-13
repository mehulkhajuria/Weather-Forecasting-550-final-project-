"""Run: .venv/Scripts/python scripts/load_dataset.py data/processed/supervised.npz"""
from pathlib import Path
import sys
import numpy as np

path=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/processed/supervised.npz')
with np.load(path,allow_pickle=False) as d:
    X_sat=d['X_goes']             # N,3,1,32,32; brightness temperature in K
    X_num=d['X_numerical']        # N,3,5,32,32; physical units, feature_names order
    Y=d['Y_rain_mm']              # N,2,32,32; hours ending +1h and +3h
    sat_mask=d['X_goes_mask']
    num_mask=d['X_num_mask']
    target_mask=d['Y_rain_mask']
    # Model-side mask-aware imputation is required: zero placeholders are not measurements.
    # Fit any scalers on chronological TRAINING samples only. Do not shuffle before splitting.
    print('Satellite',X_sat.shape,'Numerical',X_num.shape,'Rain',Y.shape)
    print('Features:',d['feature_names'].tolist())
    print('UTC issuances:',d['issuance_times_utc'].tolist())
    print('Valid fractions:',float(sat_mask.mean()),float(num_mask.mean()),float(target_mask.mean()))
    assert X_num.shape[1:] == (3,5,32,32)
