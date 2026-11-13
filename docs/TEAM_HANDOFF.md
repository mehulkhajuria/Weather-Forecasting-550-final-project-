# Team: final data interface and responsibilities

Your data engineer delivers `data/processed/supervised.npz` plus its `.json` source provenance summary. The data engineer owns the acquisition/masking/reprojection and outputs, not the model training. Data and results are not committed to Git; send only a small consented sample separately to teammates.

Class pipeline: GOES-19 infrared C13 + historical IEM ASOS + IEM MRMS Pass2 QPE rainfall totals, New Jersey / surroundings, candidate study period Aug 1–Oct 31 2026. Solar irradiation is unavailable in the baseline feed; do not add fictional solar measurements.

Numpy input: `X_goes` `[N,3,1,H,W]`, `X_numerical` `[N,3,5,H,W]`, numerical/satellite masks same shape. Output: `Y_rain_mm` `[N,2,H,W]`, with target 0 for the ONE-hour amount ending issue+1h; target 1 for one-hour amount ending issue+3h (NOT a 3-hour total). Data are unnormalized; the training team must fit standardization on the training split only.

Numerical model teammate: train model with permitted numerical input and 2 map targets; produce *out-of-fold chronological* first-stage overlay predictions for training the Fourier stage. No target-included overfit predictions.

Satellite/fusion teammate: use identical forecast issue times / targets / splits for persistence, GOES-only, direct concat, and numerical overlay; report metrics with precip-event / rainy cases separately.

Please confirm desired spatial resolution (32x32 or 64x64), history steps and grid bbox before the data engineer scales downloads. For the first case use 32x32.
