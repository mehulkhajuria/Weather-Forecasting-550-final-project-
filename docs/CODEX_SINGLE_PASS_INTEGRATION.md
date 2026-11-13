# Codex narrow integration instruction (ONLY after local real-data smoke test)

You are joining an existing tested Python v0.3 data-engineering package to a Rutgers CS550 Fourier weather-model project. **Do not regenerate our data pipeline, redesign the models or download large datasets.**

Repository: `mehulkhajuria/Weather-Forecasting-550-final-project-`.

First read: `START_HERE_v0_3.md`, `pyproject.toml`, `src/weather550/{asos,goes,mrms,numerical,dataset,collect,cli}.py`, `tests/`, and teammates' branch/model interface. Do not modify teammate code until inventorying all branches and discussing input shapes.

Implementation task:
1. Add one minimal adapter that loads `data/processed/supervised.npz` into the teammate's model input interface, preserving exact (N,T,C,H,W) tensor axis ordering, timestamps, masks, units, and +1/+3h target semantics.
2. Implement deterministic train/val/test sampling **only with chronological boundaries** (including an embargo for target windows). Fit any normalizer on TRAIN only; use the same split and targets for all four comparisons.
3. Coordinate with the first-model teammate: the numerical stage outputs TWO SPATIAL precipitation maps (+1,+3 hour), and its train overlays for the Fourier model must come from forward-chaining out-of-fold numerical models.
4. Add tests and a tiny integration demo using the *offline synthetic* fixtures. Do NOT claim the real data was validated unless supplied sample logs prove it.
5. Output a one-page record: existing code, files changed, exact input/output shapes, test evidence, research assumptions, and unresolved source-data blockers. Make changes in one branch/PR; do not force-push or overwrite `main`.

Research contract: use the submitted CS550 proposal as authority; adhere to Professor Tang's baseline and leakage advice and TA Shi's demand for named products, region, target and dates. If the teammate's expected interface differs from `supervised.npz`, make an explicit reversible adapter rather than changing scientific semantics.
