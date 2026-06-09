# enso_nao_teleconnections

To run the scripts, set up the Python environment specified in `scripts/baseclim.yml`

`ENSO_NAO_figures_slidingwindow.ipynb` takes ENSO indices and computes/makes the following summary plots:
- Monthly Standard Deviation across all experiments (similar to Moon et al. 2025)
- Power Spectra (similar to Stuecker et al. 2025)
- Sliding standard deviation and lag-1 autocorrelation
- Timeseries plot comparing experiments.

Use `lag_regression_enso_share.py` to perform lag regression given a variable and ENSO Indices. Also includes visualization of the computed regression coefficients.



