# Changes

## 1.0.0 (unreleased)

New features
- `BME_tools.read_xvg` reads GROMACS `.xvg` files; `Reweight.load_xvg` loads a single
  observable (e.g. the radius of gyration from `gmx gyrate`) with a target value and error,
  optionally as an upper/lower bound and on a time window.
- `notebook/example_06.ipynb`: reweighting towards a target radius of gyration from
  `gmx gyrate`, alone and together with chemical shifts (synthetic data in `data/rg_example.xvg`,
  `data/cs_example_exp.dat`, `data/cs_example_calc.dat.zip`).

Combining datasets and long trajectories
- The `weight` argument of `load`/`load_array` acted in the opposite direction (weight=10
  made a dataset 10x *less* important). A data point with weight k now counts as k data
  points (sigma -> sigma/sqrt(k)); weight=1 (the default) gives the same results as before.
  It is also used by the BER/CHI2 methods.
- `load_xvg`: `stride` and `use_samples`, to match data computed on a subsampled
  trajectory; `predict_xvg` gives Rg statistics after a fit. Datasets with different numbers
  of frames give an error that says so (it wrongly mentioned the initial weights w0).
- Memory and speed for large data (e.g. 1.25e6 frames x 250 chemical shifts, 2.3 GB):
  - Lagrange multipliers were `np.longdouble`, so every product with the data matrix
    bypassed BLAS (about 50x slower) and temporaries were 16 bytes per element.
  - Products are computed as matrix-vector products instead of broadcast temporaries,
    standardization does not copy the data, calculated files are read into a preallocated
    array, and datasets are stored as separate blocks instead of being concatenated.
  - Peak memory goes from >4x to about 1.2x the size of the data; on the example above the
    fit takes 40 s.
- L-BFGS-B tolerances tightened (`ftol=1e-12`, `gtol=1e-9`): with the defaults the
  minimization could stop early on ill-conditioned problems (SAXS) and results depended on
  round-off (numpy/scipy version). Converged results agree with the reference values in
  `test/test_01.py`; the `predict` output test compares numbers with a tolerance.

Packaging
- `pyproject.toml`: BME can be installed with `pip install .` (modules `BME` and
  `BME_tools`, so `import BME` works unchanged). Optional extras `[test]` and `[notebooks]`.
- `environment.yml` for conda; GitHub Actions workflow (`.github/workflows/tests.yml`)
  testing Python 3.9–3.13, NumPy 1.x and 2.x, Linux and macOS.
- `.travis.yml` updated: the old one installed the `sklearn` placeholder package,
  which now refuses to install, and targeted Python 3.4–3.8.
- `matplotlib` is imported only when `theta_scan`/`iterative_theta_scan` are used.

Bug fixes
- NumPy 2 compatibility: `np.NaN` (removed in NumPy 2) replaced by `np.nan`; slope and
  offset of `fit="scale"`/`"scale+offset"` are now converted to floats before formatting
  (crashed with NumPy 2, deprecation warning with NumPy >= 1.25).
- `Reweight.load()`/`load_array()` called after `fit()` mixed already standardized data
  with new raw data, silently giving wrong weights. Standardization is now applied to a
  copy and recomputed when data are added; `get_experiment()`/`get_calculated()` always
  return the original data.
- `fit(method="CHI2_L2")` crashed (objective returned no gradient although `jac=True`;
  gradient was missing the 1/sigma factor) and returned nothing; `"CHI2_L1"` was
  unreachable (listed as `"CHI1_L1"`) and minimized the L2 objective. Both now work and
  return `(chi2_before, chi2_after, phi)` like `"BME"` and `"BER"`.
- `fit()` with an unknown method raised a `TypeError` from a broken error message.
- `standardize(normalize="minmax")` and any unknown option raised `UnboundLocalError`;
  `normalize="bme"` printed "WHAT???" and returned scale factors equal to 1.
  `normalize="none"` is now supported.
- `check_data` counted as violations only data whose deviation was larger than 1 in raw
  units and positive; it now counts deviations larger than one sigma in both directions
  (affects the log file only).
- Final BME weights are computed with `logsumexp` (avoids precision loss in subnormal
  numbers); results are unchanged to within 1e-13.
- Initial weights `w0` whose length differs from the number of samples (e.g. with
  `use_samples`) now give a clear error instead of a numpy broadcasting error.
- `predict()` before `fit()` gives a clear error.
- `set_lambdas()`, `get_ibme_weights()`, `get_ibme_stats()` raise exceptions instead of
  calling `sys.exit()`, which killed Jupyter kernels.
- `progress()` divided by zero with `nfold=1`; `theta_scan` uses `os.makedirs` instead of
  `os.system("mkdir -p")`; NaN results in cross-validation no longer break the plot.
- iBME stops with a clear error if a BME minimization fails (it crashed on the next iteration).
- Log files of the temporary `Reweight` objects created by iBME and cross-validation are
  closed; `Reweight` has a `close()` method and can be used as a context manager.
- Input validation errors are raised as `ValueError` instead of `assert` (asserts are
  removed when Python runs with `-O`).
- Invalid escape sequences (`"\s+"`) that give `SyntaxWarning` on Python >= 3.12.

Tests
- `test/test_01.py`: the comparison helper used `abs(max(diff))`, so any negative deviation
  passed; it now uses `max(abs(diff))` (all reference values are still reproduced).
  The `filecmp` comparison of `predict` outputs was not asserted.
- New unit tests: `test/test_tools.py` (parsing, scaling, statistics, standardization) and
  `test/test_reweight.py` (analytical Gaussian solution, bounds, priors, warm start,
  alternative methods, iBME, cross-validation, input validation).
- Tests run in temporary folders and work from any directory; nose2 replaced by pytest.
