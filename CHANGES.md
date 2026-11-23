# Changes

## 1.0.0 (unreleased)

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
