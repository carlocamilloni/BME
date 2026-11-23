"""Unit tests for BME_tools (parsing, scaling, statistics, standardization)."""
import numpy as np
import pytest

import BME_tools as bt


# ---------------------------------------------------------------- srel
def test_srel_zero_for_identical_weights():
    w = np.array([0.2, 0.3, 0.5])
    assert bt.srel(w, w) == pytest.approx(0.0)


def test_srel_matches_kl_divergence_and_ignores_zero_weights():
    w0 = np.array([0.25, 0.25, 0.25, 0.25])
    w1 = np.array([0.5, 0.5, 0.0, 0.0])
    # KL(w1||w0) = sum w1 log(w1/w0) = log 2
    assert bt.srel(w0, w1) == pytest.approx(np.log(2.0))
    assert bt.srel(w0, w1) >= 0


# ---------------------------------------------------------------- parse
def test_parse_linear_data(write_dataset):
    calc = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    e, c = write_dataset("# DATA=JCOUPLINGS", [("a", 3.0, 0.5), ("b", 4.5, 1.0)], calc)
    labels, exp, calc_out, log, averaging = bt.parse(e, c)
    assert list(labels) == ["a", "b"]
    assert averaging == "linear"
    np.testing.assert_allclose(exp, [[3.0, 0.5, 0], [4.5, 1.0, 0]])
    np.testing.assert_allclose(calc_out, calc)  # frame column dropped
    assert "JCOUPLINGS" in log


def test_parse_noe_defaults_to_power6_and_transforms(write_dataset):
    r = np.array([[2.0, 3.0], [4.0, 5.0]])
    e, c = write_dataset("# DATA=NOE", [("a", 3.0, 0.3), ("b", 4.0, 0.2)], r)
    labels, exp, calc, log, averaging = bt.parse(e, c)
    assert averaging == "power_6"
    np.testing.assert_allclose(calc, r ** -6)
    np.testing.assert_allclose(exp[:, 0], np.array([3.0, 4.0]) ** -6)
    # error propagation: sigma' = p * r^-p * sigma / r
    np.testing.assert_allclose(exp[:, 1], 6 * np.array([3.0, 4.0]) ** -6 * np.array([0.3, 0.2]) / np.array([3.0, 4.0]))


@pytest.mark.parametrize("bound,expected_linear,expected_power", [
    ("UPPER", 1.0, -1.0),
    ("LOWER", -1.0, 1.0),
])
def test_parse_bounds_are_swapped_by_power_averaging(write_dataset, bound, expected_linear, expected_power):
    calc = np.array([[2.0], [3.0]])
    e, c = write_dataset("# DATA=JCOUPLINGS BOUND=%s" % bound, [("a", 3.0, 0.3)], calc, name="lin")
    assert bt.parse(e, c)[1][0, 2] == expected_linear
    e, c = write_dataset("# DATA=NOE BOUND=%s" % bound, [("a", 3.0, 0.3)], calc, name="noe")
    assert bt.parse(e, c)[1][0, 2] == expected_power


def test_parse_explicit_averaging(write_dataset):
    e, c = write_dataset("# DATA=NOE", [("a", 2.0, 0.1)], [[2.0], [4.0]])
    *_, averaging = bt.parse(e, c, averaging="linear")
    assert averaging == "linear"
    *_, averaging = bt.parse(e, c, averaging="power_3")
    assert averaging == "power_3"


def test_parse_reads_zipped_calc_file(tmp_path):
    import zipfile
    exp_file = tmp_path / "x.exp.dat"
    exp_file.write_text("# DATA=SAXS\nq1 1.0 0.1\nq2 2.0 0.1\n")
    calc_zip = tmp_path / "x.calc.dat.zip"
    with zipfile.ZipFile(calc_zip, "w") as zf:
        zf.writestr("x.calc.dat", "0 1.0 2.0\n1 1.5 2.5\n")
    labels, exp, calc, log, averaging = bt.parse(str(exp_file), str(calc_zip))
    np.testing.assert_allclose(calc, [[1.0, 2.0], [1.5, 2.5]])


@pytest.mark.parametrize("header", ["DATA=NOE", "# DATA=FOO", "# no data type"])
def test_parse_rejects_bad_header(write_dataset, header):
    e, c = write_dataset(header, [("a", 1.0, 0.1)], [[1.0]])
    with pytest.raises(ValueError):
        bt.parse(e, c)


def test_parse_rejects_bad_bound(write_dataset):
    e, c = write_dataset("# DATA=NOE BOUND=SIDEWAYS", [("a", 1.0, 0.1)], [[1.0]])
    with pytest.raises(ValueError):
        bt.parse(e, c)


def test_parse_rejects_bad_averaging(write_dataset):
    e, c = write_dataset("# DATA=NOE", [("a", 1.0, 0.1)], [[1.0]])
    with pytest.raises(ValueError):
        bt.parse(e, c, averaging="power_5")


def test_parse_rejects_mismatched_sizes(write_dataset):
    e, c = write_dataset("# DATA=CS", [("a", 1.0, 0.1), ("b", 1.0, 0.1)], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="must match"):
        bt.parse(e, c)


def test_parse_rejects_wrong_number_of_columns(tmp_path):
    e = tmp_path / "e.dat"
    e.write_text("# DATA=CS\na 1.0\nb 2.0\n")
    c = tmp_path / "c.dat"
    c.write_text("0 1 2\n")
    with pytest.raises(ValueError, match="LABEL VALUE ERROR"):
        bt.parse(str(e), str(c))


# ---------------------------------------------------------------- subsample
def test_subsample():
    label = np.array(["a", "b", "c"])
    exp = np.arange(9.0).reshape(3, 3)
    calc = np.arange(12.0).reshape(4, 3)
    l2, e2, c2, log = bt.subsample(label, exp, calc, [0, 2], [1, 2])
    assert list(l2) == ["b", "c"]
    np.testing.assert_array_equal(e2, exp[[1, 2]])
    np.testing.assert_array_equal(c2, calc[[0, 2]][:, [1, 2]])
    # empty selections leave data untouched
    l3, e3, c3, log = bt.subsample(label, exp, calc, [], [])
    assert c3.shape == (4, 3) and log == ""


# ---------------------------------------------------------------- fit_and_scale
def _scale_problem(slope, intercept, ndata=20, nsamples=50, seed=1):
    rng = np.random.default_rng(seed)
    calc = rng.uniform(1, 10, size=(nsamples, ndata))
    w = np.ones(nsamples) / nsamples
    avg = calc.mean(axis=0)
    exp = np.column_stack([slope * avg + intercept, np.full(ndata, 0.1), np.zeros(ndata)])
    return exp, calc, w


def test_fit_and_scale_scale_recovers_slope():
    exp, calc, w = _scale_problem(2.5, 0.0)
    orig = calc.copy()
    calc_avg, log = bt.fit_and_scale(exp, calc, w, "scale")
    np.testing.assert_allclose(calc, 2.5 * orig)  # calc is modified in place
    np.testing.assert_allclose(calc_avg, exp[:, 0])
    assert calc_avg.shape == (exp.shape[0],)
    assert "Slope=  2.5000" in log


def test_fit_and_scale_scale_offset_recovers_both():
    exp, calc, w = _scale_problem(0.5, 3.0)
    orig = calc.copy()
    calc_avg, log = bt.fit_and_scale(exp, calc, w, "scale+offset")
    np.testing.assert_allclose(calc, 0.5 * orig + 3.0)
    assert "Offset=  3.0000" in log


def test_fit_and_scale_no_leaves_data():
    exp, calc, w = _scale_problem(2.0, 1.0)
    orig = calc.copy()
    calc_avg, _ = bt.fit_and_scale(exp, calc, w, "no")
    np.testing.assert_array_equal(calc, orig)
    np.testing.assert_allclose(calc_avg, orig.mean(axis=0))


def test_fit_and_scale_rejects_unknown_fit():
    exp, calc, w = _scale_problem(1.0, 0.0)
    with pytest.raises(ValueError):
        bt.fit_and_scale(exp, calc, w, "quadratic")


# ---------------------------------------------------------------- calc_chi
def test_calc_chi_plain_average():
    exp = np.array([[1.0, 0.5, 0], [2.0, 1.0, 0]])
    calc = np.array([[2.0, 2.0], [2.0, 4.0]])
    w = np.array([0.5, 0.5])
    # calc avg = [2, 3]; diffs/sigma = [2, 1] -> mean of squares = 2.5
    assert bt.calc_chi(exp, calc, w) == pytest.approx(2.5)


def test_calc_chi_bounds_only_count_when_violated():
    calc = np.array([[2.0]])
    w = np.array([1.0])
    upper_ok = np.array([[3.0, 1.0, 1.0]])     # avg 2 <= 3: satisfied
    upper_bad = np.array([[1.0, 1.0, 1.0]])    # avg 2 > 1: violated
    lower_ok = np.array([[1.0, 1.0, -1.0]])    # avg 2 >= 1: satisfied
    lower_bad = np.array([[3.0, 1.0, -1.0]])   # avg 2 < 3: violated
    assert bt.calc_chi(upper_ok, calc, w) == 0.0
    assert bt.calc_chi(upper_bad, calc, w) == pytest.approx(1.0)
    assert bt.calc_chi(lower_ok, calc, w) == 0.0
    assert bt.calc_chi(lower_bad, calc, w) == pytest.approx(1.0)


# ---------------------------------------------------------------- calc_diffs
def test_calc_diffs():
    exp_avg = np.array([0.0, 0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 1.0, 1.0])
    bounds = np.array([0, 0, 1, -1])
    calc_avg = np.array([0.5, 2.0, 3.0, 3.0])  # 2nd: plain violation, 3rd: upper violated, 4th: lower ok
    chi2, rmsd, nviol, idx = bt.calc_diffs(exp_avg, sigma, bounds, calc_avg)
    assert nviol == 2
    assert sorted(idx.tolist()) == [1, 2]
    assert chi2 == pytest.approx((0.25 + 4 + 9 + 0) / 4)
    assert rmsd == pytest.approx(np.sqrt((0.25 + 4 + 9) / 4))


# ---------------------------------------------------------------- check_data
def test_check_data_counts_violations_in_both_directions():
    labels = np.array(["a", "b", "c"])
    calc = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    w = np.array([0.5, 0.5])
    exp = np.array([[5.0, 1.0, 0], [-5.0, 1.0, 0], [0.5, 1.0, 0]])
    log = bt.check_data(labels, exp, calc, w)
    # a and b deviate by 5 sigma (one above, one below), c is within 1 sigma
    assert "VIOLATIONS: 2" in log
    assert "WARNING" in log


# ---------------------------------------------------------------- calc_stats
def test_calc_stats_linear_identical_weights(tmp_path):
    rng = np.random.default_rng(0)
    calc = rng.normal(size=(30, 4))
    exp = np.column_stack([calc.mean(0) + 0.1, np.full(4, 0.2), np.zeros(4)])
    w = np.ones(30) / 30
    out = tmp_path / "stats.dat"
    stats, log = bt.calc_stats(np.array(list("abcd")), exp, calc, w, w, "linear", "no", str(out))
    assert stats[0] == pytest.approx(stats[3])
    assert stats[0] == pytest.approx(0.25)  # (0.1/0.2)^2
    text = out.read_text()
    assert text.startswith("# label exp_avg exp_sigma calc_avg calc_avg_rew violation")
    assert "# CHI2:" in text and "# ndata:" in text
    assert len([l for l in text.splitlines() if not l.startswith("#")]) == 4


def test_calc_stats_power_back_transform(write_dataset):
    # stats for NOE must be reported in distance units
    r = np.array([[3.0, 4.0], [3.0, 4.0]])
    e, c = write_dataset("# DATA=NOE", [("a", 3.5, 0.2), ("b", 4.0, 0.3)], r)
    labels, exp, calc, log, averaging = bt.parse(e, c)
    w = np.array([0.5, 0.5])
    stats, _ = bt.calc_stats(labels, exp, calc, w, w, averaging, "no", None)
    # only a deviates: (3 - 3.5)/0.2 = -2.5 -> chi2 = 6.25/2, rmsd = sqrt(0.25/2)
    assert stats[0] == pytest.approx(6.25 / 2)
    assert stats[1] == pytest.approx(np.sqrt(0.25 / 2))
    assert stats[2] == 1


def test_calc_stats_requires_optimized_weights():
    exp = np.array([[1.0, 1.0, 0.0]])
    calc = np.array([[1.0]])
    with pytest.raises(ValueError, match="fit"):
        bt.calc_stats(["a"], exp, calc, np.array([1.0]), [], "linear", "no", None)


def test_calc_stats_scale_offset_runs():
    exp, calc, w = _scale_problem(2.0, 1.0)
    stats, log = bt.calc_stats(np.arange(exp.shape[0]), exp, calc, w, w, "linear", "scale+offset", None)
    assert stats[0] == pytest.approx(0.0, abs=1e-12)
    assert "Slope=  2.0000/  2.0000" in log


# ---------------------------------------------------------------- standardize
def test_standardize_zscore():
    rng = np.random.default_rng(3)
    calc = rng.normal(5.0, 2.0, size=(1000, 3))
    exp = np.column_stack([np.array([5.5, 4.0, 6.0]), np.full(3, 0.5), np.zeros(3)])
    w = np.ones(1000) / 1000
    calc0, exp0 = calc.copy(), exp.copy()
    log, v1, v2 = bt.standardize(exp, calc, w, "zscore")
    np.testing.assert_allclose(calc.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(calc * v2 + v1, calc0)
    np.testing.assert_allclose(exp[:, 0] * v2 + v1, exp0[:, 0])
    np.testing.assert_allclose(exp[:, 1] * v2, exp0[:, 1])
    # chi2 is invariant under standardization
    assert bt.calc_chi(exp, calc, w) == pytest.approx(bt.calc_chi(exp0, calc0, w))


def test_standardize_minmax():
    calc = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    exp = np.array([[5.0, 1.0, 0], [25.0, 2.0, 0]])
    log, v1, v2 = bt.standardize(exp, calc, np.ones(3) / 3, "minmax")
    np.testing.assert_allclose(calc.min(0), 0.0)
    np.testing.assert_allclose(calc.max(0), 1.0)
    np.testing.assert_allclose(exp[:, 0], [0.5, 0.75])
    np.testing.assert_allclose(v1, [0.0, 10.0])
    np.testing.assert_allclose(v2, [10.0, 20.0])


def test_standardize_none_and_invalid():
    calc = np.array([[1.0], [2.0]])
    exp = np.array([[1.5, 0.1, 0]])
    log, v1, v2 = bt.standardize(exp, calc, np.ones(2) / 2, "none")
    assert (v1, v2) == (0.0, 1.0)
    np.testing.assert_array_equal(calc, [[1.0], [2.0]])
    with pytest.raises(ValueError):
        bt.standardize(exp, calc, np.ones(2) / 2, "whiten")


def test_standardize_bme_returns_original_scale():
    calc = np.array([[2.0, 4.0], [6.0, 8.0]])
    exp = np.array([[4.0, 0.4, 0], [5.0, 1.0, 0]])
    log, v1, v2 = bt.standardize(exp, calc, np.ones(2) / 2, "bme")
    np.testing.assert_allclose(v2, [4.0, 5.0])
    np.testing.assert_allclose(exp[:, 0], 1.0)
