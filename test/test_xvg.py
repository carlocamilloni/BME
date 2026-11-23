"""Tests for reweighting a single observable from a GROMACS .xvg file (bme-rg)."""
import os

import numpy as np
import pytest

import BME
import BME_rg
import BME_tools as bt

HEADER = """# This file was created by the test-suite
#                    :-) GROMACS - gmx gyrate, 2023.3 (-:
@    title "Radius of gyration (total and around axes)"
@    xaxis  label "Time (ps)"
@    yaxis  label "Rg (nm)"
@TYPE xy
@ legend on
@ s0 legend "Rg"
@ s1 legend "Rg\\sX\\N"
@ s2 legend "Rg\\sY\\N"
@ s3 legend "Rg\\sZ\\N"
"""


@pytest.fixture
def gyrate(tmp_path):
    """Write a gmx gyrate-like file; returns (path, time, rg)."""
    rng = np.random.default_rng(7)
    n = 4000
    rg = rng.normal(1.5, 0.1, n)
    time = np.arange(n) * 10.0
    path = tmp_path / "gyrate.xvg"
    with open(path, "w") as fh:
        fh.write(HEADER)
        for t, r in zip(time, rg):
            fh.write("%10g %11.6f %11.6f %11.6f %11.6f\n" % (t, r, 0.8 * r, 0.7 * r, 0.6 * r))
    rg = np.round(rg, 6)
    return str(path), time, rg


# ------------------------------------------------------------ reader
def test_read_xvg(gyrate):
    path, time, rg = gyrate
    t, v, legend = bt.read_xvg(path)
    np.testing.assert_allclose(t, time)
    np.testing.assert_allclose(v, rg)
    assert legend == "Rg"
    t, v, legend = bt.read_xvg(path, column=2)
    np.testing.assert_allclose(v, np.round(0.8 * rg, 6), atol=2e-6)
    assert legend == "Rg\\sX\\N"


def test_read_xvg_stops_at_first_dataset_and_without_legend(tmp_path):
    p = tmp_path / "two_sets.xvg"
    p.write_text("@ title \"x\"\n0 1.0\n1 2.0\n&\n0 5.0\n1 6.0\n")
    t, v, legend = bt.read_xvg(str(p))
    np.testing.assert_array_equal(v, [1.0, 2.0])
    assert legend is None


@pytest.mark.parametrize("content,match", [
    ("0 1.0\n1 2.0\n", "Column 3"),
    ("0 1.0 2.0\n1 2.0\n", "columns"),
    ("0 1.0\n1 abc\n", "numbers"),
    ("# only comments\n@ s0 legend \"Rg\"\n", "No data"),
])
def test_read_xvg_errors(tmp_path, content, match):
    p = tmp_path / "bad.xvg"
    p.write_text(content)
    column = 3 if match == "Column 3" else 1
    with pytest.raises(ValueError, match=match):
        bt.read_xvg(str(p), column=column)


# ------------------------------------------------------------ load_xvg
def test_load_xvg_matches_gaussian_solution(gyrate):
    path, time, rg = gyrate
    target, sigma, theta = 1.6, 0.02, 2.0
    rew = BME.Reweight("rg")
    t = rew.load_xvg(path, target, sigma)
    assert len(t) == len(rg)
    assert list(rew.get_labels()) == ["Rg"]
    rew.fit(theta)
    avg = np.dot(rew.get_weights(), rg)
    mu, s2 = rg.mean(), rg.var()
    expected = mu + (target - mu) * s2 / (s2 + theta * sigma ** 2)
    assert avg == pytest.approx(expected, abs=3e-3)


def test_load_xvg_time_window(gyrate):
    path, time, rg = gyrate
    rew = BME.Reweight("win")
    t = rew.load_xvg(path, 1.5, 0.05, begin=1000, end=2000)
    np.testing.assert_allclose(t, np.arange(1000, 2001, 10.0))
    np.testing.assert_allclose(rew.get_calculated()[:, 0], rg[100:201])
    with pytest.raises(ValueError, match="No frames"):
        BME.Reweight("empty").load_xvg(path, 1.5, 0.05, begin=1e9)


def test_load_xvg_bounds(gyrate):
    path, time, rg = gyrate
    # upper bound already satisfied: nothing changes
    rew = BME.Reweight("ub")
    rew.load_xvg(path, rg.mean() + 0.2, 0.01, bound="upper")
    c0, c1, phi = rew.fit(1.0)
    assert c0 == 0.0 and phi == pytest.approx(1.0)
    # lower bound above the average: average is pushed up
    rew = BME.Reweight("lb")
    rew.load_xvg(path, rg.mean() + 0.1, 0.01, bound="LOWER")
    rew.fit(0.1)
    assert np.dot(rew.get_weights(), rg) > rg.mean() + 0.05
    with pytest.raises(ValueError):
        BME.Reweight("bb").load_xvg(path, 1.5, 0.05, bound="middle")


def test_load_xvg_rejects_bad_sigma(gyrate):
    path, *_ = gyrate
    with pytest.raises(ValueError, match="sigma"):
        BME.Reweight("s").load_xvg(path, 1.5, 0.0)


def test_load_xvg_combines_with_other_data(gyrate):
    path, time, rg = gyrate
    rew = BME.Reweight("combo")
    rew.load_xvg(path, 1.6, 0.05)
    rew.load_xvg(path, 1.2, 0.05, column=2, label="RgX")
    assert rew.get_ndata() == 2
    assert list(rew.get_labels()) == ["Rg", "RgX"]
    c0, c1, phi = rew.fit(1.0)
    assert c1 < c0


# ------------------------------------------------------------ command line
def test_cli_writes_weights(gyrate, capsys):
    path, time, rg = gyrate
    assert BME_rg.main([path, "--target", "16", "--sigma", "0.2", "--target-units", "A",
                        "--theta", "2", "--out", "res", "--plot"]) == 0
    out = capsys.readouterr().out
    assert "target (average)   1.6000 +/- 0.0200 nm" in out
    data = np.loadtxt("res.weights.dat")
    np.testing.assert_allclose(data[:, 0], time)
    assert data[:, 1].sum() == pytest.approx(1.0, abs=1e-6)
    # same result as the library
    rew = BME.Reweight("ref")
    rew.load_xvg(path, 1.6, 0.02)
    rew.fit(2.0)
    np.testing.assert_allclose(data[:, 1], rew.get_weights(), rtol=1e-6)
    assert os.path.exists("res.png") and os.path.exists("res.log")


def test_cli_scan_and_w0(gyrate, tmp_path, capsys):
    path, time, rg = gyrate
    w0 = tmp_path / "w0.dat"
    np.savetxt(w0, np.column_stack([time, np.ones(len(time))]))
    assert BME_rg.main([path, "--target", "1.6", "--sigma", "0.02", "--scan", "--w0", str(w0)]) == 0
    out = capsys.readouterr().out
    assert "(prior)" in out and "N_eff" in out
    lines = [l for l in out.splitlines() if l.strip().startswith(("0.01", "1e+03"))]
    assert len(lines) == 2


def test_cli_warns_when_target_outside_sampled_range(gyrate, capsys):
    path, *_ = gyrate
    assert BME_rg.main([path, "--target", "5.0", "--sigma", "0.1", "--theta", "10"]) == 0
    assert "outside the range" in capsys.readouterr().out


def test_cli_reports_errors(tmp_path, capsys):
    assert BME_rg.main([str(tmp_path / "missing.xvg"), "--target", "1", "--sigma", "0.1"]) == 1
    assert "Error" in capsys.readouterr().err


# ------------------------------------------------------------ combined with other data
def write_cs(tmp_path, rg, ncs=6, stride=1, seed=3, name="cs"):
    rng = np.random.default_rng(seed)
    base = rng.uniform(50, 60, ncs)
    coup = rng.normal(0, 1.5, ncs)
    sub = rg[::stride]
    calc = base + np.outer(sub - sub.mean(), coup) + rng.normal(0, 0.3, (len(sub), ncs))
    exp_file = tmp_path / ("%s_exp.dat" % name)
    calc_file = tmp_path / ("%s_calc.dat" % name)
    with open(exp_file, "w") as fh:
        fh.write("# DATA=CS\n")
        for i in range(ncs):
            fh.write("CA_%d %.4f 0.3\n" % (i + 1, base[i] + 0.1 * coup[i]))
    with open(calc_file, "w") as fh:
        for k, row in enumerate(calc):
            fh.write("%d " % (k * stride) + " ".join("%.5f" % x for x in row) + "\n")
    return str(exp_file), str(calc_file), calc


def test_xvg_and_chemical_shifts_in_either_order(gyrate, tmp_path):
    path, time, rg = gyrate
    e, c, calc = write_cs(tmp_path, rg)
    a = BME.Reweight("cs_first")
    a.load(e, c)
    a.load_xvg(path, 1.6, 0.02)
    b = BME.Reweight("rg_first")
    b.load_xvg(path, 1.6, 0.02)
    b.load(e, c)
    ra, rb = a.fit(2.0), b.fit(2.0)
    np.testing.assert_allclose(ra, rb, rtol=1e-8)
    np.testing.assert_allclose(a.get_weights(), b.get_weights(), rtol=1e-6, atol=1e-14)
    # same as loading everything as one array
    labels = np.concatenate([a.get_labels()])
    exp_all = a.get_experiment()
    single = BME.Reweight("single")
    single.load_array(labels, exp_all, a.get_calculated())
    np.testing.assert_allclose(single.fit(2.0), ra, rtol=1e-8)
    # Rg statistics after the combined fit
    stats = a.predict_xvg(path, 1.6, 0.02)
    assert stats[3] == pytest.approx(((a.get_weights() @ rg - 1.6) / 0.02) ** 2)


def test_xvg_stride_matches_subsampled_chemical_shifts(gyrate, tmp_path):
    path, time, rg = gyrate
    e, c, calc = write_cs(tmp_path, rg, stride=10)
    rew = BME.Reweight("stride_err")
    rew.load(e, c)
    with pytest.raises(ValueError, match="same frames"):
        rew.load_xvg(path, 1.6, 0.02)
    t = rew.load_xvg(path, 1.6, 0.02, stride=10)
    np.testing.assert_allclose(t, time[::10])
    np.testing.assert_allclose(rew.get_calculated(columns=[-1])[:, 0], rg[::10])
    rew2 = BME.Reweight("use_samples")
    t2 = rew2.load_xvg(path, 1.6, 0.02, begin=500, stride=2, use_samples=[0, 1, 5])
    np.testing.assert_allclose(t2, time[50::2][[0, 1, 5]])


def test_cli_with_chemical_shifts(gyrate, tmp_path, capsys):
    path, time, rg = gyrate
    e, c, calc = write_cs(tmp_path, rg, stride=4)
    assert BME_rg.main([path, "--target", "1.6", "--sigma", "0.02", "--stride", "4", "--data", e, c,
                        "--rg-weight", "5", "--theta", "2", "--scan", "--out", "combo"]) == 0
    out = capsys.readouterr().out
    assert "additional data    cs_exp.dat: 6 data points" in out
    assert "chi2 cs_exp.dat" in out and "chi2 all data" in out
    data = np.loadtxt("combo.weights.dat")
    np.testing.assert_allclose(data[:, 0], time[::4])
    ref = BME.Reweight("ref")
    ref.load_xvg(path, 1.6, 0.02, stride=4, weight=5)
    ref.load(e, c)
    ref.fit(2.0)
    np.testing.assert_allclose(data[:, 1], ref.get_weights(), rtol=1e-6)
    # mismatching frames: clear error, no traceback
    assert BME_rg.main([path, "--target", "1.6", "--sigma", "0.02", "--data", e, c]) == 1
    assert "same frames" in capsys.readouterr().err
