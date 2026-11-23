"""Unit tests for the Reweight class on small synthetic problems."""
import os

import numpy as np
import pytest

import BME


def gaussian_problem(nsamples=20000, ndata=1, seed=0, shift=0.5, sigma=0.5):
    rng = np.random.default_rng(seed)
    calc = rng.normal(size=(nsamples, ndata))
    exp = np.column_stack([calc.mean(0) + shift, np.full(ndata, sigma), np.zeros(ndata)])
    labels = np.array(["d%d" % i for i in range(ndata)])
    return labels, exp, calc


def make(name="rew", w0=(), **data):
    labels, exp, calc = gaussian_problem(**data)
    rew = BME.Reweight(name, w0=list(w0))
    rew.load_array(labels, exp.copy(), calc.copy())
    return rew, labels, exp, calc


# ------------------------------------------------------------ BME solution
@pytest.mark.parametrize("theta", [0.5, 1.0, 4.0])
def test_bme_gaussian_analytical_solution(theta):
    # For a Gaussian ensemble (mean mu, variance s2) the BME average is
    # mu + (m - mu) * s2 / (s2 + theta * sigma2)
    rew, labels, exp, calc = make(nsamples=50000)
    chi2_before, chi2_after, phi = rew.fit(theta)
    w = rew.get_weights()
    mu, s2 = calc[:, 0].mean(), calc[:, 0].var()
    m, sigma = exp[0, 0], exp[0, 1]
    expected = mu + (m - mu) * s2 / (s2 + theta * sigma ** 2)
    assert np.dot(w, calc[:, 0]) == pytest.approx(expected, abs=5e-3)
    assert chi2_after < chi2_before
    assert 0 < phi <= 1


def test_weights_are_normalized_and_positive():
    rew, *_ = make(ndata=3, nsamples=2000)
    rew.fit(1.0)
    w = rew.get_weights()
    assert w.shape == (2000,)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0)
    assert rew.get_iterations() > 0
    assert len(rew.get_lambdas()) == 3


def test_large_theta_keeps_prior_small_theta_fits_data():
    rew, *_ = make(ndata=2, nsamples=5000)
    _, chi2_big, phi_big = rew.fit(1e6)
    assert phi_big == pytest.approx(1.0, abs=1e-4)
    rew2, *_ = make(ndata=2, nsamples=5000)
    _, chi2_small, phi_small = rew2.fit(1e-3)
    assert chi2_small < 1e-3
    assert phi_small < phi_big


def test_result_independent_of_units():
    labels, exp, calc = gaussian_problem(ndata=2, nsamples=3000)
    r1 = BME.Reweight("u1")
    r1.load_array(labels, exp.copy(), calc.copy())
    r1.fit(2.0)
    exp2 = exp.copy()
    exp2[:, :2] *= 100.0
    exp2[:, 0] += 7.0
    r2 = BME.Reweight("u2")
    r2.load_array(labels, exp2, calc * 100.0 + 7.0)
    r2.fit(2.0)
    np.testing.assert_allclose(r1.get_weights(), r2.get_weights(), rtol=1e-5, atol=1e-10)


def test_satisfied_bound_leaves_weights_unchanged():
    labels, exp, calc = gaussian_problem(ndata=1, nsamples=1000)
    exp[0, 0] = calc.mean() + 5.0   # upper bound well above the average
    exp[0, 2] = 1.0
    rew = BME.Reweight("bound")
    rew.load_array(labels, exp, calc)
    chi2_before, chi2_after, phi = rew.fit(1.0)
    assert chi2_before == 0.0
    np.testing.assert_allclose(rew.get_weights(), rew.get_w0())


def test_violated_lower_bound_is_pushed_towards_bound():
    labels, exp, calc = gaussian_problem(ndata=1, nsamples=5000, shift=1.0, sigma=0.1)
    exp[0, 2] = -1.0   # lower bound above the average
    rew = BME.Reweight("lbound")
    rew.load_array(labels, exp, calc)
    rew.fit(0.1)
    assert np.dot(rew.get_weights(), calc[:, 0]) > calc[:, 0].mean() + 0.5
    assert rew.get_lambdas()[0] <= 0.0


def test_nonuniform_w0_is_used_as_prior():
    labels, exp, calc = gaussian_problem(ndata=1, nsamples=1000)
    w0 = np.linspace(1, 2, 1000)
    rew = BME.Reweight("w0", w0=w0)
    rew.load_array(labels, exp, calc)
    np.testing.assert_allclose(rew.get_w0(), w0 / w0.sum())
    rew.fit(1e7)
    np.testing.assert_allclose(rew.get_weights(), w0 / w0.sum(), rtol=1e-3)


def test_warm_start_reproduces_cold_start():
    rew, *_ = make(ndata=2, nsamples=3000)
    rew.fit(1.0)
    rew.fit(2.0)
    w_warm = rew.get_weights()
    rew2, *_ = make(ndata=2, nsamples=3000)
    rew2.fit(2.0)
    np.testing.assert_allclose(w_warm, rew2.get_weights(), rtol=1e-4, atol=1e-9)


def test_warm_start_without_previous_fit_raises():
    rew, *_ = make()
    with pytest.raises(ValueError, match="Warm start"):
        rew.fit(1.0, lambdas_init=False)


def test_fit_does_not_modify_loaded_data():
    rew, labels, exp, calc = make(ndata=2, nsamples=500)
    rew.fit(1.0)
    np.testing.assert_array_equal(rew.get_experiment(), exp)
    np.testing.assert_array_equal(rew.get_calculated(), calc)


def test_loading_after_fit_equals_loading_everything_first():
    # regression: data loaded after a fit used to be mixed with standardized data
    labels, exp, calc = gaussian_problem(ndata=4, nsamples=2000)
    a = BME.Reweight("a")
    a.load_array(labels[:2], exp[:2].copy(), calc[:, :2].copy())
    a.fit(1.0)
    a.load_array(labels[2:], exp[2:].copy(), calc[:, 2:].copy())
    res_a = a.fit(1.0)
    b = BME.Reweight("b")
    b.load_array(labels, exp.copy(), calc.copy())
    res_b = b.fit(1.0)
    np.testing.assert_allclose(res_a, res_b, rtol=1e-6)
    np.testing.assert_allclose(a.get_weights(), b.get_weights(), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("method", ["BER", "CHI2_L2", "CHI2_L1"])
def test_alternative_methods(method):
    rew, *_ = make(ndata=3, nsamples=200)
    chi2_before, chi2_after, phi = rew.fit(1.0, method=method)
    w = rew.get_weights()
    assert chi2_after < chi2_before
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert 0 < phi <= 1


# ------------------------------------------------------------ files
def test_load_and_predict_from_files(write_dataset, tmp_path):
    labels, exp, calc = gaussian_problem(ndata=3, nsamples=500)
    rows = [(l, v, s) for l, (v, s, _) in zip(labels, exp)]
    e, c = write_dataset("# DATA=CS", rows, calc)
    rew = BME.Reweight("files")
    rew.load(e, c)
    assert rew.get_ndata() == 3 and rew.get_nsamples() == 500
    chi2_before, chi2_after, phi = rew.fit(1.0)
    out = tmp_path / "pred.dat"
    stats = rew.predict(e, c, outfile=str(out))
    assert stats[0] == pytest.approx(chi2_before)
    assert stats[3] == pytest.approx(chi2_after)
    assert out.exists()
    rew.close()
    assert os.path.exists("files.log")


def test_load_with_use_samples_and_use_data(write_dataset):
    labels, exp, calc = gaussian_problem(ndata=4, nsamples=100)
    rows = [(l, v, s) for l, (v, s, _) in zip(labels, exp)]
    e, c = write_dataset("# DATA=CS", rows, calc)
    rew = BME.Reweight("sub")
    rew.load(e, c, use_samples=np.arange(50), use_data=[0, 2])
    assert rew.get_nsamples() == 50
    assert list(rew.get_labels()) == ["d0", "d2"]


def test_w0_size_mismatch_is_reported(write_dataset):
    labels, exp, calc = gaussian_problem(ndata=1, nsamples=100)
    e, c = write_dataset("# DATA=CS", [("d0", exp[0, 0], exp[0, 1])], calc)
    rew = BME.Reweight("mismatch", w0=np.ones(80))
    with pytest.raises(ValueError, match="w0"):
        rew.load(e, c)


# ------------------------------------------------------------ input checks
def test_w0_validation():
    with pytest.raises(ValueError):
        BME.Reweight("bad", w0=[1.0, -1.0])
    r = BME.Reweight("list", w0=[1, 1, 2])
    np.testing.assert_allclose(r.get_w0(), [0.25, 0.25, 0.5])


def test_load_array_validation():
    rew = BME.Reweight("v")
    with pytest.raises(ValueError):
        rew.load_array(["a"], np.zeros((1, 2)), np.zeros((5, 1)))
    with pytest.raises(ValueError):
        rew.load_array(["a"], np.zeros((1, 3)), np.zeros((5, 2)))
    with pytest.raises(ValueError):
        rew.load_array(["a", "b"], np.zeros((1, 3)), np.zeros((5, 1)))
    rew.load_array(["a"], np.array([[0.0, 1.0, 0.0]]), np.zeros((5, 1)))
    with pytest.raises(ValueError, match="samples"):
        rew.load_array(["b"], np.array([[0.0, 1.0, 0.0]]), np.zeros((6, 1)))


def test_invalid_method():
    rew, *_ = make()
    with pytest.raises(ValueError, match="known methods"):
        rew.fit(1.0, method="MAGIC")


def test_fit_without_data():
    with pytest.raises(ValueError, match="No data"):
        BME.Reweight("empty").fit(1.0)


def test_predict_before_fit_raises():
    rew, labels, exp, calc = make()
    with pytest.raises(ValueError, match="fit"):
        rew.predict_array(labels, exp, calc)


def test_set_lambdas_only_once():
    rew = BME.Reweight("lam")
    rew.set_lambdas(np.zeros(2))
    with pytest.raises(RuntimeError):
        rew.set_lambdas(np.zeros(2))


def test_ibme_results_require_ibme():
    rew = BME.Reweight("noibme")
    with pytest.raises(RuntimeError):
        rew.get_ibme_weights()
    with pytest.raises(RuntimeError):
        rew.get_ibme_stats()


def test_context_manager_closes_log():
    with BME.Reweight("ctx") as rew:
        assert not rew.log.closed
    assert rew.log.closed


def test_progress_handles_single_fold(capsys):
    BME.progress(0, 1)
    BME.progress(0, 5)
    assert "[" in capsys.readouterr().out


# ------------------------------------------------------------ iBME / scans
def test_ibme_recovers_population_with_unknown_scale_and_offset():
    # Two conformational states with different (non affinely related) profiles.
    # The "experiment" is a 80/20 mixture, the simulation samples 50/50, and the
    # calculated data are on a different scale/offset than the experiment.
    rng = np.random.default_rng(5)
    nsamples, ndata = 1000, 30
    x = np.linspace(0, 1, ndata)
    profile_a, profile_b = np.sin(3 * x) + 2, x ** 2 + 2
    state_a = np.arange(nsamples) < nsamples // 2
    truth = np.where(state_a[:, None], profile_a, profile_b) + 0.05 * rng.normal(size=(nsamples, ndata))
    target = 0.8 * profile_a + 0.2 * profile_b
    exp = np.column_stack([target, np.full(ndata, 0.02), np.zeros(ndata)])
    calc = (truth - 2.0) / 3.0
    rew = BME.Reweight("ibme")
    rew.load_array(np.arange(ndata), exp, calc)
    chi2_0, chi2_1, phi, calc0, calc1 = rew.ibme(theta=1, iterations=30, ftol=1e-4)
    assert chi2_0 > 10
    assert chi2_1 < 0.1 * chi2_0
    w = rew.get_weights()
    assert w.sum() == pytest.approx(1.0)
    assert w[state_a].sum() == pytest.approx(0.8, abs=0.05)
    assert 0 < phi < 1
    assert calc0.shape == calc1.shape == calc.shape
    assert len(rew.get_ibme_weights()) == len(rew.get_ibme_stats()) >= 1
    assert os.path.exists("ibme_%d.weights.dat" % (len(rew.get_ibme_stats()) - 1))


def test_theta_scan_runs_and_returns_one_of_thetas():
    rew, *_ = make(ndata=8, nsamples=400)
    thetas = [0.1, 1.0, 10.0]
    best = rew.theta_scan(thetas=thetas, nfold=2)
    assert best in thetas
    assert os.path.exists(os.path.join("theta_scan", "crossval_rew.png"))


def test_iterative_theta_scan_runs():
    rng = np.random.default_rng(2)
    calc = rng.normal(size=(300, 12)) + np.linspace(0, 3, 12)
    exp = np.column_stack([1.5 * calc.mean(0) + 0.3 + 0.2, np.full(12, 0.2), np.zeros(12)])
    rew = BME.Reweight("its")
    rew.load_array(np.arange(12), exp, calc)
    thetas = [1.0, 100.0]
    best = rew.iterative_theta_scan(thetas=thetas, nfold=2, iterations=5)
    assert best in thetas
    assert os.path.exists("crossval_its.png")
