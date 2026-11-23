"""Shared pytest configuration for the BME test-suite.

* makes the repository importable when BME is not pip-installed
* forces a non-interactive matplotlib backend
* runs every test inside a temporary directory, so that the log files
  (``<name>.log``) and other outputs written by ``Reweight`` do not
  pollute the repository
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BME_DIR = os.path.dirname(TEST_DIR)
DATA_DIR = os.path.join(BME_DIR, "data")
REFERENCE_DIR = os.path.join(TEST_DIR, "reference")

if BME_DIR not in sys.path:
    sys.path.insert(0, BME_DIR)


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def write_dataset(tmp_path):
    """Write an experimental and a calculated file and return their paths.

    ``header`` is the first line of the experimental file, ``exp_rows`` a list
    of (label, value, sigma) and ``calc`` a (nsamples, ndata) array.
    """

    def _write(header, exp_rows, calc, name="ds"):
        exp_file = tmp_path / ("%s.exp.dat" % name)
        calc_file = tmp_path / ("%s.calc.dat" % name)
        with open(exp_file, "w") as fh:
            fh.write(header.rstrip("\n") + "\n")
            for label, val, sigma in exp_rows:
                fh.write("%s %r %r\n" % (label, float(val), float(sigma)))
        calc = np.asarray(calc, dtype=float)
        with open(calc_file, "w") as fh:
            fh.write("# frame values\n")
            for k, row in enumerate(calc):
                fh.write("%d " % k + " ".join(repr(float(x)) for x in row) + "\n")
        return str(exp_file), str(calc_file)

    return _write
