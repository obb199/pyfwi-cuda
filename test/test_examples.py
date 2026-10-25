"""Tests of the example scripts in ``example/scripts/`` (roadmap step 0.3, test type EXA).

The examples are user-facing scripts that only use the public API of PyFWI, the
way a user would. They are part of the contract of the fork: a change that breaks
an example is a regression even if every unit test passes. Every example must

* have a module docstring and a ``main(size="small", show=False)`` function that
  returns its numerical results;
* run as a program: ``python example/scripts/<name>.py --size small --no-show``;
* pass its own physical check below when ``main(size="small")`` runs.

How the examples are run here
-----------------------------
Inside zero guard bands (``test/oob_guard.py``): the out-of-bounds accesses of
PyFWI 0.1.10 then have a defined result, the runs are reproducible, and a long
test session cannot hit the ``CL_OUT_OF_RESOURCES`` crash that plain runs of
0.1.10 cause on NVIDIA after a few propagations. With the guard, POCL gives the
same bits as the NVIDIA GPU, so these tests also run in the CPU CI. Once step 1.7
pads the buffers inside PyFWI itself, the guard will no longer be needed.

The command-line test runs each script plainly (no guard) in a fresh process.
"""
import glob
import importlib.util
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

#: Absolute path of the repository root, and the folder of the example scripts.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(REPO, "example", "scripts")

#: Examples that have a physical check in this file. ``test_every_example_has_a_check``
#: fails if a script is added to example/scripts/ without one.
CHECKED = {"forward_surface", "forward_acquisitions"}


def example_names():
    """Names (file names without ``.py``) of the example scripts, in alphabetical order."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(EXAMPLES_DIR, "*.py")) if not os.path.basename(p).startswith("_"))


def load_example(name):
    """Import ``example/scripts/<name>.py`` as a module (the folder is not a package)."""
    spec = importlib.util.spec_from_file_location(f"pyfwi_example_{name}", os.path.join(EXAMPLES_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_example(opencl_platform):
    """Return a function ``run(name)`` that runs ``main(size="small")`` of an example inside zero guard bands."""
    from oob_guard import zero_guard_bands

    def run(name):
        module = load_example(name)
        with zero_guard_bands():
            return module.main(size="small", show=False)
    return run


def first_breaks(gather, threshold=0.01):
    """Index of the first time sample of each trace whose amplitude exceeds ``threshold`` x the gather maximum.

    Parameters
    ----------
    gather : numpy.ndarray
        Traces of one shot, shape ``(nt, n_traces)``.
    threshold : float, optional
        Fraction of the largest absolute amplitude of the gather.

    Returns
    -------
    numpy.ndarray
        One sample index per trace (a rough "first arrival" pick).
    """
    level = threshold * np.abs(gather).max()
    return np.argmax(np.abs(gather) > level, axis=0)


def test_every_example_has_a_check():
    assert set(example_names()) == CHECKED


@pytest.mark.parametrize("name", example_names())
def test_example_has_docstring_and_main(name):
    module = load_example(name)
    assert module.__doc__ and "How to run it" in module.__doc__
    params = inspect.signature(module.main).parameters
    assert list(params)[:2] == ["size", "show"]
    assert params["size"].default == "small" and params["show"].default is False
    assert {"small", "full"} <= set(module.SIZES)


@pytest.mark.parametrize("name", example_names())
def test_example_runs_as_a_program(opencl_platform, name):
    env = {**os.environ, "MPLBACKEND": "Agg"}  # never open a window
    r = subprocess.run([sys.executable, os.path.join(EXAMPLES_DIR, f"{name}.py"), "--size", "small", "--no-show"],
                       check=False, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-3000:]


def test_forward_surface(run_example):
    r = run_example("forward_surface")
    data, src_loc, rec_loc, inpa = r["data"], r["src_loc"], r["rec_loc"], r["inpa"]
    nt, nr, ns = int(1 + inpa["t"] // inpa["dt"]), rec_loc.shape[0], src_loc.shape[0]
    assert set(data) == {"vx", "vz", "taux", "tauz", "tauxz"}
    for name, d in data.items():
        assert d.shape == (nt, nr * ns) and d.dtype == np.float32 and np.isfinite(d).all(), name
    # Physics: the direct wave reaches the receivers later the farther they are from the source.
    for s in range(ns):
        picks = first_breaks(data["taux"][:, s * nr:(s + 1) * nr])
        offsets = np.abs(rec_loc[:, 0] - src_loc[s, 0])
        assert np.corrcoef(picks, offsets)[0, 1] > 0.95, f"shot {s}"


def test_forward_acquisitions(run_example):
    from helpers import coincidences

    results = run_example("forward_acquisitions")
    assert set(results) == {"crosswell", "surface", "both"}
    for acq_type, name in enumerate(("crosswell", "surface", "both")):
        r = results[name]
        rec, src, n_well, inpa = r["rec_loc"], r["src_loc"], r["n_well_rec"], r["inpa"]
        nr, ns, nt = rec.shape[0], src.shape[0], int(1 + inpa["t"] // inpa["dt"])
        # Receiver bookkeeping: crosswell = one well; surface = no well; both = left well + surface + right well.
        if acq_type == 0:
            assert n_well == nr and len(set(rec[:, 0])) == 1
        elif acq_type == 1:
            assert n_well == 0 and len(set(rec[:, 1])) == 1
        else:
            assert 0 < 2 * n_well < nr
            assert len(set(rec[:n_well, 0])) == 1 and len(set(rec[-n_well:, 0])) == 1  # two vertical wells
            assert len(set(rec[n_well:-n_well, 1])) == 1                                # one horizontal line
        assert set(r["data"]) == {"taux", "tauz"}  # components=0: pressure
        assert r["data"]["taux"].shape == (nt, nr * ns) and np.isfinite(r["data"]["taux"]).all()
        # The example's geometry has none of the race points of 0.1.10 (cross-check with the test helper).
        nz = load_example("forward_acquisitions").SIZES["small"]["nz"]
        assert coincidences(src, rec, n_well, inpa["dh"], inpa["rec_dis"], nz) == \
            {"source_on_receiver": [], "shared_receiver_cell": []}, name
        # Physics + trace order: matched with trace_positions(), arrivals get later with distance.
        for s in range(ns):
            picks = first_breaks(r["data"]["taux"][:, s * nr:(s + 1) * nr])
            distance = np.hypot(*(r["traces"] - src[s]).T)
            assert np.corrcoef(picks, distance)[0, 1] > 0.95, (name, s)
    # The left-well traces of acq_type 2 really are reversed: in rec_loc order they correlate badly.
    r = results["both"]
    nr, n_well = r["rec_loc"].shape[0], r["n_well_rec"]
    picks = first_breaks(r["data"]["taux"][:, :nr])[:n_well]
    assert np.corrcoef(picks, np.hypot(*(r["rec_loc"][:n_well] - r["src_loc"][0]).T))[0, 1] < 0
