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

The command-line test runs each script as a program (``runpy``, with its own
argument parsing) in a fresh process, also inside zero guard bands: a plain run of
``gradient.py`` makes more than a dozen propagations and crashes 0.1.10 on NVIDIA.
Step 1.7 must make the guard unnecessary here too.
"""
import glob
import importlib.util
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import has_memoizejac

#: Absolute path of the repository root, and the folder of the example scripts.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(REPO, "example", "scripts")

#: Examples that have a physical check in this file. ``test_every_example_has_a_check``
#: fails if a script is added to example/scripts/ without one.
CHECKED = {"forward_surface", "forward_acquisitions", "gradient", "fwi", "tlfwi", "torch_gradient", "models"}

#: Examples that import ``PyFWI.fwi`` or ``PyFWI.tl_fwi``. In PyFWI 0.1.10 those modules need
#: ``scipy.optimize.optimize.MemoizeJac``, missing from current SciPy, so with current SciPy
#: these examples cannot run until step 1.1; with SciPy 1.7.3 (reference environment) they must.
NEEDS_FWI = {"fwi", "tlfwi"}
FWI_IMPORT_XFAIL = pytest.mark.xfail(condition=not has_memoizejac(), strict=True,
                                     raises=(ImportError, AssertionError),
                                     reason="PyFWI.fwi needs MemoizeJac, missing from current SciPy (step 1.1)")


def example_names():
    """Names (file names without ``.py``) of the example scripts, in alphabetical order."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(EXAMPLES_DIR, "*.py")) if not os.path.basename(p).startswith("_"))


#: Examples that import ``PyFWI.model_dataset``, which imports ``numpy.lib.function_base``
#: (removed in NumPy 2) in PyFWI 0.1.10: they cannot run with NumPy 2 until step 1.1.
NEEDS_MODEL_DATASET = {"models"}
MODEL_DATASET_XFAIL = pytest.mark.xfail(condition=int(np.__version__.split(".")[0]) >= 2, strict=True,
                                        raises=(ImportError, AssertionError),
                                        reason="PyFWI.model_dataset does not import with NumPy 2 (step 1.1)")

#: Examples that need PyTorch. The CPU CI does not install it (it is large), so they are skipped there.
NEEDS_TORCH = {"torch_gradient"}
TORCH_SKIP = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")


def example_params():
    """``example_names()`` as pytest parameters, with the marks of ``NEEDS_FWI``, ``NEEDS_TORCH`` and
    ``NEEDS_MODEL_DATASET``."""
    def marks(name):
        return ([FWI_IMPORT_XFAIL] * (name in NEEDS_FWI) + [TORCH_SKIP] * (name in NEEDS_TORCH)
                + [MODEL_DATASET_XFAIL] * (name in NEEDS_MODEL_DATASET))
    return [pytest.param(n, marks=marks(n)) for n in example_names()]


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


@pytest.mark.parametrize("name", example_params())
def test_example_has_docstring_and_main(name):
    module = load_example(name)
    assert module.__doc__ and "How to run it" in module.__doc__
    params = inspect.signature(module.main).parameters
    assert list(params)[:2] == ["size", "show"]
    assert params["size"].default == "small" and params["show"].default is False
    assert {"small", "full"} <= set(module.SIZES)


#: Program that runs an example script as ``__main__`` inside zero guard bands.
#: argv: <test folder> <script> <script arguments...>
GUARDED_PROGRAM = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
from oob_guard import zero_guard_bands
script, sys.argv = sys.argv[2], sys.argv[2:]
with zero_guard_bands():
    runpy.run_path(script, run_name="__main__")
"""


@pytest.mark.parametrize("name", example_params())
def test_example_runs_as_a_program(opencl_platform, name):
    env = {**os.environ, "MPLBACKEND": "Agg"}  # never open a window
    script = os.path.join(EXAMPLES_DIR, f"{name}.py")
    r = subprocess.run([sys.executable, "-c", GUARDED_PROGRAM, os.path.dirname(__file__), script, "--size", "small",
                        "--no-show"], check=False, capture_output=True, text=True, timeout=600, env=env)
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


def test_gradient(run_example):
    r = run_example("gradient")
    true, gradients = r["true"], r["gradient"]
    shape = true["vp"].shape
    assert r["misfit"] > 0
    for chpr, g in gradients.items():
        for name in ("vp", "vs", "rho"):
            assert g[name].shape == shape and np.isfinite(g[name]).all(), (chpr, name)
    # Sign: minus the vp gradient (the steepest-descent update) is positive inside the faster
    # inclusion and larger there than in the rest of the model.
    update, inside = -gradients[20]["vp"], true["vp"] == 3000.0
    assert update[inside].mean() > 0 and update[inside].mean() > update[~inside].mean()
    # Checkpoints: a few percent already gives the gradient of dense checkpointing.
    reference = gradients[20]["vp"]
    cos5 = np.dot(gradients[5]["vp"].ravel(), reference.ravel()) / np.linalg.norm(gradients[5]["vp"]) / np.linalg.norm(reference)
    assert cos5 > 0.99999
    # Parameterisations: lam/mu/rho converted with grad_lmd_to_vd equals the vp/vs/rho gradient.
    for name in ("vp", "vs", "rho"):
        np.testing.assert_allclose(r["gradient_lmd_as_dv"][name], gradients[20][name], rtol=1e-5,
                                   atol=1e-6 * np.abs(gradients[20][name]).max(), err_msg=name)
    assert set(r["gradient_lmd"]) == {"lam", "mu", "rho"}


@FWI_IMPORT_XFAIL
def test_fwi(run_example):
    r = run_example("fwi")
    true, start, estimate = r["true"], r["start"], r["estimate"]
    assert set(estimate) == {"vp", "vs", "rho"}
    assert all(np.isfinite(v).all() and v.shape == true["vp"].shape for v in estimate.values())
    # The inversion brings vp closer to the true model (by about 20 % in the small size) ...
    assert r["error_end"] < 0.9 * r["error_start"]
    # ... by raising it inside the faster inclusion more than anywhere else ...
    update, inside = estimate["vp"] - start["vp"], true["vp"] == 3000.0
    assert update[inside].mean() > 0 and update[inside].mean() > update[~inside].mean()
    # ... and only vp is inverted (k_0=1, k_end=2).
    np.testing.assert_array_equal(estimate["vs"], start["vs"])
    np.testing.assert_array_equal(estimate["rho"], start["rho"])
    assert np.isfinite(r["misfit"]).all()


@FWI_IMPORT_XFAIL
def test_tlfwi(run_example):
    r = run_example("tlfwi")
    dm, changed, true_dm = r["dm"]["vp"], r["changed"], r["true_dm"]
    assert dm.shape == true_dm.shape and np.isfinite(dm).all()
    # vp dropped in the changed zone: the estimate is negative there (about half of the true -10 %) ...
    assert dm[changed].mean() < 0.25 * true_dm[changed].mean() < 0
    # ... much larger there than anywhere else ...
    assert np.abs(dm[changed]).mean() > 3 * np.abs(dm[~changed]).mean()
    # ... and it looks like the true change.
    assert np.corrcoef(dm.ravel(), true_dm.ravel())[0, 1] > 0.5


@TORCH_SKIP
def test_torch_gradient(run_example):
    r = run_example("torch_gradient")
    assert r["loss"] > 0
    for name in ("vp", "vs", "rho"):
        torch_g, pyfwi_g = r["torch_gradient"][name], r["pyfwi_gradient"][name]
        assert torch_g.shape == r["true"]["vp"].shape and np.isfinite(torch_g).all()
        # For an L2 loss, autograd through torchfwi.Fwi gives PyFWI's own gradient.
        np.testing.assert_allclose(torch_g, pyfwi_g, rtol=1e-5, atol=1e-6 * np.abs(pyfwi_g).max(), err_msg=name)
    update, inside = -r["torch_gradient"]["vp"], r["true"]["vp"] == 3000.0
    assert update[inside].mean() > 0 and update[inside].mean() > update[~inside].mean()


def _total_variation(image):
    """Sum of the absolute differences between neighbouring grid points (how rough an image is)."""
    return float(np.abs(np.diff(image, axis=0)).sum() + np.abs(np.diff(image, axis=1)).sum())


@MODEL_DATASET_XFAIL
def test_models(run_example):
    results = run_example("models")
    elastic, rock_physics = {"vp", "vs", "rho"}, {"phi", "cc", "sw"}
    for name, versions in results.items():
        base, monitor, smooth = versions["baseline"], versions["monitor"], versions["smooth"]
        assert set(base) in (elastic, rock_physics), name
        assert all(v.shape == (100, 100) and v.dtype == np.float32 and np.isfinite(v).all() for v in base.values())
        # For these models smoothing=1 returns the homogeneous background, without the anomalies.
        for prop in base:
            assert _total_variation(smooth[prop]) == 0 < _total_variation(base[prop]), (name, prop)
        # A monitor changes a small part of the model only.
        if monitor is not None:
            changed = np.zeros((100, 100), bool)
            for prop in base:
                changed |= monitor[prop] != base[prop]
            assert 0 < changed.mean() < 0.1, (name, changed.mean())
    assert results["perturbation_dv"]["monitor"] is None  # this model has no monitor version
    # Units differ: louboutin's density is in g/cm3, yang's in kg/m3.
    assert results["louboutin"]["baseline"]["rho"].max() < 10 < results["yang"]["baseline"]["rho"].min()
