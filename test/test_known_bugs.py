"""Known bugs of PyFWI 0.1.10, written as strict expected failures.

Each test below states the *correct* behaviour and is marked
``xfail(strict=True)``. While the bug exists the test fails and pytest reports
"xfailed". When a later step fixes the bug, the test passes, ``strict=True``
turns that into a failure ("XPASS(strict)"), and the marker must be removed in
the same commit as the fix. That way a fix can never land silently, and a
regression can never come back silently.

Race conditions are the exception: whether a race changes the result depends on
thread scheduling, so a racy test may pass by chance. Those tests use
``strict=False`` and say so in their reason.

Fixed bugs keep their test, without the marker, so that they cannot come back; the
table below marks them "(done)". In the reference environment the tests run against
the published 0.1.10, so the tests of bugs fixed in the fork are expected to fail there
(``condition=REFERENCE``) unless the bug never showed up with the old dependencies.

Where an ``xfail`` uses ``raises=``, the test only counts as "expected failure"
if it fails with that exception type, so it cannot pass for the wrong reason.
Bugs that depend on the environment (for example imports that only break with
NumPy 2) use ``condition=``: they are expected to fail only where they happen,
and must pass everywhere else.

| Test | Bug | Fixed in roadmap step |
| --- | --- | --- |
| ``test_stencils_do_not_read_outside_buffers`` | stencils read past the end / before the start of the field buffers | 1.7 / 2.4 |
| ``test_kernels_do_not_write_outside_buffers`` | crosswell and right-well ``Adj_injSrc`` accumulate past the end of the adjoint fields | 2.4 |
| ``test_receiver_loops_bound_the_receiver_index`` | 5 receiver loops have no bound on the receiver index | 2.4 |
| ``test_default_geometry_is_deterministic`` | race conditions at source/receiver coincidences (non-strict: probabilistic) | 3.4 |
| ``test_chpr_zero_gives_the_checkpointed_gradient`` | ``chpr=0`` (the default) rebuilds the field backwards through the PML: wrong gradient | 4.2 |
| ``test_lbfgs_iterates_with_physically_scaled_data`` | absolute ``pgtol = 1e-8``: L-BFGS stops at iteration 0 on realistic data | 2.4 |
| ``test_pml_profile_does_not_depend_on_numpy`` | NumPy 2 type promotion changes the PML profile in the last bit | 1.1 (done) |
| ``test_source_injection_does_not_excite_grid_noise`` | the source is added once every ``dt_scale`` steps: grid-scale noise at dh of 4-6 m | 2.4 |
| ``test_crosswell_vz_seismogram_records_vz`` | the crosswell ``vz`` seismogram stores ``vx`` | 2.4 |
| ``test_crosswell_kernels_assign_vz_to_vz_seismograms`` | same bug, lines 27 and 74 of ``elastic_crosswell.cl`` | 2.4 |
| ``test_dt_define_is_not_truncated`` | ``#define dt`` keeps only 6 decimals | 2.4 |
| ``test_float_constants_have_the_f_suffix`` | literals without ``f`` make the kernels compute in FP64 | 2.4 |
| ``test_device_choice_works_with_a_single_device_platform`` | ``inpa['device']`` fails on single-device platforms (PyOpenCL 2022 only) | 1.2 |
| ``test_optimizer_modules_import`` | ``MemoizeJac`` no longer exported by SciPy | 1.1 (done) |
| ``test_numpy2_modules_import`` | ``numpy.lib.function_base`` removed in NumPy 2 | 1.1 (done) |
| ``test_model_dataset_imports_without_segyio`` | ``segyio`` imported at module level, never declared | 1.1 |
| ``test_pyopencl_is_a_declared_dependency`` | ``pyopencl`` missing from ``install_requires`` | 1.1 |
| ``test_wheel_contains_torch_and_data`` | the wheel lacks ``PyFWI/torch`` and ``PyFWI/data`` | 1.1 |
"""
import importlib
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from helpers import forward_and_gradient, kernel_sources, pml_matches_0_1_10, tiny_case

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))




def _pyopencl_rejects_single_device_answer():
    """True if this PyOpenCL fails when ``PYOPENCL_CTX`` names the device of a single-device platform.

    PyFWI 0.1.10 sets ``PYOPENCL_CTX = "<platform>:<device>"`` when ``inpa['device']`` is
    given. PyOpenCL 2022.2.4 (reference environment) does not consume the device answer
    when the platform has a single device and raises "not all provided choices were
    used"; PyOpenCL 2026.1.4 (dev) accepts it. Returns False when there is no OpenCL.
    """
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
    except Exception:  # noqa: BLE001
        return False
    for i, p in enumerate(platforms):
        if len(p.get_devices()) == 1:
            try:
                cl.create_some_context(interactive=False, answers=[str(i), "0"])
            except cl.Error as e:
                return "not all provided choices" in str(e)
            return False
    return False


def _pyopencl_error():
    """Base class of PyOpenCL exceptions (its RuntimeError is *not* the built-in one), or Exception."""
    try:
        import pyopencl as cl
    except ImportError:
        return Exception
    return cl.Error


#: True where the single-device selection bug shows up (it depends on the PyOpenCL version).
SINGLE_DEVICE_BUG = _pyopencl_rejects_single_device_answer()


def _defines(source):
    """Return ``{name: value}`` for every ``#define NAME VALUE`` line of a generated kernel source."""
    return dict(re.findall(r"#define\s+(\w+)\s+(\S+)", source))


# ---------------------------------------------------------------------------------------------
# Kernel arithmetic and recording
# ---------------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="0.1.10 stencils read outside the field buffers (row Nz, and i-3 "
                                       "with i=2); with NaN guard bands the outputs become NaN")
def test_stencils_do_not_read_outside_buffers(opencl_platform):
    from oob_guard import nan_guard_bands

    with nan_guard_bands():
        case = tiny_case()
        d_obs, grad = forward_and_gradient(case)
    assert all(np.isfinite(v).all() for v in d_obs.values()), "seismogram has NaN: read past a buffer"
    assert all(np.isfinite(v).all() for v in grad.values()), "gradient has NaN: read past a buffer"


@pytest.mark.parametrize("acq_type", [
    pytest.param(0, marks=pytest.mark.xfail(strict=True, reason="Adj_injSrc of elastic_crosswell.cl has no receiver "
                                                                "bound: it reads past the residual and accumulates "
                                                                "past the end of the adjoint fields")),
    1,  # the surface branches are bounded: no out-of-bounds write
    pytest.param(2, marks=pytest.mark.xfail(strict=True, reason="the right-well branch of Adj_injSrc in elastic.cl "
                                                                "has no receiver bound")),
])
def test_kernels_do_not_write_outside_buffers(opencl_platform, acq_type):
    from oob_guard import guard_bands

    # Bands of 1.0: a write, or an accumulation of a value read out of bounds, changes them.
    with guard_bands(fill=1.0) as report:
        forward_and_gradient(tiny_case(acq_type=acq_type, components=4))
    touched = report.touched()
    assert not touched, f"written outside {len(touched)} buffer(s) of shapes {[b.shape for b in touched]}"


@pytest.mark.xfail(strict=True, reason="elastic.cl lines 89 and 352 and elastic_crosswell.cl lines 24, 48 and 71 "
                                       "loop over every grid row without bounding the receiver index")
def test_receiver_loops_bound_the_receiver_index():
    # A receiver loop is a kernel branch 'if (i % dxr == 0 && ...)' or 'if (j % dxr == 0 && ...)'.
    # Its receiver index i/dxr (or ir) must be compared with the number of receivers, either in
    # the condition or in the next two lines; otherwise grid rows beyond the last receiver are
    # recorded into the following time sample and injected from past the end of the residual.
    unbounded = []
    for name in ("elastic.cl", "elastic_crosswell.cl", "elastic_surface.cl"):
        with open(os.path.join(REPO, "src", "PyFWI", name)) as f:
            lines = f.read().splitlines()
        for k, line in enumerate(lines):
            if re.search(r"if\s*\(.*dxr\s*==\s*0", line):
                window = "\n".join(lines[k:k + 3])
                if not re.search(r"(ir|[ij]\s*/\s*dxr)\s*<", window):
                    unbounded.append(f"{name}:{k + 1}")
    assert not unbounded, f"receiver loops without an index bound: {unbounded}"


@pytest.mark.xfail(strict=False, reason="race conditions: with the default geometry of acq_parameters a source sits on "
                                        "a receiver cell (acq_type 1) and two receivers share a cell where the wells "
                                        "cross the surface line (acq_type 2); non-strict because a race may happen to "
                                        "give equal results")
@pytest.mark.parametrize("acq_type", [1, 2])
def test_default_geometry_is_deterministic(opencl_platform, acq_type):
    from oob_guard import zero_guard_bands

    runs = []
    for _ in range(3):
        with zero_guard_bands():  # rule out the out-of-bounds effects: only the races remain
            d, g = forward_and_gradient(tiny_case(acq_type=acq_type, components=4, ns=2, race_free=False))
        runs.append({**d, **g})
    for other in runs[1:]:
        for k in runs[0]:
            np.testing.assert_array_equal(other[k], runs[0][k], err_msg=k)


@pytest.mark.xfail(strict=True, reason="with chpr=0 (the default of WavePropagator) only two snapshots are kept and "
                                       "the forward field is rebuilt backwards through the absorbing boundary over "
                                       "the whole recording, which is not exact: the gradient is wrong")
def test_chpr_zero_gives_the_checkpointed_gradient(opencl_platform):
    from oob_guard import zero_guard_bands

    gradients = {}
    for chpr in (0, 20):
        with zero_guard_bands():
            gradients[chpr] = forward_and_gradient(tiny_case(ns=2, chpr=chpr))[1]["vp"]
    a, b = gradients[0].ravel(), gradients[20].ravel()
    cosine = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    assert cosine > 0.999, f"cosine similarity with the checkpointed gradient: {cosine:.3f}"


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FWI.lbfgs passes an absolute pgtol=1e-8 to L-BFGS-B; with a unit-amplitude "
                          "source the gradients are smaller than that, so it 'converges' at iteration 0")
def test_lbfgs_iterates_with_physically_scaled_data(opencl_platform, monkeypatch):
    import io
    from contextlib import redirect_stdout

    import PyFWI.fwi as fwi_module
    from oob_guard import zero_guard_bands

    iterations = []
    original = fwi_module.fmin_l_bfgs_b

    def traced(*args, **kwargs):
        x, f, info = original(*args, **kwargs)
        iterations.append(info["nit"])  # number of L-BFGS iterations of this stage
        return x, f, info

    monkeypatch.setattr(fwi_module, "fmin_l_bfgs_b", traced)
    with zero_guard_bands():
        case = tiny_case(ns=2, inpa_extra={"energy_balancing": True})
        d_obs = case.W.forward_modeling(case.model, False)
        fwi = fwi_module.FWI(d_obs, case.inpa, case.src, case.rec_loc, case.model["vp"].shape,
                             components=0, chpr=20, n_well_rec=case.n_well_rec)
        with redirect_stdout(io.StringIO()):  # PyFWI prints the misfit of every evaluation
            fwi(case.m0, method="lbfgs", freqs=[25], iter=[2], n_params=1, k_0=1, k_end=2)
    assert iterations and iterations[0] >= 1, f"L-BFGS iterations per stage: {iterations}"


def test_pml_profile_does_not_depend_on_numpy():
    # Fixed in step 1.1: pml_delta_calculation computes in float64 explicitly.
    assert pml_matches_0_1_10()


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="the source sample is added once every dt_scale internal steps (an impulse train); at "
                          "grid spacings of 4 to 6 m it excites a grid-scale checkerboard mode that never leaves")
def test_source_injection_does_not_excite_grid_noise(opencl_platform):
    from tools.math_baseline import homogeneous_run

    # 600 m homogeneous model, 5 m grid, 1.2 s: the wave should leave through the absorbing boundary.
    _, W = homogeneous_run(600.0, 5.0, 4, (300.0, 300.0), [(400.0, 300.0), (500.0, 300.0)], t=1.2, chpr=10)
    energy = [float(np.sum(W.W["taux"][:, :, 0, k].astype(np.float64) ** 2)) for k in range(W.nchp)]
    # Measured: 2e-2 with dt = 4 ms (dt_scale = 7); 2e-9 when the source is injected at every step.
    assert energy[-1] / max(energy) < 1e-6


@pytest.mark.xfail(strict=True, reason="elastic_crosswell.cl stores vx in the vz seismogram")
def test_crosswell_vz_seismogram_records_vz(opencl_platform):
    # components=2 records vx and vz; with an explosive source in an elastic medium they differ.
    case = tiny_case(acq_type=0, components=2)
    d = case.W.forward_modeling(case.model, False)
    assert np.abs(d["vx"]).max() > 0
    assert not np.array_equal(d["vz"], d["vx"]), "the vz seismogram is a copy of vx"


@pytest.mark.xfail(strict=True, reason="lines 27 and 74 of elastic_crosswell.cl assign vx[...] to seismogram_vzi")
def test_crosswell_kernels_assign_vz_to_vz_seismograms():
    with open(os.path.join(REPO, "src", "PyFWI", "elastic_crosswell.cl")) as f:
        source = f.read()
    rhs = re.findall(r"seismogram_vzi\[[^\]]*\]\s*=\s*(\w+)\[", source)  # array read on the right-hand side
    assert len(rhs) == 2
    assert rhs == ["vz", "vz"], f"seismogram_vzi is assigned from {rhs}"


@pytest.mark.xfail(strict=True, reason="'#define dt %f' keeps 6 decimals: 0.000571 instead of 0.000571428...")
def test_dt_define_is_not_truncated():
    dt = 0.004 / 7  # internal step for a user dt of 4 ms
    value = float(_defines(kernel_sources(8, dt=dt)[0])["dt"].rstrip("fF"))
    assert abs(value - dt) / dt < 1e-7, f"dt in the kernel is {value!r}, should be {dt!r}"


@pytest.mark.xfail(strict=True, reason="float constants are written without the f suffix, so they are double "
                                       "and the hot kernels compute in FP64")
def test_float_constants_have_the_f_suffix():
    defines = _defines(kernel_sources(8)[0])
    floats = {name: defines[name] for name in ("dt", "dx", "dz", "c1", "c2", "c3", "c4")}
    assert all(v.endswith("f") for v in floats.values()), floats


# ---------------------------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------------------------

@pytest.mark.xfail(condition=SINGLE_DEVICE_BUG, strict=True, raises=_pyopencl_error(),
                   reason="with PyOpenCL 2022, create_some_context reports a 'left over' device choice "
                          "on single-device platforms")
def test_device_choice_works_with_a_single_device_platform(single_device_platform):
    case = tiny_case(inpa_extra={"platform": single_device_platform, "device": 0})
    assert case.W.ctx.devices[0].platform.name == _platform_name(single_device_platform)


def _platform_name(index):
    """Name of the OpenCL platform with the given index."""
    import pyopencl as cl
    return cl.get_platforms()[index].name


# ---------------------------------------------------------------------------------------------
# Imports and packaging
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["PyFWI.fwi", "PyFWI.tl_fwi", "PyFWI.optimization"])
def test_optimizer_modules_import(module):
    # Fixed in step 1.1: PyFWI._scipy_compat provides MemoizeJac and an fmin_l_bfgs_b that
    # tolerates the removal of its iprint and disp arguments.
    importlib.import_module(module)


@pytest.mark.parametrize("module", ["PyFWI.seismic_io", "PyFWI.model_dataset"])
def test_numpy2_modules_import(module):
    # Fixed in step 1.1: unused imports of numpy.lib.function_base and numpy.core were removed.
    importlib.import_module(module)


@pytest.mark.xfail(strict=True, raises=ImportError, reason="segyio is imported at module level but is not a "
                                                           "declared dependency")
def test_model_dataset_imports_without_segyio(monkeypatch):
    # Pretend segyio is not installed, and force a fresh import of the modules that use it.
    monkeypatch.setitem(sys.modules, "segyio", None)
    for name in ("PyFWI.model_dataset", "PyFWI.seismic_io"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        importlib.import_module("PyFWI.model_dataset")
    except ImportError as e:
        if "segyio" in str(e):
            raise  # this bug
        pytest.skip(f"PyFWI.model_dataset fails to import for another reason: {e}")


@pytest.mark.xfail(strict=True, reason="pyopencl is only in build-system.requires, not in install_requires")
def test_pyopencl_is_a_declared_dependency():
    requires = importlib.metadata.requires("PyFWI") or []
    names = {re.split(r"[ ;<>=!~\[]", r, maxsplit=1)[0].lower() for r in requires}
    assert "pyopencl" in names, sorted(names)


@pytest.mark.skipif(importlib.util.find_spec("setuptools") is None, reason="setuptools is needed to build the wheel")
@pytest.mark.xfail(strict=True, reason="setup.py only packages PyFWI and the .cl files")
def test_wheel_contains_torch_and_data(tmp_path):
    # Build from a copy, so that the build artefacts never touch the working tree.
    for name in ("setup.py", "pyproject.toml", "README.md"):
        shutil.copy(os.path.join(REPO, name), tmp_path)
    shutil.copytree(os.path.join(REPO, "src"), tmp_path / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    build = subprocess.run([sys.executable, "-c", "from setuptools import build_meta as b; print(b.build_wheel('dist'))"],
                           cwd=tmp_path, check=True, capture_output=True, text=True)
    wheel = tmp_path / "dist" / build.stdout.strip().splitlines()[-1]
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert "PyFWI/wave_propagation.py" in names  # sanity check of the build itself
    assert "PyFWI/torch/wave_propagation.py" in names, "PyFWI/torch is missing from the wheel"
    assert "PyFWI/data/Dupuy2011_Moduli_data.mat" in names, "PyFWI/data is missing from the wheel"
