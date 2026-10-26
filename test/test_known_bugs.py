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
| ``test_stencils_do_not_read_outside_buffers`` | stencils read past the end / before the start of the field buffers (inside the allocation since 1.7; the reads before the start go in the ``fixed`` mode) | Phase 3 (halo cells) |
| ``test_kernels_do_not_write_outside_buffers`` | crosswell and right-well ``Adj_injSrc`` accumulate past the end of the adjoint fields | 2.4 (``fixed`` mode) |
| ``test_receiver_loops_bound_the_receiver_index`` | 5 receiver loops have no bound on the receiver index | 2.4 (``fixed`` mode) |
| ``test_default_geometry_is_deterministic`` | race conditions at source/receiver coincidences (non-strict: probabilistic) | 3.4 |
| ``test_chpr_zero_gives_the_checkpointed_gradient`` | ``chpr=0`` (the default) rebuilds the field backwards through the PML: wrong gradient | 4.2 |
| ``test_lbfgs_iterates_with_physically_scaled_data`` | absolute ``pgtol = 1e-8``: L-BFGS stops at iteration 0 on realistic data | 2.4 |
| ``test_kernel_files_are_closed`` | ``kernel_caller`` leaves the three ``.cl`` files open (``ResourceWarning``) | 2.4 (done) |
| ``test_pml_profile_does_not_depend_on_numpy`` | NumPy 2 type promotion changes the PML profile in the last bit | 1.1 (done) |
| ``test_source_injection_does_not_excite_grid_noise`` | the source is added once every ``dt_scale`` steps: grid-scale noise at dh of 4-6 m | 2.4 (``fixed`` mode) |
| ``test_seismograms_do_not_depend_on_the_source_sampling`` | same bug, seen in the seismograms: 3 to 6 % off at dh = 10 m, several times the signal at 5 m | 2.4 (``fixed`` mode) |
| ``test_residual_is_injected_where_it_was_recorded`` | the residual is injected one recorded sample off when rounding makes ``nt_ext`` differ from the recorded steps | 2.4 (``fixed`` mode) |
| ``test_adjoint_step_is_the_transpose_of_the_forward_step`` | the adjoint updates run in the wrong order (fork only: needs the engine) | 2.4 (``fixed`` mode) |
| ``test_adjoint_fields_are_damped_as_the_forward_ones`` | the adjoint kernels damp the adjoint fields with other combinations of the absorbing profiles | 2.4 (``fixed`` mode) |
| ``test_vp_gradient_agrees_with_finite_differences`` | ``Grad`` pairs the forward field with the adjoint field of two steps earlier, and the adjoint absorbing layer differs: 7 % off | 2.4 (``fixed`` mode) |
| ``test_gradient_shear_strain_is_that_of_the_forward_kernels`` | ``Grad`` computes the shear strain from ``vz`` twice, never from ``vx`` | 2.4 (``fixed`` mode) |
| ``test_vs_and_density_gradients_agree_with_finite_differences`` | same bug (and the two above): vs 60 % and density 135 % off | 2.4 (``fixed`` mode) |
| ``test_receivers_are_recorded_inside_the_seismograms`` | the same rounding records the last sample one row past the seismograms (``dt`` = 5 ms) | 2.4 (``fixed`` mode) |
| ``test_crosswell_vz_seismogram_records_vz`` | the crosswell ``vz`` seismogram stores ``vx`` | 2.4 (``fixed`` mode) |
| ``test_crosswell_kernels_assign_vz_to_vz_seismograms`` | same bug, lines 27 and 74 of ``elastic_crosswell.cl`` | 2.4 (``fixed`` mode) |
| ``test_dt_define_is_not_truncated`` | ``#define dt`` keeps only 6 decimals | 2.4 (``fixed`` mode) |
| ``test_finite_difference_coefficients_are_consistent`` | the order-8 coefficients sum to 1436/1434: waves 0.14 % too fast | 2.4 (``fixed`` mode) |
| ``test_float_constants_have_the_f_suffix`` | literals without ``f`` make the kernels compute in FP64 | 2.4 (``fixed`` mode) |
| ``test_device_choice_works_with_a_single_device_platform`` | ``inpa['device']`` fails on single-device platforms (PyOpenCL 2022 only) | 1.2 (done) |
| ``test_optimizer_modules_import`` | ``MemoizeJac`` no longer exported by SciPy | 1.1 (done) |
| ``test_numpy2_modules_import`` | ``numpy.lib.function_base`` removed in NumPy 2 | 1.1 (done) |
| ``test_model_dataset_imports_without_segyio`` | ``segyio`` imported at module level, never declared | 1.1 (done) |
| ``test_pyopencl_is_a_declared_dependency`` | ``pyopencl`` missing from ``install_requires`` | 1.1 (done) |
| ``test_wheel_contains_torch_and_data`` | the wheel lacks ``PyFWI/torch`` and ``PyFWI/data`` | 1.1 (done) |
"""
import contextlib
import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys
import zipfile

import numpy as np
import pytest

from helpers import active_source, forward_and_gradient, kernel_sources, pml_matches_0_1_10, running_reference, tiny_case

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#: True in the reference environment (published 0.1.10): fixed bugs are still expected to fail there.
REFERENCE = running_reference()

#: The two numeric modes of the fork (step 2.4): the legacy mode keeps the bugs of 0.1.10 on
#: purpose, the fixed mode corrects them (in the reference environment both are 0.1.10).
NUMERIC_MODES = [pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy mode: as 0.1.10")),
                 pytest.param("fixed", marks=pytest.mark.xfail(condition=REFERENCE, strict=True,
                                                               reason="0.1.10 has no fixed mode"))]


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

@contextlib.contextmanager
def _filled_outside_buffers(monkeypatch, fill):
    """Fill what lies just outside every device buffer with ``fill``; yield a function that lists the
    buffers written outside, given the propagator.

    0.1.10 (reference environment): guard bands around its buffers (``oob_guard``). The fork (step
    1.7): the margins it allocates itself around every buffer (``PyFWI.engine.propagation.MARGIN_FILL``),
    read back by ``WavePropagator.margins_changed``. Either way a kernel that reads outside a buffer
    sees ``fill``, and one that writes outside changes it.
    """
    from oob_guard import guard_bands

    if REFERENCE:
        with guard_bands(fill=fill) as report:
            yield lambda W: report.touched()
    else:
        import PyFWI.engine.propagation as engine
        monkeypatch.setattr(engine, "MARGIN_FILL", fill)
        yield lambda W: W.margins_changed()


@pytest.mark.xfail(strict=True, reason="the stencils read one row past the field edges (row Nz), in both "
                                       "numeric modes: that zero margin is the outer boundary of the grid")
@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
def test_stencils_do_not_read_outside_buffers(opencl_platform, monkeypatch, numerics):
    # 0.1.10: the stencils read outside the field buffers (row Nz, and i-3 with i=2): with NaN there,
    # the outputs become NaN. Step 1.7 keeps these reads inside the allocation, where they read 0.
    # The fixed mode of step 2.4 drops the zero-coefficient terms of order 4 (the reads before the
    # start) but keeps the read of row Nz: not updating the last row instead changes the outer
    # boundary and worsens the absorbing layer (measured: reflection 1.09 % -> 1.23 %, gradient
    # error against finite differences 31 % -> 38 %). Explicit halo cells come with Phase 3.
    with _filled_outside_buffers(monkeypatch, np.nan):
        case = tiny_case(inpa_extra={"numerics": numerics})
        d_obs, grad = forward_and_gradient(case)
    assert all(np.isfinite(v).all() for v in d_obs.values()), "seismogram has NaN: read past a buffer"
    assert all(np.isfinite(v).all() for v in grad.values()), "gradient has NaN: read past a buffer"


#: Acquisitions and numeric modes of the write test: in the legacy mode the crosswell and the right
#: well of the combined acquisition write outside the buffers, as 0.1.10 does; the fixed mode never.
_LEGACY_WRITES = pytest.mark.xfail(strict=True, reason="Adj_injSrc has no receiver bound in the crosswell kernels "
                                                       "and in the right well of elastic.cl: it reads past the "
                                                       "residual and accumulates past the adjoint fields")
WRITE_CASES = [pytest.param(0, "legacy", marks=_LEGACY_WRITES), (1, "legacy"),
               pytest.param(2, "legacy", marks=_LEGACY_WRITES)]
WRITE_CASES += [pytest.param(acq, "fixed", marks=pytest.mark.xfail(condition=REFERENCE and acq != 1, strict=True,
                                                                   reason="0.1.10 has no fixed mode"))
                for acq in (0, 1, 2)]


@pytest.mark.parametrize("acq_type, numerics", WRITE_CASES)
def test_kernels_do_not_write_outside_buffers(opencl_platform, monkeypatch, acq_type, numerics):
    # 1.0 outside the buffers: a write, or an accumulation of a value read out of bounds, changes it.
    with _filled_outside_buffers(monkeypatch, 1.0) as written_outside:
        case = tiny_case(acq_type=acq_type, components=4, inpa_extra={"numerics": numerics})
        forward_and_gradient(case)
    touched = written_outside(case.W)
    assert not touched, f"written outside {len(touched)} buffer(s): {touched}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_receiver_loops_bound_the_receiver_index(numerics):
    # 0.1.10: elastic.cl lines 89 and 352 and elastic_crosswell.cl lines 24, 48 and 71 loop over every
    # grid row without bounding the receiver index (the fork keeps them in the legacy branch).
    # A receiver loop is a kernel branch 'if (i % dxr == 0 && ...)' or 'if (j % dxr == 0 && ...)'.
    # Its receiver index i/dxr (or ir) must be compared with the number of receivers, either in
    # the condition or in the next two lines; otherwise grid rows beyond the last receiver are
    # recorded into the following time sample and injected from past the end of the residual.
    import PyFWI
    unbounded = []
    for name in ("elastic.cl", "elastic_crosswell.cl", "elastic_surface.cl"):
        with open(os.path.join(os.path.dirname(PyFWI.__file__), name)) as f:
            lines = active_source(f.read(), fixed=numerics == "fixed").splitlines()
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


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_source_injection_does_not_excite_grid_noise(opencl_platform, monkeypatch, numerics):
    # 0.1.10 adds the source sample once every dt_scale internal steps (an impulse train); at grid
    # spacings of 4 to 6 m it excites a grid-scale checkerboard mode that never leaves. The fixed
    # mode resamples the source to every internal step (PyFWI.engine.propagation.internal_source).
    from tools.math_baseline import homogeneous_run

    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    # 600 m homogeneous model, 5 m grid, 1.2 s: the wave should leave through the absorbing boundary.
    _, W = homogeneous_run(600.0, 5.0, 4, (300.0, 300.0), [(400.0, 300.0), (500.0, 300.0)], t=1.2, chpr=10)
    energy = [float(np.sum(W.W["taux"][:, :, 0, k].astype(np.float64) ** 2)) for k in range(W.nchp)]
    # Measured: 2e-2 with dt = 4 ms (dt_scale = 7); 2e-9 when the source is injected at every step
    # (dt = 0.5 ms), and 2.5e-9 in the fixed mode.
    assert energy[-1] / max(energy) < 1e-6


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_seismograms_do_not_depend_on_the_source_sampling(opencl_platform, numerics):
    # The same wavelet sampled at dt = 4 ms, or at the internal step dt = 4/7 ms (dt_scale = 1:
    # injected as it is at every step), must give the same seismograms at the recorded samples.
    # PyFWI adds the source sample as it is at each injection, so the second run puts 7 times more
    # source into every recorded interval: its seismograms are divided by 7. Both runs propagate
    # with the same internal step, so only the injection differs.
    def pressure(dt):
        case = tiny_case(t=0.4, inpa_extra={"dt": dt, "numerics": numerics})
        return np.asarray(case.W.forward_modeling(case.model, False)["taux"], np.float64)

    coarse, fine = pressure(0.004), pressure(0.004 / 7)[::7] / 7
    rows = min(len(coarse), len(fine))
    difference = np.linalg.norm(coarse[:rows] - fine[:rows]) / np.linalg.norm(fine[:rows])
    # Measured: 3.4e-2 in the legacy mode (8.7 with a 5 m grid, where the grid noise dominates);
    # 2.7e-4 in the fixed mode (5.2e-4 at 5 m), the error of interpolating the truncated Ricker.
    assert difference < 2e-3, f"relative difference {difference:.1e}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_residual_is_injected_where_it_was_recorded(opencl_platform, numerics):
    # 0.1.10 counts the residual samples down from nt_ext - 1 at the last internal step that is a
    # multiple of dt_scale. nt and nt_ext are rounded down separately: with dt = 4 ms, t = 0.4 s
    # gives nt_ext = 101 rows for 100 recorded steps, and every residual sample is injected one
    # recorded sample (7 internal steps) too early; t = 0.3 s has as many rows as recorded steps.
    # A residual that is zero after 200 ms must give the same gradient with both lengths: the
    # longer recording only adds zeros. With every step stored as a checkpoint (chpr=100) the
    # forward field is the same in both runs.
    gradients = []
    for t in (0.3, 0.4):
        case = tiny_case(t=t, chpr=100, inpa_extra={"numerics": numerics})
        case.W.forward_modeling(case.m0, False)
        rng = np.random.default_rng(1)
        res = {}
        for par in ("vx", "vz", "taux", "tauz", "tauxz"):
            res[par] = np.zeros((case.W.nt_ext, case.W.nr * case.W.ns), np.float32)
            res[par][5:50] = rng.standard_normal((45, case.W.nr * case.W.ns))
        gradients.append(case.W.gradient(res, False))
    for key, short in gradients[0].items():
        difference = np.linalg.norm(gradients[1][key] - short) / np.linalg.norm(short)
        # Measured on vp: 0.54 in 0.1.10 and in the legacy mode; 0 (bit for bit) in the fixed mode.
        assert difference < 1e-5, f"{key}: relative difference {difference:.1e}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_adjoint_step_is_the_transpose_of_the_forward_step(opencl_platform, monkeypatch, numerics):
    # The dot-product test of one time step: <(A - I) u, v> = <u, (A^T - I) v> for random fields u
    # and v, A being the forward step (PropagationEngine._forward_step) and A^T the adjoint updates
    # of the gradient loop (_adjoint_updates). The forward step updates the velocities, then the
    # stresses; its transpose updates the adjoint velocities, then the adjoint stresses. 0.1.10 runs
    # the adjoint stresses first. The fields are zero near the edges and the absorbing layer is
    # off, so that only the interior operators are compared; the stresses are scaled by an
    # impedance (and the adjoint stresses by its inverse) so that no block of the step dominates
    # (tools/math_baseline.py:metric_adjoint_dot).
    from tools.math_baseline import metric_adjoint_dot

    if REFERENCE:
        pytest.skip("needs the propagation engine of the fork")
    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    mismatch = metric_adjoint_dot()["relative_mismatch"]
    # Measured: 4.3e-3 in the legacy mode, 1.1e-7 (FP32 rounding) in the fixed mode.
    assert mismatch < 1e-5, f"relative mismatch {mismatch:.1e}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_adjoint_fields_are_damped_as_the_forward_ones(numerics):
    # The forward kernels damp every field by dt (dxpml + dzpml) inside the absorbing layer. The
    # adjoint kernels of 0.1.10 damp Ataux with dxpml alone, Atauz with dzpml alone, Avx with
    # 2 dxpml + dzpml and Avz with dxpml + 2 dzpml: the adjoint field is not absorbed as the forward
    # one, and part of it comes back from the boundary.
    import PyFWI
    with open(os.path.join(os.path.dirname(PyFWI.__file__), "elastic.cl")) as f:
        source = active_source(f.read(), fixed=numerics == "fixed")
    damping = []
    for kernel in ("Adj_update_tau", "Adj_update_v"):
        body = source[source.index(f"__kernel void {kernel}("):]
        body = body[:body.index("\n}")]
        damping += [line.strip() for line in body.splitlines() if "pml[center]" in line and "+=" in line]
    assert len(damping) == 5, damping  # Ataux, Atauz, Atauxz, Avx, Avz
    assert all("(dxpml[center] + dzpml[center])" in line for line in damping), damping


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_vp_gradient_agrees_with_finite_differences(opencl_platform, monkeypatch, numerics):
    # The mathematical-baseline check (tools/math_baseline.py): six smooth interior perturbations of
    # vp, the absorbing layer pinned. 0.1.10 injects the residual of step t, updates the adjoint
    # field and only then runs Grad, pairing the forward field after step t with the adjoint field
    # of about two steps earlier; the fixed mode runs Grad first (PropagationEngine._adjoint_step)
    # and damps the adjoint fields as the forward ones.
    from tools.math_baseline import metric_gradient_fd

    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    error = metric_gradient_fd("vp")["median_abs_error"]
    # Measured: 6.9 % in 0.1.10 and in the legacy mode; in the fixed mode 1.9 % with the adjoint
    # order alone, 0.6 % with the adjoint damping too.
    assert error < 0.015, f"median |ratio - 1| = {error:.3f}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_gradient_shear_strain_is_that_of_the_forward_kernels(numerics):
    # update_tauxz updates tauxz with mu (DxmVz + DzpVx), DzpVx being the z derivative of vx. Grad
    # correlates the adjoint of tauxz with DzpVx + DxmVz too, but its DzpVx is the z derivative of
    # vz (0.1.10): the gradient with respect to mu, hence vs and density, is wrong.
    import PyFWI
    with open(os.path.join(os.path.dirname(PyFWI.__file__), "elastic.cl")) as f:
        source = active_source(f.read(), fixed=numerics == "fixed")

    def strain(kernel):
        body = source[source.index(f"__kernel void {kernel}("):]
        definition = body[body.index("float DzpVx="):]
        return definition[:definition.index(";")]

    assert "vx[below]" in strain("update_tauxz")
    assert strain("Grad") == strain("update_tauxz"), strain("Grad")


@pytest.mark.parametrize("parameter, largest", [("vs", 0.01), ("rho", 0.03)])
@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_vs_and_density_gradients_agree_with_finite_differences(opencl_platform, monkeypatch, numerics, parameter,
                                                                 largest):
    # The check of test_vp_gradient_agrees_with_finite_differences for vs and density. Their
    # gradients take the mu gradient of Grad, whose shear strain was that of vz (and density the
    # other terms of Grad too).
    from tools.math_baseline import metric_gradient_fd

    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    error = metric_gradient_fd(parameter)["median_abs_error"]
    # Measured: vs 60 % and density 135 % in 0.1.10 and in the legacy mode; 0.27 % and 1.4 % in
    # the fixed mode (with its adjoint order and absorbing layer).
    assert error < largest, f"median |ratio - 1| = {error:.3f}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_receivers_are_recorded_inside_the_seismograms(opencl_platform, monkeypatch, numerics):
    # The same rounding the other way: with dt = 5 ms (dt_scale = 9) and t = 0.25 s there are 451
    # internal steps, so 51 recorded steps (0, 9, ..., 450), but nt_ext = 50 rows. 0.1.10 records
    # the last step in row 50, past the end of the five seismograms.
    with _filled_outside_buffers(monkeypatch, 1.0) as written_outside:
        case = tiny_case(t=0.25, inpa_extra={"dt": 0.005, "numerics": numerics})
        case.W.forward_modeling(case.model, False)
        assert written_outside(case.W) == []


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_crosswell_vz_seismogram_records_vz(opencl_platform, numerics):
    # elastic_crosswell.cl of 0.1.10 stores vx in the vz seismogram. components=2 records vx and vz;
    # with an explosive source in an elastic medium they differ.
    case = tiny_case(acq_type=0, components=2, inpa_extra={"numerics": numerics})
    d = case.W.forward_modeling(case.model, False)
    assert np.abs(d["vx"]).max() > 0
    assert not np.array_equal(d["vz"], d["vx"]), "the vz seismogram is a copy of vx"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_crosswell_kernels_assign_vz_to_vz_seismograms(numerics):
    # 0.1.10: lines 27 and 74 of elastic_crosswell.cl assign vx[...] to seismogram_vzi. The fork
    # keeps them in the legacy branch of #ifdef PYFWI_FIXED blocks (step 2.4).
    import PyFWI
    with open(os.path.join(os.path.dirname(PyFWI.__file__), "elastic_crosswell.cl")) as f:
        source = active_source(f.read(), fixed=numerics == "fixed")
    rhs = re.findall(r"seismogram_vzi\[[^\]]*\]\s*=\s*(\w+)\[", source)  # array read on the right-hand side
    assert len(rhs) == 2
    assert rhs == ["vz", "vz"], f"seismogram_vzi is assigned from {rhs}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_finite_difference_coefficients_are_consistent(numerics):
    # A staggered first derivative sum_m c_m (f(x + (m - 1/2) h) - f(x - (m - 1/2) h)) / h is exact
    # for a linear f only when c1 + 3 c2 + 5 c3 + 7 c4 = 1; otherwise every wave travels that much
    # too fast or too slow. Order 4 (9/8, -1/24) is right. Order 8 in 0.1.10 uses 1715/1434,
    # -114/1434, 14/1434 and -1/1434, the Taylor coefficients times 1434 rounded to integers: the
    # sum is 1436/1434, waves 0.14 % too fast (measured against the analytic solution: arrivals
    # 0.16 ms early after 300 m), and the rounding leaves a second-order error term: order 8
    # converges at order 2.2 to 2.7 instead of 7.5 (tools/math_baseline.py:metric_convergence).
    for order in (4, 8):
        defines = _defines(kernel_sources(order, numerics=numerics)[0])
        c1, c2, c3, c4 = (float(defines[f"c{m}"].rstrip("f")) for m in (1, 2, 3, 4))
        assert abs(c1 + 3 * c2 + 5 * c3 + 7 * c4 - 1) < 1e-7, f"order {order}: {c1 + 3 * c2 + 5 * c3 + 7 * c4}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_dt_define_is_not_truncated(numerics):
    # 0.1.10: '#define dt %f' keeps 6 decimals, 0.000571 instead of 0.000571428...
    dt = 0.004 / 7  # internal step for a user dt of 4 ms
    value = float(_defines(kernel_sources(8, dt=dt, numerics=numerics)[0])["dt"].rstrip("fF"))
    assert abs(value - dt) / dt < 1e-7, f"dt in the kernel is {value!r}, should be {dt!r}"


@pytest.mark.parametrize("numerics", NUMERIC_MODES)
def test_float_constants_have_the_f_suffix(numerics):
    # 0.1.10 writes the float constants without the f suffix: they are double, and the hot
    # kernels compute in FP64 (1/64 of the FP32 rate on GeForce).
    defines = _defines(kernel_sources(8, numerics=numerics)[0])
    floats = {name: defines[name] for name in ("dt", "dx", "dz", "c1", "c2", "c3", "c4")}
    assert all(v.endswith("f") for v in floats.values()), floats


# ---------------------------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------------------------

@pytest.mark.xfail(condition=REFERENCE and SINGLE_DEVICE_BUG, strict=True, raises=_pyopencl_error(),
                   reason="with PyOpenCL 2022, create_some_context reports a 'left over' device choice "
                          "on single-device platforms (fixed in the fork)")
def test_device_choice_works_with_a_single_device_platform(single_device_platform):
    # Fixed in step 1.2: the fork chooses the device itself (PyFWI._opencl_device), so it no
    # longer depends on the PyOpenCL version; test_device_selection.py simulates the case.
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


@pytest.mark.xfail(condition=REFERENCE, strict=True, raises=ImportError,
                   reason="0.1.10 imports segyio at module level but does not declare it (fixed in the fork)")
def test_model_dataset_imports_without_segyio(monkeypatch):
    # Pretend segyio is not installed, and force a fresh import of the modules that use it.
    monkeypatch.setitem(sys.modules, "segyio", None)
    for name in ("PyFWI.model_dataset", "PyFWI.seismic_io"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    # Fixed in step 1.1: segyio is imported only by the functions that read SEG-Y files.
    try:
        importlib.import_module("PyFWI.model_dataset")
    except ImportError as e:
        if "segyio" in str(e):
            raise  # this bug
        pytest.skip(f"PyFWI.model_dataset fails to import for another reason: {e}")


@pytest.mark.xfail(condition=REFERENCE, strict=True,
                   reason="0.1.10 lists pyopencl only in build-system.requires, not as a dependency (fixed in the fork)")
def test_pyopencl_is_a_declared_dependency():
    # Fixed in step 1.1. The distribution that provides the PyFWI package is "PyFWI" for the
    # published 0.1.10 and "pyfwi-cuda" for the fork.
    distribution = (importlib.metadata.packages_distributions().get("PyFWI") or ["PyFWI"])[0]
    requires = importlib.metadata.requires(distribution) or []
    names = {re.split(r"[ ;<>=!~\[]", r, maxsplit=1)[0].lower() for r in requires if "extra ==" not in r}
    assert "pyopencl" in names, sorted(names)


def test_wheel_contains_torch_and_data(built_distributions):
    # Fixed in step 1.1 (setup.py of 0.1.10 only packaged PyFWI and the .cl files). The wheel is
    # built from this repository, so the test runs where setuptools is installed (not in the
    # reference environment).
    with zipfile.ZipFile(built_distributions["wheel"]) as zf:
        names = zf.namelist()
    assert "PyFWI/wave_propagation.py" in names  # sanity check of the build itself
    assert "PyFWI/torch/wave_propagation.py" in names, "PyFWI/torch is missing from the wheel"
    assert "PyFWI/data/Dupuy2011_Moduli_data.mat" in names, "PyFWI/data is missing from the wheel"


@pytest.mark.xfail(condition=REFERENCE, strict=True, reason="0.1.10 opens the .cl files without closing them")
def test_kernel_files_are_closed():
    # Fixed in step 2.4: kernel_caller read the three kernel files without closing them, and Python
    # warned (ResourceWarning) when the file objects were collected.
    import gc
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        kernel_sources(4)
        gc.collect()
    leaks = [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)]
    assert not leaks, leaks
