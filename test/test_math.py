"""Mathematical correctness must not get worse than in PyFWI 0.1.10 (roadmap step 0.6).

``tools/math_baseline.py`` measures four metrics (gradient against finite differences,
agreement with the analytic solution, reflection of the absolute boundary, residual
energy) and stored the values of the published 0.1.10 in
``test/golden/math_baseline.json``. Every later version must do at least as well; the
fixes of Phases 2-4 are expected to improve some of them (the baseline is then raised
by regenerating it with the new version, never lowered).

The tests also state the absolute quality reached by 0.1.10, so a reader knows what
the numbers mean: the propagation agrees with the analytic solution to a correlation
above 0.998 with no time shift, the boundary reflects about 1 %, the wavefield decays
to 1e-8 of its peak, and the gradient is only approximate (its directional derivatives
differ from finite differences by about 7 % in the median for vp, 60 % for vs and 135 % for
density).

Both numeric modes of the fork are measured (``PYFWI_NUMERICS``). ``legacy`` runs the kernels
of 0.1.10 and gives the baseline values; ``fixed`` (step 2.4) must not be worse, and the tests
below also state what it reaches: the gradient within 1.5 % of finite differences for vp, vs and
density, an adjoint step that is the transpose of the forward one, and the analytic agreement of
order 8 restored by its corrected coefficients. The dot-product test needs the engine of the
fork: the baseline of 0.1.10 has none, and the legacy mode measures the kernels of 0.1.10.
"""
import json
import os

import pytest

from helpers import running_reference

#: Absolute path of the repository root and of the stored baseline.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_FILE = os.path.join(REPO, "test", "golden", "math_baseline.json")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def baseline():
    """Metrics of the published 0.1.10, as stored by ``tools/math_baseline.py --write``."""
    with open(BASELINE_FILE) as f:
        stored = json.load(f)
    assert stored["pyfwi"] == "0.1.10"
    return stored["metrics"]


#: The numeric modes measured; the published 0.1.10 (reference environment) has only legacy.
MODES = ["legacy", pytest.param("fixed", marks=pytest.mark.skipif(running_reference(),
                                                                  reason="0.1.10 has no fixed mode"))]


@pytest.fixture(scope="module", params=MODES)
def numerics(request):
    """The numeric mode of the measured metrics (``PYFWI_NUMERICS``)."""
    return request.param


@pytest.fixture(scope="module")
def measured(numerics):
    """Metrics of the installed PyFWI in one numeric mode (measured once per mode; about 15 s on a GPU)."""
    from conftest import _opencl_platforms, _pyopencl_ctx
    from tools.math_baseline import measure

    platforms = _opencl_platforms()
    if not platforms:
        pytest.skip("no OpenCL platform")
    chosen = next((p for p in platforms if "NVIDIA" in p[1]), platforms[0])
    settings = {"PYOPENCL_CTX": _pyopencl_ctx(chosen[0], chosen[2]), "PYFWI_NUMERICS": numerics}
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        return measure()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_gradient_agrees_with_finite_differences_at_least_as_well(baseline, measured):
    assert measured["gradient_fd"]["median_abs_error"] <= baseline["gradient_fd"]["median_abs_error"] + 0.02


def test_vs_gradient_agrees_with_finite_differences_at_least_as_well(baseline, measured):
    assert measured["gradient_fd_vs"]["median_abs_error"] <= baseline["gradient_fd_vs"]["median_abs_error"] + 0.02


def test_density_gradient_agrees_with_finite_differences_at_least_as_well(baseline, measured):
    assert measured["gradient_fd_rho"]["median_abs_error"] <= baseline["gradient_fd_rho"]["median_abs_error"] + 0.02


def test_fixed_mode_gradient_agrees_with_finite_differences(numerics, measured):
    # Step 2.4: adjoint order, adjoint absorbing layer and shear strain of Grad corrected.
    # Measured: vp 0.6 %, vs 0.27 %, density 1.4 % (0.1.10: 6.9 %, 60 %, 135 %).
    if numerics != "fixed":
        pytest.skip("the legacy mode keeps the gradient of 0.1.10")
    for metric in ("gradient_fd", "gradient_fd_vs", "gradient_fd_rho"):
        assert measured[metric]["median_abs_error"] < 0.03, (metric, measured[metric])


def test_adjoint_step_is_the_transpose_of_the_forward_step(numerics, measured):
    # Measured: 4.3e-3 in the legacy mode (the kernels of 0.1.10 in their order), 1.1e-7 in the
    # fixed mode, the rounding of FP32; the criterion of the roadmap for the new kernels is 1e-5.
    mismatch = measured["adjoint_dot"]["relative_mismatch"]
    if mismatch is None:
        pytest.skip("the published 0.1.10 exposes no single time step")
    assert mismatch < (1e-5 if numerics == "fixed" else 1e-2), mismatch


def test_propagation_agrees_with_the_analytic_solution(baseline, measured):
    assert measured["analytic"]["min_correlation"] >= baseline["analytic"]["min_correlation"] - 1e-4
    assert measured["analytic"]["max_peak_shift"] <= baseline["analytic"]["max_peak_shift"]
    assert measured["analytic"]["min_correlation"] > 0.998  # absolute level of 0.1.10


def test_absorbing_boundary_reflects_no_more(baseline, measured):
    assert measured["pml_reflection"]["relative_error"] <= 1.05 * baseline["pml_reflection"]["relative_error"]
    assert measured["pml_reflection"]["relative_error"] < 0.02  # about 1 % in 0.1.10


def test_wavefield_decays(baseline, measured):
    assert measured["late_energy"]["end_over_peak"] <= max(10 * baseline["late_energy"]["end_over_peak"], 1e-7)
