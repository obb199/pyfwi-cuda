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
differ from finite differences by about 30 % in the median, sometimes with the wrong
sign).
"""
import json
import os

import pytest

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


@pytest.fixture(scope="module")
def measured():
    """Metrics of the installed PyFWI (measured once per test session; about 15 s on a GPU)."""
    from conftest import _opencl_platforms, _pyopencl_ctx
    from tools.math_baseline import measure

    platforms = _opencl_platforms()
    if not platforms:
        pytest.skip("no OpenCL platform")
    chosen = next((p for p in platforms if "NVIDIA" in p[1]), platforms[0])
    previous = os.environ.get("PYOPENCL_CTX")
    os.environ["PYOPENCL_CTX"] = _pyopencl_ctx(chosen[0], chosen[2])
    try:
        return measure()
    finally:
        if previous is None:
            os.environ.pop("PYOPENCL_CTX", None)
        else:
            os.environ["PYOPENCL_CTX"] = previous


def test_gradient_agrees_with_finite_differences_at_least_as_well(baseline, measured):
    assert measured["gradient_fd"]["median_abs_error"] <= baseline["gradient_fd"]["median_abs_error"] + 0.02


def test_propagation_agrees_with_the_analytic_solution(baseline, measured):
    assert measured["analytic"]["min_correlation"] >= baseline["analytic"]["min_correlation"] - 1e-4
    assert measured["analytic"]["max_peak_shift"] <= baseline["analytic"]["max_peak_shift"]
    assert measured["analytic"]["min_correlation"] > 0.998  # absolute level of 0.1.10


def test_absorbing_boundary_reflects_no_more(baseline, measured):
    assert measured["pml_reflection"]["relative_error"] <= 1.05 * baseline["pml_reflection"]["relative_error"]
    assert measured["pml_reflection"]["relative_error"] < 0.02  # about 1 % in 0.1.10


def test_wavefield_decays(baseline, measured):
    assert measured["late_energy"]["end_over_peak"] <= max(10 * baseline["late_energy"]["end_over_peak"], 1e-7)
