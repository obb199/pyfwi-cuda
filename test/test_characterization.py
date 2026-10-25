"""Characterisation: the fork must reproduce the golden masters of PyFWI 0.1.10 (roadmap step 0.5).

Every case of the golden-master matrix is run again with the PyFWI installed in the
current environment and compared, layer by layer, with what the published 0.1.10
produced in the reference environment (``test/golden/``): the propagation cases
(fast), and the inversion and example cases (marked ``slow``: several minutes).

How strict the comparison is
----------------------------
* **Bit for bit** whenever the installed PyFWI computes the same PML damping profile
  as 0.1.10 (``helpers.pml_matches_0_1_10``). That is the case in the reference
  environment, and it is what step 1.1 restores under NumPy 2: with the PML profile
  pinned, every seismogram and gradient was measured identical, bit for bit.
* With **interim tolerances** otherwise (NumPy 2 before step 1.1): the PML profile
  differs in the last bit, which moves seismograms by up to 5.7e-7 and gradients by
  up to 2.8e-5 in relative L2 norm (measured). ``TOLERANCE`` leaves a margin.
* The **misfit** is a single float64 sum whose summation order depends on the NumPy
  version: relative difference up to 5e-8 even with the PML pinned; compared with
  ``MISFIT_RTOL``.

A failure prints the worst arrays, so a regression can be traced to the first layer
that changes (raw seismogram -> data -> raw gradient -> gradient).

Mutation test (measured): scaling the source wavelet by 1 + 1e-7 is caught in the
bit-for-bit mode (15 arrays flagged) but **not** in the tolerance mode. That blind spot
is why the PML fix is the first change of step 1.1.
"""
import importlib.util

import numpy as np
import pytest

from golden_cases import example_cases, inversion_cases, propagation_cases
from golden_runner import run_case, run_propagation_case
from golden_store import load_case, load_manifest
from helpers import has_memoizejac, pml_matches_0_1_10

#: True where the comparison must be bit for bit (see the module docstring).
BITWISE = pml_matches_0_1_10()

#: Interim relative-L2 tolerances per output layer, used only when BITWISE is False.
TOLERANCE = {"raw_seismogram": 1e-6, "observed": 1e-6, "predicted": 1e-6, "raw_gradient": 1e-4, "gradient": 1e-4}

#: Relative tolerance of the misfit (a float64 sum; see the module docstring).
MISFIT_RTOL = 1e-6 if BITWISE else 1e-5

CASES = {case.name: case for case in propagation_cases()}
MANIFEST = load_manifest()


def relative_l2(expected, actual):
    """||actual - expected|| / ||expected||, in float64 (0 when both are zero, inf if only ``expected`` is)."""
    e, a = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    norm = np.linalg.norm(e)
    diff = np.linalg.norm(a - e)
    return diff / norm if norm else (0.0 if diff == 0 else np.inf)


@pytest.mark.skipif(MANIFEST is None, reason="no golden masters")
@pytest.mark.parametrize("name", sorted(CASES))
def test_propagation_matches_golden(opencl_platform, name):
    golden, actual = load_case(name), run_propagation_case(CASES[name])
    assert actual.keys() == golden.keys()
    failures = []
    for key, expected in golden.items():
        layer = key.split("/")[0]
        if layer == "misfit":
            ok = abs(float(actual[key]) - float(expected)) <= MISFIT_RTOL * abs(float(expected))
            detail = f"relative {abs(float(actual[key]) / float(expected) - 1):.1e}"
        elif BITWISE:
            ok = actual[key].dtype == expected.dtype and np.array_equal(actual[key], expected)
            detail = f"relative L2 {relative_l2(expected, actual[key]):.1e} (bit-for-bit required)"
        else:
            error = relative_l2(expected, actual[key])
            ok, detail = error <= TOLERANCE[layer], f"relative L2 {error:.1e} > {TOLERANCE[layer]:.0e}"
        if not ok:
            failures.append(f"{key}: {detail}")
    assert not failures, "\n".join(failures)


#: Interim relative-L2 tolerance for inversion and example outputs, used only when BITWISE is
#: False: they contain gradients (up to 2.8e-5 measured) and models updated from them.
OUTPUT_TOLERANCE = 1e-4

#: Why some cases cannot run in some environments (marks applied below).
NEEDS_FWI = pytest.mark.xfail(condition=not has_memoizejac(), strict=True, raises=ImportError,
                              reason="PyFWI.fwi needs MemoizeJac, missing from current SciPy (step 1.1)")
NEEDS_TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
EXAMPLE_MARKS = {"fwi": [NEEDS_FWI], "tlfwi": [NEEDS_FWI], "torch_gradient": [NEEDS_TORCH]}


def _compare_outputs(name, actual):
    """Compare flattened inversion/example outputs with the golden master; return the failures."""
    golden = load_case(name)
    failures = [f"missing or unexpected keys: {sorted(set(golden) ^ set(actual))}"] if golden.keys() != actual.keys() else []
    for key in sorted(set(golden) & set(actual)):
        expected = golden[key]
        if BITWISE:
            if not (actual[key].dtype == expected.dtype and np.array_equal(actual[key], expected)):
                failures.append(f"{key}: relative L2 {relative_l2(expected, actual[key]):.1e} (bit-for-bit required)")
        elif (error := relative_l2(expected, actual[key])) > OUTPUT_TOLERANCE:
            failures.append(f"{key}: relative L2 {error:.1e} > {OUTPUT_TOLERANCE:.0e}")
    return failures


@pytest.mark.slow
@pytest.mark.skipif(MANIFEST is None, reason="no golden masters")
@pytest.mark.parametrize("case", [pytest.param(c, marks=[NEEDS_FWI], id=c.name) for c in inversion_cases()])
def test_inversion_matches_golden(opencl_platform, case):
    failures = _compare_outputs(case.name, run_case(case))
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(MANIFEST is None, reason="no golden masters")
@pytest.mark.parametrize("case", [pytest.param(c, marks=EXAMPLE_MARKS.get(c.script, []), id=c.name)
                                  for c in example_cases()])
def test_example_matches_golden(opencl_platform, case):
    failures = _compare_outputs(case.name, run_case(case))
    assert not failures, "\n".join(failures)
