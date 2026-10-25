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
* **Optimiser outputs in float64** (inversion and example models) may differ in the last bit
  when SciPy is not the version of the golden masters (``FLOAT64_ACROSS_SCIPY``).

A failure prints the worst arrays, so a regression can be traced to the first layer
that changes (raw seismogram -> data -> raw gradient -> gradient).

Guard bands
-----------
The golden masters were made inside zero guard bands (``test/oob_guard.py``), which give the
out-of-bounds accesses of 0.1.10 a defined result. The published 0.1.10 (reference environment)
still needs them; the fork pads its own buffers since step 1.7 and runs here **without** them
(``GUARD``), so these tests also check that the padding reproduces the guard bands bit for bit.

Mutation test (measured): scaling the source wavelet by 1 + 1e-7 is caught in the
bit-for-bit mode (15 arrays flagged) but **not** in the tolerance mode. That blind spot
is why the PML fix is the first change of step 1.1.
"""
import importlib.util

import numpy as np
import pytest
import scipy

from golden_cases import example_cases, inversion_cases, propagation_cases
from golden_runner import run_case, run_propagation_case
from golden_store import load_case, load_manifest
from helpers import pml_matches_0_1_10, running_reference

#: True where the comparison must be bit for bit (see the module docstring).
BITWISE = pml_matches_0_1_10()

#: Run inside zero guard bands: only for the published 0.1.10 (see the module docstring).
GUARD = running_reference()

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
    golden, actual = load_case(name), run_propagation_case(CASES[name], guard=GUARD)
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
NEEDS_TORCH = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
EXAMPLE_MARKS = {"torch_gradient": [NEEDS_TORCH]}


#: True when the installed SciPy is the one the golden masters were made with (1.7.3).
SAME_SCIPY = MANIFEST is not None and scipy.__version__ == MANIFEST["environment"]["scipy"]

#: Relative L2 allowed on float64 outputs of the optimisers when SciPy differs from the golden
#: masters' one. L-BFGS-B of SciPy >= 1.15 is a C translation of the Fortran code of SciPy 1.7: the
#: iterates agree, but some float64 operations round differently in the last bit (measured 4e-17 on
#: the FWI example, 1.5e-15 on a time-lapse difference). float32 outputs must still match bit for bit.
FLOAT64_ACROSS_SCIPY = 1e-12


def _compare_outputs(name, actual):
    """Compare flattened inversion/example outputs with the golden master; return the failures."""
    golden = load_case(name)
    failures = [f"missing or unexpected keys: {sorted(set(golden) ^ set(actual))}"] if golden.keys() != actual.keys() else []
    for key in sorted(set(golden) & set(actual)):
        expected, error = golden[key], relative_l2(golden[key], actual[key])
        if BITWISE:
            identical = actual[key].dtype == expected.dtype and np.array_equal(actual[key], expected)
            ulp_level = (not SAME_SCIPY and actual[key].dtype == expected.dtype == np.float64
                         and error <= FLOAT64_ACROSS_SCIPY)
            if not (identical or ulp_level):
                failures.append(f"{key}: relative L2 {error:.1e} (bit-for-bit required"
                                f"{'' if SAME_SCIPY else ', or 1e-12 for float64 across SciPy versions'})")
        elif error > OUTPUT_TOLERANCE:
            failures.append(f"{key}: relative L2 {error:.1e} > {OUTPUT_TOLERANCE:.0e}")
    return failures


@pytest.mark.slow
@pytest.mark.skipif(MANIFEST is None, reason="no golden masters")
@pytest.mark.parametrize("case", inversion_cases(), ids=lambda c: c.name)
def test_inversion_matches_golden(opencl_platform, case):
    failures = _compare_outputs(case.name, run_case(case, guard=GUARD))
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.skipif(MANIFEST is None, reason="no golden masters")
@pytest.mark.parametrize("case", [pytest.param(c, marks=EXAMPLE_MARKS.get(c.script, []), id=c.name)
                                  for c in example_cases()])
def test_example_matches_golden(opencl_platform, case):
    failures = _compare_outputs(case.name, run_case(case, guard=GUARD))
    assert not failures, "\n".join(failures)
