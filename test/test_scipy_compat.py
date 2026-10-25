"""Tests of the SciPy compatibility layer (``PyFWI/_scipy_compat.py``), no device needed.

``MemoizeJac`` must behave like the private helper of SciPy 1.7 that PyFWI 0.1.10 used
(checked directly against it in the reference environment), and the
``fmin_l_bfgs_b`` wrapper must give exactly the result of SciPy's own function.

The module is loaded from this repository by file path, so the tests also run in the
reference environment (where the installed PyFWI is the published 0.1.10, which has no
such module) and compare with SciPy 1.7's own ``MemoizeJac`` there.
"""
import importlib.util
import os

import numpy as np
import pytest
from scipy import optimize

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "PyFWI", "_scipy_compat.py")
_SPEC = importlib.util.spec_from_file_location("repository_scipy_compat", _PATH)
_COMPAT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPAT)
MemoizeJac, fmin_l_bfgs_b = _COMPAT.MemoizeJac, _COMPAT.fmin_l_bfgs_b


class CountingQuadratic:
    """f(x) = sum((x - shift)^2), returning (value, gradient) and counting its evaluations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, shift=0.0):
        self.calls += 1
        return float(np.sum((x - shift) ** 2)), 2 * (x - shift)


def test_value_and_gradient_come_from_one_evaluation():
    f = CountingQuadratic()
    memo = MemoizeJac(f)
    x = np.array([1.0, 2.0])
    assert memo(x) == 5.0
    np.testing.assert_array_equal(memo.derivative(x), [2.0, 4.0])
    assert f.calls == 1  # the gradient was served from the cache


def test_a_new_point_is_evaluated():
    f = CountingQuadratic()
    memo = MemoizeJac(f)
    memo(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(memo.derivative(np.array([1.0, 3.0])), [2.0, 6.0])
    assert f.calls == 2


def test_arguments_are_not_part_of_the_cache_key():
    # As in SciPy 1.7: only x decides whether the cached result is reused. PyFWI passes the
    # same extra arguments during a whole optimisation, so this never returns a stale value.
    f = CountingQuadratic()
    memo = MemoizeJac(f)
    x = np.array([1.0, 3.0])
    assert memo(x, 0.0) == 10.0
    assert memo(x, 1.0) == 10.0 and f.calls == 1


def test_the_cache_keeps_a_copy_of_x():
    f = CountingQuadratic()
    memo = MemoizeJac(f)
    x = np.array([1.0, 2.0])
    memo(x)
    x[0] = 5.0  # an optimiser may modify its array in place
    memo(x)
    assert f.calls == 2


def test_same_behaviour_as_scipy_1_7():
    try:
        from scipy.optimize.optimize import MemoizeJac as ScipyMemoizeJac
    except ImportError:
        pytest.skip("this SciPy has no MemoizeJac (only SciPy < 1.8 does)")
    points = [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.5, 2.0]), np.array([0.5, 2.0])]
    ours, theirs = CountingQuadratic(), CountingQuadratic()
    a, b = MemoizeJac(ours), ScipyMemoizeJac(theirs)
    for i, x in enumerate(points):
        if i % 2:
            np.testing.assert_array_equal(a.derivative(x), b.derivative(x))
        else:
            assert a(x) == b(x)
    assert ours.calls == theirs.calls


def test_lbfgsb_wrapper_matches_scipy_and_ignores_print_options():
    f = MemoizeJac(CountingQuadratic())
    x0 = np.array([3.0, -1.0, 2.0])
    ours = fmin_l_bfgs_b(f, x0, f.derivative, args=(0.5,), m=10, factr=1e7, pgtol=1e-8, iprint=-1, bounds=None,
                         maxfun=15000, maxiter=5, disp=None, callback=None, maxls=10)
    g = MemoizeJac(CountingQuadratic())
    reference = optimize.fmin_l_bfgs_b(g, x0, g.derivative, args=(0.5,), m=10, factr=1e7, pgtol=1e-8, bounds=None,
                                       maxfun=15000, maxiter=5, callback=None, maxls=10)
    np.testing.assert_array_equal(ours[0], reference[0])
    assert ours[1] == reference[1]
    np.testing.assert_allclose(ours[0], 0.5)


def test_lbfgsb_wrapper_still_rejects_unknown_arguments():
    f = MemoizeJac(CountingQuadratic())
    with pytest.raises(TypeError):
        fmin_l_bfgs_b(f, np.zeros(2), f.derivative, not_an_argument=1)
