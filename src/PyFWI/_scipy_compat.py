"""Compatibility layer between PyFWI and the SciPy optimisers it uses.

PyFWI 0.1.10 was written for SciPy 1.7 and relied on two things that later SciPy
versions removed:

* ``scipy.optimize.optimize.MemoizeJac``, a private helper (the module
  ``scipy.optimize.optimize`` became private in SciPy 1.8 and no longer exports it);
* the ``iprint`` and ``disp`` arguments of ``scipy.optimize.fmin_l_bfgs_b``, which only
  control how much progress the optimiser prints (removed in recent SciPy versions).

This module provides both in a way that works with old and new SciPy, so that
``PyFWI.fwi``, ``PyFWI.tl_fwi`` and ``PyFWI.optimization`` import and run everywhere.

Used by
-------
``PyFWI.fwi``, ``PyFWI.tl_fwi`` and ``PyFWI.optimization``. Tested by
``test/test_scipy_compat.py``.
"""
import inspect

import numpy as np
from scipy import optimize


class MemoizeJac:
    """Wrap a function that returns ``(value, gradient)`` so that value and gradient are computed once per point.

    SciPy's optimisers ask for the objective value (``fun(x)``) and for its gradient
    (``jac(x)``) through two separate calls. In PyFWI both come from the same expensive
    computation (a forward modelling plus an adjoint-state gradient), so this wrapper
    runs it once for a given ``x`` and serves the second request from the cache. It
    behaves like ``scipy.optimize.optimize.MemoizeJac`` of SciPy 1.7.

    Parameters
    ----------
    fun : callable
        ``fun(x, *args) -> (value, gradient)``.

    Attributes
    ----------
    x : numpy.ndarray or None
        Copy of the last point at which ``fun`` was evaluated.
    jac : numpy.ndarray or None
        Gradient at ``x``.

    Examples
    --------
    >>> f = MemoizeJac(lambda x: (float(x @ x), 2 * x))
    >>> scipy.optimize.fmin_l_bfgs_b(f, x0, f.derivative)   # doctest: +SKIP
    """

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        """Evaluate ``fun`` unless the cache already holds the result for this ``x``."""
        if self._value is None or self.jac is None or not np.all(x == self.x):
            self.x = np.asarray(x).copy()
            value, gradient = self.fun(x, *args)
            self.jac = gradient
            self._value = value

    def __call__(self, x, *args):
        """Return the objective value at ``x`` (computing value and gradient if needed)."""
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        """Return the gradient at ``x`` (computing value and gradient if needed)."""
        self._compute_if_needed(x, *args)
        return self.jac


#: Keyword arguments that the installed ``scipy.optimize.fmin_l_bfgs_b`` accepts.
_LBFGSB_PARAMETERS = set(inspect.signature(optimize.fmin_l_bfgs_b).parameters)


def fmin_l_bfgs_b(func, x0, fprime=None, **kwargs):
    """Call ``scipy.optimize.fmin_l_bfgs_b``, dropping the arguments the installed SciPy no longer accepts.

    Only ``iprint`` and ``disp`` are ever dropped: they control how much the optimiser
    prints, never what it computes. Any other unknown argument is passed on, so a real
    mistake still raises ``TypeError``.

    Parameters
    ----------
    func, x0, fprime
        As for ``scipy.optimize.fmin_l_bfgs_b``.
    **kwargs
        Other arguments of ``scipy.optimize.fmin_l_bfgs_b``.

    Returns
    -------
    tuple
        ``(x, f, d)`` as returned by ``scipy.optimize.fmin_l_bfgs_b``.
    """
    for name in ("iprint", "disp"):
        if name not in _LBFGSB_PARAMETERS:
            kwargs.pop(name, None)
    return optimize.fmin_l_bfgs_b(func, x0, fprime, **kwargs)
