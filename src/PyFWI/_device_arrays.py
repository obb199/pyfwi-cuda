"""Host work of an inversion moved to the GPU, on request (roadmap step 4.1, an opt-in feature of Phase 4).

Since Phase 3 the propagation is fast enough that the work the host does between two propagations
shows: in an inversion of the size of the small benchmark scenario, the multi-scale filters of the cost
function (FFTs of every trace, three times per evaluation) took about a third of each evaluation, with
the GPU idle. ``PYFWI_CUDA_COST=1`` runs that work on the GPU with CuPy:

* the zero-phase low-pass filters of the multi-scale strategy (``fwi_tools._zero_phase_filter``), with
  ``cupy.fft`` in float32 (the host computes them in float64, which GeForce cards run 64 times slower);
* the L2 cost function and its residual (``fwi_tools.CostFunction.l2``); the other cost functions, and
  the source weighting, stay on the host;
* the Gaussian smoothing of the gradient (``WavePropagator.gradient``, ``g_smooth``), with
  ``cupyx.scipy.ndimage.gaussian_filter``.

The functions keep their interface: they take and return NumPy arrays, the copies to and from the
GPU happen inside. The results differ from those of the host by rounding only (the FFT and the sums
are computed in another order, the filters in float32): at most 1e-6 relative, checked by
``test/test_device_arrays.py``.
Without the variable (the default), or with ``PYFWI_CUDA_COST=0``, nothing changes.

Used by
-------
``PyFWI.fwi_tools`` (``cost_seismic``, ``_zero_phase_filter``, ``CostFunction``) and
``PyFWI.wave_propagation`` (``gaussian_filter``). Tested by ``test/test_device_arrays.py``.
"""
import importlib
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter as host_gaussian_filter

#: The environment variable that moves the cost function, its filters and the smoothing to the GPU.
ENVIRONMENT_VARIABLE = "PYFWI_CUDA_COST"


def cost_on_gpu(environ=None):
    """True when ``PYFWI_CUDA_COST`` asks for the GPU (``1``); False when unset, empty or ``0``.

    Raises
    ------
    ValueError
        Another value.
    RuntimeError
        ``1`` without CuPy (installed by the ``cuda13`` and ``cuda12`` extras).
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get(ENVIRONMENT_VARIABLE, "")).strip()
    if value in ("", "0"):
        return False
    if value != "1":
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be 0 or 1, not {value!r}")
    try:
        importlib.import_module("cupy")
    except ImportError as error:
        raise RuntimeError(f"{ENVIRONMENT_VARIABLE}=1 needs CuPy (pip install 'pyfwi-cuda[cuda13]'): {error}") from None
    return True


def array_module(x):
    """``cupy`` for a CuPy array, else ``numpy``: the module whose functions apply to ``x``."""
    return sys.modules["cupy"] if type(x).__module__.split(".")[0] == "cupy" else np


def to_device(x):
    """A CuPy copy of the NumPy array ``x`` (``x`` itself when it is already on the GPU)."""
    import cupy
    return cupy.asarray(x)


def to_host(x):
    """A NumPy copy of the CuPy array ``x`` (``x`` itself when it is a NumPy array)."""
    return x.get() if array_module(x) is not np else x


def gaussian_filter(x, sigma):
    """``scipy.ndimage.gaussian_filter(x, sigma)``, on the GPU with ``PYFWI_CUDA_COST=1``.

    Used by ``WavePropagator.gradient`` to smooth the gradient (``g_smooth``); the result is a NumPy
    array either way, of the dtype of ``x``.
    """
    if not cost_on_gpu():
        return host_gaussian_filter(x, sigma)
    from cupyx.scipy.ndimage import gaussian_filter as device_gaussian_filter
    return to_host(device_gaussian_filter(to_device(x), sigma))
