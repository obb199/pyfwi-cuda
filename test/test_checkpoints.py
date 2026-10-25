"""Checkpoints of the forward field and the public attribute ``W.W`` (roadmap step 1.5).

The gradient rebuilds the forward field backwards in time and restarts from stored copies of it,
the *checkpoints* (``chpr`` per cent of the time steps). PyFWI 0.1.10 stored them in ``W.W`` as
``{component: array (tnz, tnx, ns, nchp)}``, where the field of one shot at one checkpoint is
scattered in memory, so each copy went through a temporary array. Since step 1.5 the fork stores
them as ``(ns, nchp, tnz, tnx)`` and copies each field straight from and to the device without
blocking; ``W.W`` is a property that shows the same arrays in the old layout, as views.

The API tests run in both environments: what holds for the published 0.1.10 must hold for the
fork (shape, dtype, axis order, values, writing into ``W.W`` and assigning it).
"""
import tracemalloc

import numpy as np
import pytest

from helpers import forward_and_gradient, running_reference, tiny_case
from oob_guard import zero_guard_bands

#: The five components, in the order of the dictionary W.W.
COMPONENTS = ["vx", "vz", "taux", "tauz", "tauxz"]


def _device_field(W, name):
    """The current content of the device buffer of one field component (e.g. ``W.vx_b``)."""
    import pyopencl as cl
    host = np.empty((W.tnz, W.tnx), np.float32)
    cl.enqueue_copy(W.queue, host, getattr(W, f"{name}_b"))
    return host


def test_W_has_the_layout_of_0_1_10(opencl_platform):
    # Inside zero guard bands: on POCL the out-of-bounds reads of 0.1.10 give NaN otherwise.
    with zero_guard_bands():
        case = tiny_case(ns=2)
        W = case.W
        W.forward_modeling(case.model, False)
    tnz, tnx, ns, nchp = int(W.tnz), int(W.tnx), int(W.ns), int(W.nchp)
    assert list(W.W) == COMPONENTS
    for name, array in W.W.items():
        assert array.shape == (tnz, tnx, ns, nchp) and array.dtype == np.float32, name
    # The last checkpoint is the last time step (chp[-1] = nt - 1): for the last shot it must be
    # what the device buffers hold at the end, which fixes the meaning of every axis.
    assert W.chp[-1] == W.nt - 1
    for name in COMPONENTS:
        np.testing.assert_array_equal(W.W[name][:, :, ns - 1, nchp - 1], _device_field(W, name), err_msg=name)
    # The shots differ (sources at different places), and the field grows from zero in time.
    assert not np.array_equal(W.W["vx"][:, :, 0, -1], W.W["vx"][:, :, 1, -1])
    assert np.abs(W.W["vx"][:, :, 0, 1]).max() < np.abs(W.W["vx"][:, :, 0, -1]).max()


def _gradient_after_forward(case, change=None):
    """Forward modelling of the starting model, an optional change of the checkpoints, then the gradient.

    The residual is that of ``forward_and_gradient`` (observed data of the true model).
    """
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    d_obs = process.prepare_residual(case.W.forward_modeling(case.model, False), 1)
    d_est = process.prepare_residual(case.W.forward_modeling(case.m0, False), 1)
    _, adj = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))
    if change is not None:
        change(case.W)
    return case.W.gradient(adj, False)


def _double_the_middle_checkpoint(W):
    """Write into W.W, as user code of 0.1.10 could: double the field vx of shot 0 at the middle checkpoint."""
    W.W["vx"][:, :, 0, int(W.nchp) // 2] *= 2


def _assign_a_doubled_copy(W):
    """Assign W.W, as PyFWI.optimization does: a copy with the same change."""
    copy = {name: array.copy() for name, array in W.W.items()}
    copy["vx"][:, :, 0, int(W.nchp) // 2] *= 2
    W.W = copy


def test_writing_into_W_or_assigning_it_changes_the_gradient(opencl_platform):
    with zero_guard_bands():
        plain = _gradient_after_forward(tiny_case())
        written = _gradient_after_forward(tiny_case(), _double_the_middle_checkpoint)
        assigned = _gradient_after_forward(tiny_case(), _assign_a_doubled_copy)
        unchanged = _gradient_after_forward(tiny_case(), lambda W: setattr(W, "W", W.W))
    assert not np.array_equal(written["vp"], plain["vp"])  # the gradient uses what was written
    for key in plain:
        np.testing.assert_array_equal(assigned[key], written[key], err_msg=key)
        np.testing.assert_array_equal(unchanged[key], plain[key], err_msg=key)


@pytest.mark.skipif(running_reference(), reason="the storage layout is the fork's (step 1.5)")
def test_checkpoints_are_stored_contiguously_and_W_shows_views(opencl_platform):
    case = tiny_case(ns=2)
    W = case.W
    W.forward_modeling(case.model, False)
    for name in COMPONENTS:
        store = W._checkpoints[name]
        assert store.shape == (W.ns, W.nchp, W.tnz, W.tnx) and store[1, 3].flags.c_contiguous
        assert np.shares_memory(W.W[name], store)  # a view, not a copy


@pytest.mark.skipif(running_reference(), reason="measures the fork's copies (step 1.5)")
def test_forward_and_gradient_make_no_copy_of_the_checkpoints(opencl_platform):
    # MEM: with every time step stored (chpr = 100) the checkpoints dominate the memory. Forward
    # and gradient copy fields straight from and to the storage, so the extra memory they need is a
    # small fraction of it (a temporary copy of the checkpoints would need as much again).
    case = tiny_case(ns=2, chpr=100)
    stored = sum(array.nbytes for array in case.W._checkpoints.values())
    forward_and_gradient(case)  # warm-up: buffers and caches of the first run
    tracemalloc.start()
    try:
        forward_and_gradient(case)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert stored > 40e6 and peak < 0.25 * stored, (stored, peak)
