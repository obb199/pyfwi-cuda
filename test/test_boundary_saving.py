"""Boundary saving (roadmap step 4.2): the forward field rebuilt from a band stored at every step.

With ``PYFWI_BOUNDARY_SAVING=1`` the forward modelling stores, at every step, the fields of the cells of
the undamped rectangle within ``sdo`` of its edge, and the last field; the gradient rebuilds the forward
field backwards from it. The reference is the exact rebuild: a checkpoint at every step (``chpr=100``).
The tests of the roadmap: the gradient against that reference (at most 1e-4), the memory proportional
to the perimeter on three grid sizes (MEM), a long run without drift (EST), and a complete inversion
(FWI). The bookkeeping of the engine is tested on the fake backend (``test_engine.py``), the variable
by ``test_backend_selection.py``.
"""
import os
import sys

import numpy as np
import pytest

from helpers import forward_and_gradient, running_reference, tiny_case

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no boundary saving")

#: Largest relative difference of the gradient against the exact rebuild (the roadmap's tolerance).
TOLERANCE = 1e-4


def relative(expected, actual):
    """``||actual - expected|| / ||expected||`` in float64."""
    e, a = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    return float(np.linalg.norm(a - e) / np.linalg.norm(e))


def test_it_needs_the_native_kernels(opencl_platform, monkeypatch):
    monkeypatch.setenv("PYFWI_BACKEND", "opencl")
    monkeypatch.setenv("PYFWI_BOUNDARY_SAVING", "1")
    with pytest.raises(ValueError, match="needs the native CUDA kernels"):
        tiny_case()


def _gradients(monkeypatch, kw):
    """``(exact, saved, propagator)``: forward and gradient with a checkpoint at every step, then with
    boundary saving."""
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.delenv("PYFWI_BOUNDARY_SAVING", raising=False)
    exact = forward_and_gradient(tiny_case(chpr=100, **kw))
    monkeypatch.setenv("PYFWI_BOUNDARY_SAVING", "1")
    case = tiny_case(chpr=20, **kw)
    return exact, forward_and_gradient(case), case.W


@pytest.mark.gpu
@pytest.mark.parametrize("kw", [{"sdo": 4, "ns": 2}, {"sdo": 8, "acq_type": 2, "components": 4, "ns": 3},
                                {"sdo": 4, "acq_type": 0, "components": 4, "ns": 2, "t": 0.6}])
def test_the_gradient_is_that_of_the_exact_rebuild(cuda_device, monkeypatch, kw):
    # Measured: 4.8e-7, 7.7e-7 and 1.2e-6 (the checkpoints of chpr=20 differ from the exact rebuild by up to
    # 6.6e-5 on the same cases: the rebuild between two checkpoints goes through the absorbing layer).
    (d_exact, g_exact), (d_saved, g_saved), W = _gradients(monkeypatch, kw)
    assert W.boundary_saving and list(W.chp) == [int(W.nt) - 1]
    for key in d_exact:  # the forward modelling does not change
        np.testing.assert_array_equal(d_saved[key], d_exact[key], err_msg=key)
    for key in g_exact:
        assert relative(g_exact[key], g_saved[key]) <= TOLERANCE, key


@pytest.mark.gpu
def test_the_memory_is_proportional_to_the_perimeter(cuda_device, monkeypatch):
    # MEM: three grid sizes; the store of a shot and step is 5 fields of the band (sdo cells around the
    # undamped rectangle), against 5 whole fields for one checkpoint.
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_BOUNDARY_SAVING", "1")
    per_cell = []
    for n in (40, 80, 160):
        case = tiny_case(nz=n, nx=n, t=0.05)
        W, engine = case.W, case.W._engine
        W.forward_modeling(case.model, False)  # the band and its store come with the absorbing layer
        z0, z1, x0, x1 = (int(v) for v in engine.undamped)
        height, width, w = z1 - z0 + 1, x1 - x0 + 1, int(W.sdo)
        assert engine.n_band == height * width - (height - 2 * w) * (width - 2 * w)
        per_step = engine.backend.nbytes(engine._band_store) / (int(W.ns) * int(W.nt))
        assert per_step == 5 * engine.n_band * 4
        per_cell.append(engine.n_band / (2 * (height + width)))  # cells of the band per cell of the perimeter
        assert per_step < 0.25 * 5 * engine._plane * 4  # far below one checkpoint per step
    assert max(per_cell) / min(per_cell) < 1.1  # proportional to the perimeter: about w cells per edge cell


@pytest.mark.gpu
def test_a_long_run_does_not_drift(cuda_device, monkeypatch):
    # EST: 3,500 steps rebuilt backwards in float32 without a checkpoint: the rounding of every undone
    # step does not accumulate into the gradient.
    (_, g_exact), (_, g_saved), W = _gradients(monkeypatch, {"sdo": 8, "acq_type": 2, "ns": 1, "t": 2.0})
    assert int(W.nt) > 3000
    for key in g_exact:
        assert relative(g_exact[key], g_saved[key]) <= TOLERANCE, key


@pytest.mark.gpu
def test_a_complete_inversion(cuda_device, monkeypatch):
    # FWI: the multi-scale inversion of the example script with boundary saving gives the model of the
    # exact rebuild (a checkpoint at every step; the example itself uses chpr=0, the inexact rebuild of
    # a known bug).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "scripts"))
    import fwi

    from PyFWI import wave_propagation

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    every_step = wave_propagation.checkpoint_steps
    monkeypatch.setattr(wave_propagation, "checkpoint_steps", lambda chpr, nt: every_step(100, nt))
    exact = fwi.main(size="small")
    monkeypatch.setenv("PYFWI_BOUNDARY_SAVING", "1")
    saved = fwi.main(size="small")
    assert saved["error_end"] < 0.9 * saved["error_start"]
    assert relative(exact["estimate"]["vp"], saved["estimate"]["vp"]) <= TOLERANCE
