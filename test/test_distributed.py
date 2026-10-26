"""The shots of a propagator shared between processes (roadmap step 4.4, ``PyFWI.engine.distributed``).

``PYFWI_CUDA_DEVICES`` lists CUDA devices, one process per entry. With a single RTX the test of the roadmap
runs two processes on the same GPU (``0,0``): the results must be the bits of one process (DET), and a
failing process must give a clear error. The engine's bookkeeping is tested on the fake backend
(``test_engine.py``). Scaling with several GPUs needs a machine that has them: recorded at gate G4.
"""
import os
import sys

import numpy as np
import pytest

from helpers import forward_and_gradient, running_reference, tiny_case

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no distributed shots")]

KW = {"sdo": 8, "acq_type": 2, "components": 4, "ns": 3}


def _run(monkeypatch, devices, **environment):
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    if devices is None:
        monkeypatch.delenv("PYFWI_CUDA_DEVICES", raising=False)
    else:
        monkeypatch.setenv("PYFWI_CUDA_DEVICES", devices)
    case = tiny_case(**KW)
    try:
        return forward_and_gradient(case), case.W
    finally:
        getattr(case.W._engine, "close", lambda: None)()


@pytest.mark.parametrize("environment", [{}, {"PYFWI_FP16": "1", "PYFWI_BOUNDARY_SAVING": "1"}])
def test_two_processes_give_the_bits_of_one(cuda_device, monkeypatch, environment):
    # DET: the gradient of every shot comes back on its own and the shots are added in their order, so one
    # process and two (here on the same GPU) give the same bits; with the opt-in features too (the FP16
    # scales are powers of two: a shot's rounding does not depend on its process).
    (d_one, g_one), W = _run(monkeypatch, "0", **environment)
    (d_two, g_two), _ = _run(monkeypatch, "0,0", **environment)
    assert type(W._engine).__name__ == "DistributedEngine" and W._engine.blocks == [range(3)]
    for key in d_one:
        np.testing.assert_array_equal(d_one[key], d_two[key], err_msg=key)
    for key in g_one:
        np.testing.assert_array_equal(g_one[key], g_two[key], err_msg=key)
    # Against the engine of a single process: the same seismograms; the gradients add the shots in
    # another order (the slots of a batch, step 3.9).
    (d_single, g_single), _ = _run(monkeypatch, None, **environment)
    for key in d_one:
        np.testing.assert_array_equal(d_one[key], d_single[key], err_msg=key)
    for key in g_one:
        assert np.linalg.norm(g_one[key] - g_single[key]) <= 1e-5 * np.linalg.norm(g_single[key]), key


def test_a_failing_process_gives_a_clear_error(cuda_device, monkeypatch):
    with pytest.raises(RuntimeError, match=r"PyFWI worker 1 \(CUDA device 9\) failed:"):
        _run(monkeypatch, "0,9")
    monkeypatch.setenv("PYFWI_BACKEND", "opencl")
    monkeypatch.setenv("PYFWI_CUDA_DEVICES", "0")
    with pytest.raises(ValueError, match="PYFWI_CUDA_DEVICES needs the CUDA backend"):
        tiny_case()


def test_an_inversion_in_two_processes(cuda_device, monkeypatch):
    # FWI: the multi-scale inversion of the example script in two processes gives the bits of one process.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "scripts"))
    import fwi

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    results = []
    for devices in ("0", "0,0"):
        monkeypatch.setenv("PYFWI_CUDA_DEVICES", devices)
        results.append(fwi.main(size="small"))
    np.testing.assert_array_equal(results[0]["estimate"]["vp"], results[1]["estimate"]["vp"])
    assert results[1]["error_end"] < 0.9 * results[1]["error_start"]
