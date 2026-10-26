"""Numeric modes (roadmap step 2.4): ``legacy`` computes what PyFWI 0.1.10 computes, ``fixed`` corrects it.

The mode of a propagator: ``inpa['numerics']``, else ``PYFWI_NUMERICS``, else the default of its
backend (OpenCL: ``legacy``, CUDA: ``fixed``). The ``legacy`` sources are exactly those of 0.1.10
(the characterisation tests check its results bit for bit). The corrections of the ``fixed`` mode
so far:

* the floating-point constants of the ``#define`` block are float literals (suffix ``f``): the
  kernels compute in FP32 only (test PRE: no FP64 instruction in the PTX). Alone this changes the
  results by rounding (measured: 1e-6 for seismograms, 5e-5 for gradients, relative L2);
* ``dt`` keeps 9 significant digits instead of 6 decimals (0.000571428... instead of 0.000571).
  0.1.10 samples the source with the true step but propagates with the truncated one, 0.075 %
  shorter; correcting it changes seismograms by about 2 % and gradients by about 6 %.
"""
import numpy as np
import pytest

from golden_cases import propagation_cases
from golden_runner import run_propagation_case
from golden_store import load_case
from helpers import kernel_sources, running_reference, tiny_case

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no numeric modes")

#: The kernels that run at every time step.
HOT = ("update_velx", "update_velz", "update_tauz", "update_tauxz", "Adj_update_tau", "Adj_update_v", "Grad")


def test_fixed_mode_writes_float_constants_and_the_full_time_step():
    legacy, fixed = kernel_sources(8)[0], kernel_sources(8, numerics="fixed")[0]
    assert "#define dt     0.000571\n" in legacy and "PYFWI_FIXED" not in legacy
    assert "#define dt     5.714285714e-04f\n" in fixed and "#define PYFWI_FIXED 1" in fixed
    assert "#define c4     -6.973500697e-04f\n" in fixed


@pytest.mark.gpu
@pytest.mark.parametrize("order", [4, 8])
def test_fixed_kernels_have_no_fp64_instruction(cuda_device, order):
    # PRE: in the fixed mode no kernel uses FP64 (the legacy mode does, see test_check_env.py).
    from tools.check_env import compile_ptx, count_fp64
    main, crosswell, surface = kernel_sources(order, numerics="fixed")
    counts = count_fp64(compile_ptx(main))
    assert set(HOT) <= set(counts) and not any(counts.values()), counts
    for source in (crosswell, surface):
        counts = count_fp64(compile_ptx(source))
        assert {"injSrc", "Adj_injSrc"} <= set(counts) and not any(counts.values()), counts


def test_propagators_take_their_mode(opencl_platform, monkeypatch):
    monkeypatch.delenv("PYFWI_NUMERICS", raising=False)
    monkeypatch.delenv("PYFWI_BACKEND", raising=False)
    assert tiny_case().W.numerics == "legacy"
    W = tiny_case(inpa_extra={"numerics": "fixed"}).W
    assert W.numerics == "fixed" and "PYFWI_FIXED" in W.kernel_caller()[0]


@pytest.mark.parametrize("backend", ["opencl", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fixed_against_legacy_on_the_propagation_cases(opencl_platform, monkeypatch, request, backend):
    # The differences published in the CHANGELOG: the full dt dominates (about 2 % on seismograms,
    # 6 % on gradients); the bounds catch a fixed mode that would change much more, or nothing.
    if backend == "cuda":
        request.getfixturevalue("cuda_device")  # skips without CuPy or without a GPU
    monkeypatch.setenv("PYFWI_BACKEND", f"{backend}:strict" if backend == "cuda" else backend)
    monkeypatch.setenv("PYFWI_NUMERICS", "fixed")
    worst = {}
    for case in propagation_cases():
        golden, fixed = load_case(case.name), run_propagation_case(case, guard=False)
        for key in golden:
            layer = key.split("/")[0]
            e, a = np.asarray(golden[key], np.float64), np.asarray(fixed[key], np.float64)
            worst[layer] = max(worst.get(layer, 0), np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-300))
    assert 5e-3 < worst["raw_seismogram"] < 5e-2 and 5e-3 < worst["raw_gradient"] < 0.15, worst
