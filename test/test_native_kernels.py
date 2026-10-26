"""The native CUDA kernels of Phase 3 (``PyFWI/kernels/elastic.cu``), test type OTM.

In the fixed numeric mode the CUDA backend runs the native kernels: the stencil kernels written for
CUDA, which the steps of Phase 3 optimise one at a time (the injection kernels still come from the
OpenCL files). Every step is checked against the *parity* kernels, the OpenCL files compiled for CUDA
in the same mode (``PYFWI_CUDA_KERNELS=parity``), on the 16 propagation cases of the golden masters,
with the tolerance of the roadmap (OTM: seismograms 1e-5, gradients 1e-4, relative L2).

Measured at step 3.2 (the order as a template parameter, the expressions in the order of elastic.cl):
bit for bit equal to the parity kernels on every case.
"""
import numpy as np
import pytest

from golden_cases import propagation_cases
from golden_runner import run_propagation_case
from helpers import native_kernel_source, running_reference, tiny_case

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no CUDA kernels")

#: OTM tolerance of the roadmap, per output layer (relative L2 against the parity kernels).
TOLERANCE = {"raw_seismogram": 1e-5, "observed": 1e-5, "predicted": 1e-5, "misfit": 1e-5,
             "raw_gradient": 1e-4, "gradient": 1e-4}

#: The kernels of the native program: every kernel of the engine but the injection ones.
NATIVE = ("MakeAllZero", "MakeGradZero", "update_velx", "update_velz", "update_tauz", "update_tauxz",
          "Adj_update_tau", "Adj_update_v", "Grad")


def relative_l2(expected, actual):
    """``||actual - expected|| / ||expected||`` in float64 (0 when both are zero)."""
    e, a = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    norm, diff = np.linalg.norm(e), np.linalg.norm(a - e)
    return diff / norm if norm else float(diff > 0)


@pytest.mark.gpu
@pytest.mark.parametrize("case", propagation_cases(), ids=lambda c: c.name)
def test_native_kernels_match_the_parity_kernels(cuda_device, monkeypatch, case):
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_NUMERICS", "fixed")
    outputs = {}
    for kernels in ("parity", "native"):
        monkeypatch.setenv("PYFWI_CUDA_KERNELS", kernels)
        outputs[kernels] = run_propagation_case(case, guard=False)
    parity, native = outputs["parity"], outputs["native"]
    assert parity.keys() == native.keys()
    failures = [f"{key}: {relative_l2(parity[key], native[key]):.1e}" for key in parity
                if relative_l2(parity[key], native[key]) > TOLERANCE[key.split("/")[0]]]
    assert not failures, "\n".join(failures)


@pytest.mark.gpu
def test_the_fixed_mode_runs_the_native_kernels(cuda_device, monkeypatch):
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.delenv("PYFWI_CUDA_KERNELS", raising=False)
    W = tiny_case(inpa_extra={"numerics": "fixed"}).W
    assert W.cuda_kernels == "native" and W.prg_native is not None
    assert all(getattr(W.knl, name).function_name == name for name in NATIVE)
    assert W.knl.injSrc.function_name == "injSrc"  # from the OpenCL files, until step 3.4
    legacy = tiny_case(inpa_extra={"numerics": "legacy"}).W
    assert legacy.cuda_kernels == "parity" and legacy.prg_native is None
    monkeypatch.setenv("PYFWI_CUDA_KERNELS", "parity")
    assert tiny_case(inpa_extra={"numerics": "fixed"}).W.cuda_kernels == "parity"


@pytest.mark.gpu
@pytest.mark.parametrize("order", [4, 8])
def test_native_kernels_have_no_fp64_instruction(cuda_device, order):
    # PRE: the native kernels compute in FP32 only.
    from tools.check_env import compile_ptx, count_fp64
    counts = count_fp64(compile_ptx(native_kernel_source(order)))
    assert set(NATIVE) <= set(counts) and not any(counts.values()), counts


@pytest.mark.gpu
def test_order_4_reads_fewer_cells_than_order_8(cuda_device):
    # The order is a template parameter: order 4 has no term of c3 and c4 (zero), so each derivative
    # reads 4 cells instead of 8 (loads in the PTX of update_velx: two derivatives and five fields).
    from tools.check_env import compile_ptx

    def loads(order):
        ptx = compile_ptx(native_kernel_source(order))
        body = ptx[ptx.index(".entry update_velx("):]
        body = body[:body.index("\n}")]
        return body.count("ld.global")

    assert loads(4) + 8 <= loads(8), (loads(4), loads(8))
