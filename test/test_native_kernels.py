"""The native CUDA kernels of Phase 3 (``PyFWI/kernels/elastic.cu``), test type OTM.

In the fixed numeric mode the CUDA backend runs the native kernels, written for CUDA, which the
steps of Phase 3 optimise one at a time. Every step is checked against the *parity* kernels, the OpenCL files compiled for CUDA
in the same mode (``PYFWI_CUDA_KERNELS=parity``), on the 16 propagation cases of the golden masters,
with the tolerance of the roadmap (OTM: seismograms 1e-5, gradients 1e-4, relative L2).

Measured, the largest difference on the 16 cases:

* step 3.2 (the order as a template parameter, the expressions in the order of elastic.cl): none,
  bit for bit equal to the parity kernels;
* step 3.3 (multiplications by 1 / dx and 1 / dz instead of divisions): seismograms 9.3e-7, misfit
  5.6e-6, gradients 3.8e-5 (rounding);
* step 3.4 (source, receivers and residual handled one thread per receiver): the same;
* step 3.5 (fused kernels): the same; the fused kernels give the same bits as the separate native
  kernels (``test_fused_kernels_give_the_bits_of_the_separate_ones``);
* step 3.6 (the time step on the device): the same;
* step 3.7 (CUDA graphs): the same; replaying the graphs gives the bits of the step-by-step loops
  (``test_cuda_graphs_give_the_bits_of_the_step_by_step_loops``);
* step 3.8 (checkpoints on the device): the same, with the checkpoints on the device, on the host or
  split (``test_checkpoints_give_the_same_bits_on_the_device_on_the_host_and_split``);
* step 3.10 (no damping read inside the absorbing layer's zero rectangle): the same; the fused kernels,
  which skip the reads, still give the bits of the separate ones, which do not.
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
    assert W.native_injection and W.knl.inject_and_record.function_name == "inject_and_record"
    legacy = tiny_case(inpa_extra={"numerics": "legacy"}).W
    assert legacy.cuda_kernels == "parity" and legacy.prg_native is None
    monkeypatch.setenv("PYFWI_CUDA_KERNELS", "parity")
    assert tiny_case(inpa_extra={"numerics": "fixed"}).W.cuda_kernels == "parity"


@pytest.mark.gpu
@pytest.mark.parametrize("kw", [{"sdo": 4}, {"sdo": 8, "acq_type": 2, "components": 4},
                                {"sdo": 4, "acq_type": 0, "components": 4, "chpr": 0}])
def test_fused_kernels_give_the_bits_of_the_separate_ones(cuda_device, monkeypatch, kw):
    # Step 3.5: update_v, update_tau and grad_adjoint_v do, cell by cell, what the separate native
    # kernels do; the engine runs the separate ones when its ``fused`` flag is off.
    from helpers import forward_and_gradient

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_NUMERICS", "fixed")
    fused_case, separate_case = tiny_case(**kw), tiny_case(**kw)
    assert fused_case.W._engine.fused
    separate_case.W._engine.fused = False
    fused, separate = forward_and_gradient(fused_case), forward_and_gradient(separate_case)
    for a, b in zip(fused, separate):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.gpu
@pytest.mark.parametrize("chpr", [0, 20, 100])
def test_cuda_graphs_give_the_bits_of_the_step_by_step_loops(cuda_device, monkeypatch, chpr):
    # Step 3.7: chpr 0 (two checkpoints: long runs, split into graphs of MAX_GRAPH_STEPS), 20 and 100
    # (a checkpoint at every step: graphs of one step).
    from helpers import forward_and_gradient

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_NUMERICS", "fixed")
    runs = []
    for graphs in ("1", "0"):
        monkeypatch.setenv("PYFWI_CUDA_GRAPHS", graphs)
        case = tiny_case(sdo=8, acq_type=2, components=4, chpr=chpr)
        assert case.W._engine.graphs == (graphs == "1")
        runs.append(forward_and_gradient(case))
    for a, b in zip(*runs):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.gpu
def test_checkpoints_give_the_same_bits_on_the_device_on_the_host_and_split(cuda_device, monkeypatch):
    # MEM, step 3.8: with a forced budget the checkpoints of the 3 shots stay on the host, of 2 of
    # them on the device, or of all; the results are the same bits, and the tier holds during the runs.
    from helpers import forward_and_gradient

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_NUMERICS", "fixed")
    kw = {"ns": 3, "sdo": 8, "acq_type": 2, "components": 4, "chpr": 50}
    probe = tiny_case(**kw).W._engine
    per_shot = 5 * probe.nchp * probe._plane * 4
    runs = []
    for shots in (0, 2, 3):
        monkeypatch.setenv("PYFWI_CUDA_CHECKPOINT_BUDGET", str(shots * per_shot))
        case = tiny_case(**kw)
        assert case.W._engine._device_shots == shots
        runs.append(forward_and_gradient(case))
        assert case.W._engine._device_shots == shots
    for other in runs[1:]:
        for a, b in zip(runs[0], other):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)


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


@pytest.mark.gpu
@pytest.mark.parametrize("acq_type", [0, 1, 2])
def test_receiver_table_finds_the_cells_the_opencl_kernels_record(cuda_device, monkeypatch, acq_type):
    # Step 3.4: the native kernels read the cell of every receiver from receiver_table. Fill vx with
    # 1 + the flat index of each cell, record one row with injSrc of the parity kernels: the seismogram
    # then holds 1 + the cell each receiver reads (0 where no work item records).
    from PyFWI.engine.propagation import receiver_table

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_NUMERICS", "fixed")
    monkeypatch.setenv("PYFWI_CUDA_KERNELS", "parity")
    W = tiny_case(acq_type=acq_type, components=4).W
    engine = W._engine
    cells = np.arange(1, W.tnz * W.tnx + 1, dtype=np.float32).reshape(W.tnz, W.tnx)
    engine.backend.to_device(engine.vx_b, cells)
    engine._set_injection_args(0)
    engine._inject(np.zeros(5, np.float32), 0)
    engine.backend.finish()
    recorded = np.empty(int(W.nt_ext) * int(W.nr), np.float32)
    engine.backend.to_host(recorded, engine.seismogramid_vx_b)
    table, _, _ = receiver_table(W.acq_type, int(W.tnz), int(W.tnx), int(W.nr), int(W.dxr), int(W.n_surface_rec),
                                 int(W.n_well_rec), (int(W.rec_surface_const), int(W.rec_surface_var)),
                                 (int(W.rec_top_left_const), int(W.rec_top_left_var)),
                                 (int(W.rec_top_right_const), int(W.rec_top_right_var)))
    np.testing.assert_array_equal(recorded[:int(W.nr)].astype(np.int64) - 1, table)
