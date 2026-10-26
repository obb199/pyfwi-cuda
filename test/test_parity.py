"""Parity of the CUDA backend with OpenCL on the same GPU (roadmap step 2.3, test type PAR).

The CUDA backend runs the same kernel files as OpenCL; parity is checked in both numeric modes,
the same mode on both backends. The two compilers (NVRTC and NVIDIA's OpenCL compiler) do not emit
the same instructions (contractions into fused multiply-adds, ordering), so the results are not
identical bit for bit. ``TOLERANCE`` is the final tolerance of gate G2, about 3 times the largest
difference measured on the whole matrix of the golden masters by ``tools/parity_report.py``
(``docs/cuda-migration/reports/parity-0.2.0.md``), relative L2 norm:

* seismograms and data at most 1.1e-6 in both modes, the order of a single bit of the PML profile
  (step 0.5: 5.7e-7); the misfit of the propagation cases 3.0e-6;
* gradients 1.9e-5 in ``legacy`` (whose constants make part of the arithmetic FP64) and 3.6e-4 in
  ``fixed`` (FP32 only), on one case: wells and surface, pressure data, energy balancing, whose
  gradient is the most sensitive to rounding (compiling the CUDA kernels without contraction into
  fused multiply-adds alone changes it by 7e-4; NVIDIA OpenCL and POCL differ by 3e-5); every other
  case is below 7e-5;
* the inversions (FWI with L-BFGS and CG, 2 iterations; eight time-lapse strategies, 1 iteration):
  models 3.5e-5, misfits 8.2e-5.

Release 0.3.0 (``parity-0.3.0.md``), where the ``fixed`` mode of CUDA runs the native kernels of Phase
3 with the shots in batches: the same largest differences (seismograms and data 1.1e-6, gradients
3.6e-4 on the same case, inverted models 3.4e-5, inversion misfits 4.7e-5), so the tolerances stay.

Also here: grids that are not a multiple of the CUDA block, determinism and ``compute-sanitizer``
on the CUDA kernels (no error: the padding of step 1.7 keeps the inherited out-of-bounds accesses
inside the allocations). The example scripts and the rest of the public API run on both backends
through the ``backend`` fixture of ``conftest.py`` (step 2.5).
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from golden_cases import inversion_cases, propagation_cases
from golden_runner import run_inversion_case, run_propagation_case
from helpers import forward_and_gradient, running_reference, tiny_case

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no CUDA")]

#: Largest relative L2 difference between CUDA and OpenCL, per numeric mode and output layer (see
#: the docstring); ``model`` and ``inversion_misfit`` are the outputs of the inversion cases.
TOLERANCE = {mode: {"raw_seismogram": 3e-6, "observed": 3e-6, "predicted": 3e-6, "misfit": 1e-5,
                    "raw_gradient": gradient, "gradient": gradient, "model": 1e-4, "inversion_misfit": 2.5e-4}
             for mode, gradient in (("legacy", 6e-5), ("fixed", 1e-3))}

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def relative_l2(expected, actual):
    """||actual - expected|| / ||expected|| in float64 (0 when both are zero)."""
    e, a = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    norm, diff = np.linalg.norm(e), np.linalg.norm(a - e)
    return diff / norm if norm else float(diff > 0)


def _on(backend, monkeypatch, run):
    """Run ``run()`` with ``PYFWI_BACKEND=backend``."""
    monkeypatch.setenv("PYFWI_BACKEND", backend)
    return run()


@pytest.fixture
def both(nvidia_opencl_platform, cuda_device, monkeypatch):
    """Run a function on OpenCL and on CUDA (the same GPU), in the numeric mode of ``PYFWI_NUMERICS``
    (legacy unless the test sets it); returns ``(opencl, cuda)``."""
    monkeypatch.setenv("PYOPENCL_CTX", str(nvidia_opencl_platform))
    monkeypatch.setenv("PYFWI_NUMERICS", "legacy")  # CUDA defaults to the fixed mode (step 2.4)
    return lambda run: (_on("opencl", monkeypatch, run), _on("cuda:strict", monkeypatch, run))


def _differences(opencl, cuda, numerics, layer_of=lambda key: key.split("/")[0]):
    """The outputs whose difference exceeds ``TOLERANCE``, as messages."""
    assert opencl.keys() == cuda.keys()
    failures = []
    for key in opencl:
        layer, error = layer_of(key), relative_l2(opencl[key], cuda[key])
        if error > TOLERANCE[numerics][layer]:
            failures.append(f"{key}: relative L2 {error:.1e} > {TOLERANCE[numerics][layer]:.0e}")
    return failures


@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
@pytest.mark.parametrize("case", propagation_cases(), ids=lambda c: c.name)
def test_cuda_matches_opencl_on_every_propagation_case(both, monkeypatch, case, numerics):
    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    opencl, cuda = both(lambda: run_propagation_case(case, guard=False))
    failures = _differences(opencl, cuda, numerics)
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
@pytest.mark.parametrize("case", inversion_cases(), ids=lambda c: c.name)
def test_inversions_agree_between_backends(both, monkeypatch, case, numerics):
    # The inversion cases of the golden masters (FWI with L-BFGS and CG, 2 iterations; the time-lapse
    # strategies, 1 iteration): the inverted models and the misfits of CUDA and OpenCL.
    from tools.parity_report import kind_of

    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    opencl, cuda = both(lambda: run_inversion_case(case, guard=False))
    failures = _differences(opencl, cuda, numerics,
                            lambda key: "inversion_misfit" if kind_of(key) == "misfit" else "model")
    assert not failures, "\n".join(failures)


def test_parity_report_renders_every_output():
    from tools.parity_report import kind_of, render, summary

    results = {"legacy": {"a1_case": {"raw_seismogram/vx": 1e-7, "raw_gradient/vp": 2e-5, "misfit": 3e-7},
                          "fwi_lbfgs": {"model/vp": 4e-6, "misfit/0": 5e-5}}}
    assert [kind_of(k) for k in ("raw_gradient/mu", "misfit", "model/vp", "dm/vp", "b1/vs", "misfit/0/1")] == \
        ["raw_gradient", "misfit", "model", "model", "model", "misfit"]
    assert summary(results["legacy"])["misfit"] == (3e-7, "a1_case", "misfit")
    assert summary(results["legacy"])["inversion misfit"] == (5e-5, "fwi_lbfgs", "misfit/0")
    report = render(results, {"GPU": "test GPU"}, "9.9.9")
    assert report.startswith("# Parity of the CUDA backend with OpenCL (release 9.9.9)") and "- GPU: test GPU" in report
    assert "| raw_gradient | 2.0e-05 | a1_case | `raw_gradient/vp` |" in report
    assert report.count("\n| a1_case |") == 3 and "| fwi_lbfgs | `model/vp` | 4.0e-06 |" in report


@pytest.mark.parametrize("tnz, tnx", [(61, 67), (100, 100), (127, 129)])
def test_grids_that_are_not_a_multiple_of_the_block(both, tnz, tnx):
    # tiny_case adds a 10-point boundary on each side: the grids are exactly tnz x tnx.
    def run():
        return forward_and_gradient(tiny_case(nz=tnz - 20, nx=tnx - 20, t=0.15))
    (d_opencl, g_opencl), (d_cuda, g_cuda) = both(run)
    for key in d_opencl:
        assert relative_l2(d_opencl[key], d_cuda[key]) <= TOLERANCE["legacy"]["observed"], key
    for key in g_opencl:
        assert relative_l2(g_opencl[key], g_cuda[key]) <= TOLERANCE["legacy"]["gradient"], key


@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
def test_cuda_results_do_not_depend_on_the_block(cuda_device, monkeypatch, numerics):
    # Step 3.1: the block shape (the rule of CUDABackend.block, or PYFWI_CUDA_BLOCK) changes no bit.
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_NUMERICS", numerics)
    runs = []
    for forced in (None, "64x4", "16x8"):
        if forced:
            monkeypatch.setenv("PYFWI_CUDA_BLOCK", forced)
        runs.append(forward_and_gradient(tiny_case(acq_type=2, components=4, nz=41, nx=53)))
    for data, gradient in runs[1:]:
        for key in data:
            np.testing.assert_array_equal(data[key], runs[0][0][key], err_msg=key)
        for key in gradient:
            np.testing.assert_array_equal(gradient[key], runs[0][1][key], err_msg=key)


def test_cuda_runs_are_deterministic(cuda_device, monkeypatch):
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    first, second = forward_and_gradient(tiny_case(acq_type=2)), forward_and_gradient(tiny_case(acq_type=2))
    for a, b in zip(first, second):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


#: Forward modelling and gradient of the three acquisitions on CUDA, each CuPy array in its own
#: allocation (no memory pool), so that compute-sanitizer sees the bounds of every allocation; then 3
#: shots in one batch (step 3.9: the slots, the checkpoint copies of a batch, the sum of the slots), the
#: same with boundary saving (step 4.2), and with half-precision fields too (step 4.3).
#: argv: <test folder>.
SANITIZED = """
import os, sys
sys.path[:0] = [sys.argv[1], os.path.dirname(sys.argv[1])]
os.environ["PYFWI_BACKEND"] = "cuda:strict"
import cupy
cupy.cuda.set_allocator(None)
from helpers import forward_and_gradient, tiny_case
for acq_type in (0, 1, 2):
    forward_and_gradient(tiny_case(acq_type=acq_type, components=4, t=0.1))
batched = tiny_case(acq_type=2, components=4, ns=3, t=0.1)
assert batched.W._engine.batch == 3
forward_and_gradient(batched)
os.environ["PYFWI_BOUNDARY_SAVING"] = "1"  # step 4.2: the band stored and written back
saved = tiny_case(acq_type=2, components=4, ns=3, t=0.1)
assert saved.W.boundary_saving
forward_and_gradient(saved)
os.environ["PYFWI_FP16"] = "1"  # step 4.3: half-precision fields, with boundary saving
half = tiny_case(acq_type=2, components=4, ns=3, t=0.1)
assert half.W.fp16
forward_and_gradient(half)
"""


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("compute-sanitizer") is None, reason="compute-sanitizer not installed")
def test_compute_sanitizer_finds_no_error_in_the_cuda_kernels(cuda_device, tmp_path):
    program = tmp_path / "sanitized.py"
    program.write_text(SANITIZED)
    run = subprocess.run(["bash", os.path.join(REPO, "tools", "profile.sh"), "memcheck", str(program),
                          os.path.dirname(__file__)], env={**os.environ, "PY": sys.executable},
                         capture_output=True, text=True, timeout=1800, check=False)
    assert run.returncode == 0 and "ERROR SUMMARY: 0 errors" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]


#: The CUDA 12 environment of tools/setup_envs.sh (roadmap step 2.6): the [cuda12] extra.
CUDA12_PYTHON = os.path.join(os.path.dirname(REPO), "envs", "cuda12", "bin", "python")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(CUDA12_PYTHON), reason="envs/cuda12 not created (tools/setup_envs.sh cuda12)")
def test_fast_parity_tests_pass_with_cuda_12(cuda_device):
    # The fast PAR tests and the CUDA backend tests with the CUDA 12 runtime and NVRTC 12 of
    # cupy-cuda12x, against NVIDIA OpenCL on the same GPU (this repository, installed editable there).
    program = "import cupy; print(cupy.cuda.runtime.runtimeGetVersion(), cupy.cuda.nvrtc.getVersion())"
    versions = subprocess.run([CUDA12_PYTHON, "-c", program], capture_output=True, text=True, check=True)
    runtime, nvrtc = versions.stdout.split(maxsplit=1)
    assert runtime.startswith("12") and nvrtc.startswith("(12,"), versions.stdout
    run = subprocess.run([CUDA12_PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow",
                          os.path.join("test", "test_parity.py"), os.path.join("test", "test_cuda_backend.py")],
                         cwd=REPO, capture_output=True, text=True, timeout=1800, check=False)
    assert run.returncode == 0, run.stdout[-3000:]
    assert " passed" in run.stdout and "skipped" not in run.stdout, run.stdout[-1000:]
