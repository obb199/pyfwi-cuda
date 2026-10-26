"""Parity of the CUDA backend with OpenCL on the same GPU (roadmap step 2.3, test type PAR).

The CUDA backend runs the same kernel files as OpenCL; parity is checked in the ``legacy`` numeric
mode of both (CUDA defaults to ``fixed`` since step 2.4, see ``test_numerics.py``). The two
compilers (NVRTC and NVIDIA's OpenCL compiler) do not emit the same instructions (contractions
into fused multiply-adds, ordering), so the results are not identical bit for bit: measured on the
16 propagation cases of the golden masters, the seismograms differ by at most 1.1e-6 and the
gradients by 1.9e-5 in relative L2 norm, the same order as the effect of a single bit of the PML
profile measured in step 0.5 (5.7e-7 and 2.8e-5). ``TOLERANCE`` leaves a margin of about 3; the
final tolerance is set at gate G2.

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

from golden_cases import propagation_cases
from golden_runner import run_propagation_case
from helpers import forward_and_gradient, running_reference, tiny_case

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no CUDA")]

#: Largest relative L2 difference between CUDA and OpenCL, per output layer (see the docstring).
TOLERANCE = {"raw_seismogram": 3e-6, "observed": 3e-6, "predicted": 3e-6, "misfit": 3e-6,
             "raw_gradient": 6e-5, "gradient": 6e-5}

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
    """Run a function on OpenCL and on CUDA (the same GPU), both in the legacy mode; returns ``(opencl, cuda)``."""
    monkeypatch.setenv("PYOPENCL_CTX", str(nvidia_opencl_platform))
    monkeypatch.setenv("PYFWI_NUMERICS", "legacy")  # CUDA defaults to the fixed mode (step 2.4)
    return lambda run: (_on("opencl", monkeypatch, run), _on("cuda:strict", monkeypatch, run))


@pytest.mark.parametrize("case", propagation_cases(), ids=lambda c: c.name)
def test_cuda_matches_opencl_on_every_propagation_case(both, case):
    opencl, cuda = both(lambda: run_propagation_case(case, guard=False))
    assert opencl.keys() == cuda.keys()
    failures = []
    for key in opencl:
        layer, error = key.split("/")[0], relative_l2(opencl[key], cuda[key])
        if error > TOLERANCE[layer]:
            failures.append(f"{key}: relative L2 {error:.1e} > {TOLERANCE[layer]:.0e}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("tnz, tnx", [(61, 67), (100, 100), (127, 129)])
def test_grids_that_are_not_a_multiple_of_the_block(both, tnz, tnx):
    # tiny_case adds a 10-point boundary on each side: the grids are exactly tnz x tnx.
    def run():
        return forward_and_gradient(tiny_case(nz=tnz - 20, nx=tnx - 20, t=0.15))
    (d_opencl, g_opencl), (d_cuda, g_cuda) = both(run)
    for key in d_opencl:
        assert relative_l2(d_opencl[key], d_cuda[key]) <= TOLERANCE["observed"], key
    for key in g_opencl:
        assert relative_l2(g_opencl[key], g_cuda[key]) <= TOLERANCE["gradient"], key


def test_cuda_runs_are_deterministic(cuda_device, monkeypatch):
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    first, second = forward_and_gradient(tiny_case(acq_type=2)), forward_and_gradient(tiny_case(acq_type=2))
    for a, b in zip(first, second):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


#: Forward modelling and gradient of the three acquisitions on CUDA, each CuPy array in its own
#: allocation (no memory pool), so that compute-sanitizer sees the bounds of every allocation.
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
