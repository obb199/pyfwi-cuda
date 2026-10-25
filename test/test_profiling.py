"""Self-tests of the profiling bench (roadmap step 0.8).

Before relying on a tool to find problems in the CUDA kernels of Phase 2, check that it
works on this machine, with the CUDA 13 runtime installed by pip:

* ``compute-sanitizer`` must report a deliberate out-of-bounds write in a CuPy kernel,
  and nothing for a correct kernel (the sanitizer installed with the system CUDA
  Toolkit is version 2024.1, from CUDA 12.4);
* Nsight Systems must record the kernels of a CuPy script;
* Nsight Compute can only be checked when the driver gives access to the performance
  counters (``RmProfilingAdminOnly: 0``); otherwise the test is skipped with the reason;
* ``tools/opencl_profile.py`` must account for the GPU time of PyFWI's OpenCL kernels.
"""
import os
import shutil
import subprocess
import sys

import pytest

#: Absolute path of the repository root and of the recipes script.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_SH = os.path.join(REPO, "tools", "profile.sh")

#: A CuPy program with one kernel launch: out[i + shift] = 1 for i < n, with shift from argv.
#: shift = 1 writes one element past the end of ``out`` (n elements).
CUPY_PROGRAM = """
import sys
import cupy as cp
import numpy as np
kernel = cp.RawKernel(r'''extern "C" __global__ void fill(float* out, int n, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i + shift] = 1.0f;
}''', "fill")
n = 1024
out = cp.zeros(n, dtype=cp.float32)
kernel((n // 256,), (256,), (out, np.int32(n), np.int32(int(sys.argv[1]))))
cp.cuda.Device().synchronize()
print("done")
"""


def _write_program(tmp_path):
    path = tmp_path / "fill.py"
    path.write_text(CUPY_PROGRAM)
    return str(path)


def _counters_restricted():
    """True when the NVIDIA driver restricts the performance counters to root."""
    try:
        with open("/proc/driver/nvidia/params") as f:
            return "RmProfilingAdminOnly: 1" in f.read()
    except OSError:
        return True


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("compute-sanitizer") is None, reason="compute-sanitizer not installed")
def test_sanitizer_reports_an_out_of_bounds_write_and_nothing_else(cuda_device, tmp_path):
    program = _write_program(tmp_path)
    clean = subprocess.run(["bash", PROFILE_SH, "memcheck", program, "0"], env={**os.environ, "PY": sys.executable},
                           capture_output=True, text=True, timeout=300, check=False)
    assert clean.returncode == 0, clean.stdout[-2000:] + clean.stderr[-2000:]
    assert "ERROR SUMMARY: 0 errors" in clean.stdout
    broken = subprocess.run(["bash", PROFILE_SH, "memcheck", program, "1"], env={**os.environ, "PY": sys.executable},
                            capture_output=True, text=True, timeout=300, check=False)
    assert broken.returncode == 99, broken.stdout[-2000:] + broken.stderr[-2000:]
    assert "Invalid __global__ write" in broken.stdout


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("nsys") is None, reason="Nsight Systems not installed")
def test_nsight_systems_records_the_kernel(cuda_device, tmp_path):
    program = _write_program(tmp_path)
    r = subprocess.run(["bash", PROFILE_SH, "nsys", program, "0"], cwd=tmp_path,
                       env={**os.environ, "PY": sys.executable, "OUT": str(tmp_path / "trace")},
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (tmp_path / "trace.nsys-rep").exists()
    assert "fill" in r.stdout  # the kernel appears in the kernel summary


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("ncu") is None, reason="Nsight Compute not installed")
@pytest.mark.skipif(_counters_restricted(), reason="GPU performance counters restricted to root (RmProfilingAdminOnly: 1)")
def test_nsight_compute_measures_the_kernel(cuda_device, tmp_path):
    program = _write_program(tmp_path)
    r = subprocess.run(["bash", PROFILE_SH, "ncu", program, "0"], cwd=tmp_path,
                       env={**os.environ, "PY": sys.executable, "OUT": str(tmp_path / "kernels")},
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "fill" in r.stdout and "gpu__time_duration.sum" in r.stdout


def test_profile_script_usage_error():
    r = subprocess.run(["bash", PROFILE_SH, "nothing"], capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "usage:" in r.stderr


def test_opencl_profile_accounts_for_gpu_time(opencl_platform):
    from tools.opencl_profile import profile_scenario
    result = profile_scenario("tiny")
    assert result["launches"] > 0 and 0 < result["gpu_s"] < result["wall_s"]
    assert {"update_velx", "Grad", "injSrc"} <= set(result["kernels"])
    assert sum(k["launches"] for k in result["kernels"].values()) == result["launches"]
