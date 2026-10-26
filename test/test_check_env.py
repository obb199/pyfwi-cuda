"""Tests of the environment checker (``tools/check_env.py``) and of the environment files.

What is covered
---------------
* Pure helpers of ``check_env`` (no device needed): FP64 counting in PTX, the
  check/warning bookkeeping, the synthetic model, the output sanity rule.
* The pinned environment files in ``requirements/``: the reference environment
  must keep the versions of PyFWI 0.1.10, and both environments must use the
  same POCL so that CPU results can be compared bit for bit.
* ``tools/setup_envs.sh``: valid bash, usage error on a bad argument.
* The command-line program, with and without a visible GPU.
* With hardware: the PyFWI OpenCL kernels compile as CUDA (marker ``gpu``) and
  the PyFWI smoke test is finite and deterministic on NVIDIA OpenCL (marker ``opencl``).
"""
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import check_env

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: A hand-written PTX listing with one kernel that uses FP64 and one that does not.
#: ``.reg .f64`` is a register declaration, not an instruction, and must not be counted;
#: ``cvt.f64.f32``, ``mul.f64`` and ``cvt.rn.f32.f64`` are three FP64 instructions.
PTX = """
.visible .entry k_double(
    .param .u64 p
)
{
    .reg .f64 %fd<4>;
    ld.global.f32 %f1, [%rd1];
    cvt.f64.f32 %fd1, %f1;
    mul.f64 %fd2, %fd1, 0d3FF2000000000000;
    cvt.rn.f32.f64 %f2, %fd2;
    ret;
}
.visible .entry k_float(
    .param .u64 p
)
{
    .reg .f32 %f<4>;
    mul.f32 %f2, %f1, 0f3F900000;
    ret;
}
"""


def test_count_fp64_counts_per_kernel():
    assert check_env.count_fp64(PTX) == {"k_double": 3, "k_float": 0}


def test_count_fp64_empty_ptx():
    assert check_env.count_fp64("") == {}


def test_check_known_issue_is_warning_not_failure(capsys):
    check_env.FAILURES.clear()
    assert check_env.check("passes", True)
    assert not check_env.check("known", False, known_issue="old bug")
    assert check_env.FAILURES == []  # a known issue is not a failure
    assert not check_env.check("breaks", False)
    assert check_env.FAILURES == ["breaks"]
    out = capsys.readouterr().out
    assert "[WARN ] known: old bug" in out and "[FAIL ] breaks" in out
    check_env.FAILURES.clear()


def test_inclusion_model_is_elastic_and_float32():
    m = check_env.inclusion_model(40, 50)
    assert set(m) == {"vp", "vs", "rho"}
    assert all(v.shape == (40, 50) and v.dtype == np.float32 for v in m.values())
    assert set(np.unique(m["vp"])) == {2500.0, 3000.0}
    np.testing.assert_allclose(m["vp"] / m["vs"], np.sqrt(3.0), rtol=1e-6)
    smooth = check_env.inclusion_model(40, 50, smooth=1)["vp"]
    assert 2500.0 <= smooth.min() and smooth.max() <= 3000.0 and len(np.unique(smooth)) > 2


def test_outputs_ok():
    good = {"d_taux": np.ones(3), "d_vx": np.zeros(3), "g_vp": np.ones(2)}
    assert check_env.outputs_ok(good)  # an all-zero d_vx is allowed (pressure-only recording)
    assert not check_env.outputs_ok({**good, "d_taux": np.array([1.0, np.nan, 0.0])})
    assert not check_env.outputs_ok({**good, "d_taux": np.zeros(3)})
    assert not check_env.outputs_ok({**good, "g_vp": np.zeros(2)})


def _pins(name):
    """Read ``requirements/<name>`` and return ``{package (lower case): version}`` for ``pkg==version`` lines."""
    pins = {}
    with open(os.path.join(REPO, "requirements", name)) as f:
        for line in f:
            m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line.strip())
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def test_reference_env_matches_original_requirements():
    ref = _pins("ref-0.1.10.txt")
    # Versions of the requirements.txt of 0.1.10: the reference must not depend on any fix.
    for pkg, ver in {"numpy": "1.21.6", "scipy": "1.7.3", "matplotlib": "3.5.3",
                     "pyopencl": "2022.2.4", "segyio": "1.9.11"}.items():
        assert ref[pkg] == ver, pkg
    assert ref["torch"].startswith("1.13.1")
    assert "pyfwi" not in ref  # installed separately, from PyPI, without dependencies


def test_dev_env_has_cuda13_stack():
    dev = _pins("dev.txt")
    assert dev["cupy-cuda13x"].startswith("14.")
    assert dev["cuda-core"] and dev["pyopencl"] and dev["pytest-benchmark"]
    assert "cupy-cuda12x" not in dev


def test_cuda12_env_has_cuda12_stack_and_the_rest_of_dev():
    # envs/cuda12 (roadmap step 2.6): CuPy and its toolkit for CUDA 12; everything else at the
    # versions of dev.txt, so that a difference between the two environments is the CUDA version.
    cuda12, dev = _pins("cuda12.txt"), _pins("dev.txt")
    assert cuda12["cupy-cuda12x"].startswith("14.") and cuda12["nvidia-cuda-nvrtc-cu12"].startswith("12.")
    assert "cupy-cuda13x" not in cuda12 and cuda12["cuda-core"] and cuda12["pyopencl"]
    shared = {name for name in cuda12 if name in dev and not re.match(r"(cuda-|cupy-|nvidia-)", name)}
    assert {"numpy", "scipy", "pyopencl", "pocl-binary-distribution", "pytest"} <= shared
    assert {name: cuda12[name] for name in shared} == {name: dev[name] for name in shared}


def test_both_envs_pin_same_pocl():
    # Bit-for-bit comparisons on the CPU OpenCL need the same POCL compiler on both sides.
    assert _pins("ref-0.1.10.txt")["pocl-binary-distribution"] == _pins("dev.txt")["pocl-binary-distribution"]


def test_setup_envs_script_is_valid_bash():
    script = os.path.join(REPO, "tools", "setup_envs.sh")
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", script], check=True)  # -n: parse only, run nothing
    r = subprocess.run(["bash", script, "nothing"], check=False, capture_output=True, text=True)
    assert r.returncode == 2 and "usage:" in r.stderr and "cuda12" in r.stderr


def _run_check_env(*args, env=None):
    """Run ``tools/check_env.py`` as a program with ``args``; return the CompletedProcess."""
    return subprocess.run([sys.executable, "-W", "ignore", os.path.join(REPO, "tools", "check_env.py"), *args],
                          check=False, capture_output=True, text=True, timeout=300, env=env)


def test_check_env_cli_runs():
    from conftest import _nvidia_opencl_platform
    gpu_flag = [] if _nvidia_opencl_platform() is not None else ["--no-gpu"]
    r = _run_check_env("--skip-pyfwi", *gpu_flag)
    assert "== Versions ==" in r.stdout and "== Result ==" in r.stdout
    assert r.returncode == 0, r.stdout[-2000:]


def test_check_env_no_gpu_mode_turns_missing_gpu_into_info():
    # Hide the GPU (affects both CUDA and NVIDIA OpenCL), as on a CI machine.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    strict = _run_check_env("--skip-pyfwi", env=env)
    assert strict.returncode == 1 and "[FAIL ] NVIDIA OpenCL present" in strict.stdout
    relaxed = _run_check_env("--skip-pyfwi", "--no-gpu", env=env)
    assert relaxed.returncode == 0, relaxed.stdout[-2000:]
    assert "[INFO ] NVIDIA OpenCL missing (--no-gpu)" in relaxed.stdout


def _kernel_sources(order):
    """Kernel sources generated by ``kernel_caller`` of 0.1.10, without an OpenCL context.

    ``WavePreparation.kernel_caller`` only reads attributes of ``self``, so a
    ``SimpleNamespace`` with plausible values for a 100x100 model + 20-point PML
    (140x140 grid), 4 shots and 100 surface receivers is enough.

    Returns the tuple ``(elastic.cl, elastic_crosswell.cl, elastic_surface.cl)`` sources.
    """
    from PyFWI.wave_propagation import WavePreparation
    fake = SimpleNamespace(sdo=order // 2,          # PyFWI stores half of the order
                           tnz=140, tnx=140,        # grid size including the PML
                           ns=4, nr=100,            # sources and receivers
                           dt=0.004 / 7, dh=7.0,    # internal time step (s) and grid spacing (m)
                           npml=20, n_surface_rec=100, n_well_rec=0, src_cts=25,
                           rec_top_left_const=0, rec_top_left_var=0, rec_top_right_const=0,
                           rec_top_right_var=0, rec_surface_const=22, rec_surface_var=0)
    return WavePreparation.kernel_caller(fake)


@pytest.mark.gpu
@pytest.mark.parametrize("order", [4, 8])
def test_opencl_kernels_compile_as_cuda_and_legacy_uses_fp64(cuda_device, order):
    main, crosswell, surface = _kernel_sources(order)
    counts = check_env.count_fp64(check_env.compile_ptx(main))
    hot = ["update_velx", "update_velz", "update_tauz", "update_tauxz",
           "Adj_update_tau", "Adj_update_v", "Grad"]
    assert set(hot) <= set(counts)
    # Characterises the legacy numerics: literals without the f suffix make the hot kernels use FP64.
    assert all(counts[k] > 0 for k in hot), counts
    for src in (crosswell, surface):
        assert {"injSrc", "Adj_injSrc"} <= set(check_env.count_fp64(check_env.compile_ptx(src)))


@pytest.mark.opencl
def test_pyfwi_smoke_on_nvidia_opencl_is_finite_and_deterministic(nvidia_opencl_platform):
    a, _, _ = check_env.pyfwi_case(nvidia_opencl_platform, 4)
    b, _, _ = check_env.pyfwi_case(nvidia_opencl_platform, 4)
    assert check_env.outputs_ok(a)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


#: The environments of tools/setup_envs.sh, next to the repository.
ENVS = os.path.join(os.path.dirname(REPO), "envs")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ref-0.1.10", "dev", "cuda12"])
def test_check_env_passes_in_every_environment(nvidia_opencl_platform, name):
    # Roadmap steps 0.1 and 2.6: the full check (with the PyFWI smoke run) in each environment.
    python = os.path.join(ENVS, name, "bin", "python")
    if not os.path.exists(python):
        pytest.skip(f"envs/{name} not created (tools/setup_envs.sh {name})")
    r = subprocess.run([python, "-W", "ignore", os.path.join(REPO, "tools", "check_env.py")], cwd=REPO,
                       check=False, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
