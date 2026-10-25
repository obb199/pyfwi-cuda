"""Testes do verificador de ambiente (tools/check_env.py) e dos arquivos de ambiente."""
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import check_env

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
    # .reg .f64 é declaração, não instrução; as três instruções com .f64 contam.
    assert check_env.count_fp64(PTX) == {"k_double": 3, "k_float": 0}


def test_count_fp64_empty_ptx():
    assert check_env.count_fp64("") == {}


def test_check_known_issue_is_warning_not_failure(capsys):
    check_env.FAILURES.clear()
    assert check_env.check("passa", True)
    assert not check_env.check("conhecido", False, known_issue="bug antigo")
    assert check_env.FAILURES == []
    assert not check_env.check("quebra", False)
    assert check_env.FAILURES == ["quebra"]
    out = capsys.readouterr().out
    assert "[AVISO] conhecido: bug antigo" in out and "[FALHA] quebra" in out
    check_env.FAILURES.clear()


def test_test_model_is_elastic_and_float32():
    m = check_env.test_model(40, 50)
    assert set(m) == {"vp", "vs", "rho"}
    assert all(v.shape == (40, 50) and v.dtype == np.float32 for v in m.values())
    assert set(np.unique(m["vp"])) == {2500.0, 3000.0}
    np.testing.assert_allclose(m["vp"] / m["vs"], np.sqrt(3.0), rtol=1e-6)
    smooth = check_env.test_model(40, 50, smooth=1)["vp"]
    assert 2500.0 <= smooth.min() and smooth.max() <= 3000.0 and len(np.unique(smooth)) > 2


def test_outputs_ok():
    good = {"d_taux": np.ones(3), "d_vx": np.zeros(3), "g_vp": np.ones(2)}
    assert check_env.outputs_ok(good)
    assert not check_env.outputs_ok({**good, "d_taux": np.array([1.0, np.nan, 0.0])})
    assert not check_env.outputs_ok({**good, "d_taux": np.zeros(3)})
    assert not check_env.outputs_ok({**good, "g_vp": np.zeros(2)})


def _pins(name):
    pins = {}
    with open(os.path.join(REPO, "requirements", name)) as f:
        for line in f:
            m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line.strip())
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def test_reference_env_matches_original_requirements():
    ref = _pins("ref-0.1.10.txt")
    # Versões do requirements.txt da 0.1.10: a referência não pode depender de correções.
    for pkg, ver in {"numpy": "1.21.6", "scipy": "1.7.3", "matplotlib": "3.5.3",
                     "pyopencl": "2022.2.4", "segyio": "1.9.11"}.items():
        assert ref[pkg] == ver, pkg
    assert ref["torch"].startswith("1.13.1")
    assert "pyfwi" not in ref  # instalado à parte, do PyPI, sem dependências


def test_dev_env_has_cuda13_stack():
    dev = _pins("dev.txt")
    assert dev["cupy-cuda13x"].startswith("14.")
    assert dev["cuda-core"] and dev["pyopencl"] and dev["pytest-benchmark"]
    assert "cupy-cuda12x" not in dev


def test_both_envs_pin_same_pocl():
    # Comparações bit a bit no OpenCL de CPU exigem o mesmo compilador POCL nos dois lados.
    assert _pins("ref-0.1.10.txt")["pocl-binary-distribution"] == _pins("dev.txt")["pocl-binary-distribution"]


def test_setup_envs_script_is_valid_bash():
    script = os.path.join(REPO, "tools", "setup_envs.sh")
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", script], check=True)
    r = subprocess.run(["bash", script, "nada"], check=False, capture_output=True, text=True)
    assert r.returncode == 2 and "uso:" in r.stderr


def _run_check_env(*args, env=None):
    return subprocess.run([sys.executable, "-W", "ignore", os.path.join(REPO, "tools", "check_env.py"), *args],
                          check=False, capture_output=True, text=True, timeout=300, env=env)


def test_check_env_cli_runs():
    from conftest import _nvidia_opencl_platform
    gpu_flag = [] if _nvidia_opencl_platform() is not None else ["--no-gpu"]
    r = _run_check_env("--skip-pyfwi", *gpu_flag)
    assert "== Versões ==" in r.stdout and "== Resultado ==" in r.stdout
    assert r.returncode == 0, r.stdout[-2000:]


def test_check_env_no_gpu_mode_turns_missing_gpu_into_info():
    # Esconde a GPU (vale para o CUDA e para o OpenCL da NVIDIA), como numa máquina de CI.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    strict = _run_check_env("--skip-pyfwi", env=env)
    assert strict.returncode == 1 and "[FALHA] OpenCL NVIDIA presente" in strict.stdout
    relaxed = _run_check_env("--skip-pyfwi", "--no-gpu", env=env)
    assert relaxed.returncode == 0, relaxed.stdout[-2000:]
    assert "[INFO ] OpenCL NVIDIA ausente (--no-gpu)" in relaxed.stdout


def _kernel_sources(order):
    """Fontes geradas pelo kernel_caller da 0.1.10, sem precisar de contexto OpenCL."""
    from PyFWI.wave_propagation import WavePreparation
    fake = SimpleNamespace(sdo=order // 2, tnz=140, tnx=140, ns=4, nr=100, dt=0.004 / 7, dh=7.0,
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
    # Caracteriza o modo legacy: literais sem sufixo f geram FP64 nos kernels quentes.
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
