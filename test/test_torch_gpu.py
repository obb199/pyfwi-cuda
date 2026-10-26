"""``PyFWI.torchfwi`` with tensors on the GPU (roadmap step 4.5), test type INT.

``torchfwi.Fwi`` takes the model as CPU or CUDA tensors and returns the seismograms and the gradient on
the device of the model; CUDA tensors are exchanged with CuPy through DLPack, which shares the memory (the
same data pointer) instead of copying it. These tests need PyTorch built for CUDA: they run in
``envs/torch`` (``tools/setup_envs.sh torch``), where the slow test at the end starts them from the other
environments, whose PyTorch is CPU-only.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import running_reference

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no GPU tensors")]
needs_cuda_torch = pytest.mark.skipif(not torch.cuda.is_available(),
                                      reason="PyTorch without CUDA (run in envs/torch: tools/setup_envs.sh torch)")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TORCH_PYTHON = os.path.join(os.path.dirname(REPO), "envs", "torch", "bin", "python")
sys.path.insert(0, os.path.join(REPO, "example", "scripts"))


@needs_cuda_torch
def test_dlpack_shares_the_memory(cuda_device):
    # The exchanges with CuPy do not copy: a CUDA tensor and the CuPy array made from it (and back) have
    # the same data pointer; the values reach NumPy unchanged.
    import cupy

    from PyFWI.torchfwi import _like, _to_numpy
    tensor = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    assert cupy.from_dlpack(tensor).data.ptr == tensor.data_ptr()
    np.testing.assert_array_equal(_to_numpy(tensor), np.arange(12, dtype=np.float32).reshape(3, 4))
    values = np.linspace(0, 1, 6, dtype=np.float32)
    back = _like(values, tensor)
    assert back.is_cuda and cupy.from_dlpack(back).data.ptr == back.data_ptr()
    np.testing.assert_array_equal(back.cpu().numpy(), values)


@needs_cuda_torch
def test_gpu_tensors_give_the_results_of_cpu_tensors(cuda_device, monkeypatch):
    # The same seismograms and gradients, bit for bit, from CPU and CUDA tensors; the outputs on the
    # device of the model.
    import torch_gradient

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    cpu, gpu = torch_gradient.main(size="small"), torch_gradient.main(size="small", device="cuda")
    assert cpu["device"] == "cpu" and gpu["device"].startswith("cuda")
    assert gpu["loss"] == pytest.approx(cpu["loss"], rel=1e-6)  # PyTorch sums in another order on the GPU
    for key in ("vp", "vs", "rho"):
        np.testing.assert_array_equal(cpu["torch_gradient"][key], gpu["torch_gradient"][key], err_msg=key)
        np.testing.assert_array_equal(gpu["torch_gradient"][key], gpu["pyfwi_gradient"][key], err_msg=key)


@needs_cuda_torch
def test_the_gradient_against_finite_differences(cuda_device, monkeypatch):
    # The vp gradient of a PyTorch loss through Fwi, with CUDA tensors, against a centred finite
    # difference of the loss along a smooth perturbation (the fixed mode is within 1 % of finite
    # differences for vp: test_math.py). As there, a corner cell above every velocity pins the absorbing
    # layer, which follows max(vp): otherwise the finite difference measures the layer moving too.
    import torch_gradient

    import PyFWI.acquisition as acq
    import PyFWI.wave_propagation as wave
    from PyFWI.torchfwi import Fwi

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    p = torch_gradient.SIZES["small"]
    true, start = torch_gradient.make_models(p["nz"], p["nx"])
    for model in (true, start):
        model["vp"][0, 0] = 3200.0  # pins the absorbing layer (see above)
    inpa = {"ns": p["ns"], "sdo": 4, "fdom": 15, "dh": p["dh"], "dt": 0.004, "acq_type": 1, "t": p["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": p["dh"]}
    src_loc, rec_loc, _, nw = acq.acq_parameters(p["ns"], p["dh"], p["dh"] * p["nx"], p["dh"] * p["nz"], p["dh"], 4,
                                                 acq_type=1)
    src_loc[:, 1] -= p["dh"]
    src = acq.Source(src_loc, p["dh"], inpa["dt"])
    src.Ricker(inpa["fdom"])
    W = wave.WavePropagator(inpa, src, rec_loc, (p["nz"], p["nx"]), components=0, n_well_rec=nw, chpr=100)
    with torch.no_grad():
        observed = Fwi.apply(W, *(torch.tensor(true[k], device="cuda") for k in ("vp", "vs", "rho")))

    def loss(vp):
        taux, tauz = Fwi.apply(W, vp, *(torch.tensor(start[k], device="cuda") for k in ("vs", "rho")))
        return 0.5 * torch.sum((taux - observed[0]) ** 2) + 0.5 * torch.sum((tauz - observed[1]) ** 2)

    vp = torch.tensor(start["vp"], device="cuda", requires_grad=True)
    loss(vp).backward()
    # A smooth bump inside the model, as in tools/math_baseline.py: away from the edges (the absorbing layer
    # copies the edge values, expand_model) and from the surface (PyFWI masks the gradient near the sources
    # and receivers). Measured: within 0.5 % for steps of 0.5 to 8 m/s.
    z, x = np.mgrid[:p["nz"], :p["nx"]]
    direction = np.exp(-((z - 24) ** 2 + (x - p["nx"] / 2) ** 2) / (2 * 4.0 ** 2)).astype(np.float32)
    direction[:12], direction[36:], direction[:, :8], direction[:, p["nx"] - 8:] = 0, 0, 0, 0
    step = 2.0  # m/s
    with torch.no_grad():
        plus = loss(torch.tensor(start["vp"] + step * direction, device="cuda"))
        minus = loss(torch.tensor(start["vp"] - step * direction, device="cuda"))
    finite = float(plus - minus) / (2 * step)
    analytic = float(np.sum(vp.grad.cpu().numpy() * direction))
    assert analytic == pytest.approx(finite, rel=0.02)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(TORCH_PYTHON), reason="envs/torch not created (tools/setup_envs.sh torch)")
def test_the_gpu_tensor_tests_pass_in_the_torch_environment(cuda_device):
    # The tests above, and the example scripts with CPU tensors, in envs/torch (PyTorch for CUDA 13).
    program = "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
    check = subprocess.run([TORCH_PYTHON, "-c", program], capture_output=True, text=True, check=True)
    assert check.stdout.split()[0] == "True", check.stdout
    run = subprocess.run([TORCH_PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow",
                          os.path.join("test", "test_torch_gpu.py"), os.path.join("test", "test_examples.py"),
                          "-k", "torch or dlpack or gpu_tensors or finite"],
                         cwd=REPO, capture_output=True, text=True, timeout=1800, check=False)
    assert run.returncode == 0, run.stdout[-3000:]
    assert " passed" in run.stdout and "skipped" not in run.stdout, run.stdout[-1000:]
