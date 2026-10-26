"""Choice of the backend (roadmap step 2.2): ``PyFWI.backends.selection``.

Every branch of the choice is checked without a GPU: CuPy missing (its import blocked), no GPU, an
old driver, a broken CUDA, no CUDA backend in this PyFWI, ``cuda:strict``, and ``inpa`` pointing to
a CPU or to an NVIDIA GPU (simulated OpenCL devices, matched by PCI bus ID). The last tests use
the real machine when it has the hardware.
"""
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from helpers import running_reference, tiny_case

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no backend selection")


def nvidia_device(bus=4):
    """A simulated NVIDIA OpenCL device on PCI bus ``bus``."""
    return SimpleNamespace(name="NVIDIA GeForce RTX 3060", platform=SimpleNamespace(name="NVIDIA CUDA"),
                           pci_domain_id_nv=0, pci_bus_id_nv=bus, pci_slot_id_nv=0)


#: A simulated CPU OpenCL device (POCL), which has no PCI bus ID.
CPU = SimpleNamespace(name="pthread-AMD Ryzen 7 5700X", platform=SimpleNamespace(name="Portable Computing Language"))


def usable(*bus_ids):
    """A probe that says CUDA can run, with these CUDA devices."""
    from PyFWI.backends.selection import CudaStatus
    return lambda: CudaStatus(True, "", tuple(bus_ids))


def unusable(reason="CuPy is not installed"):
    """A probe that says CUDA cannot run."""
    from PyFWI.backends.selection import CudaStatus
    return lambda: CudaStatus(False, reason)


def never():
    raise AssertionError("CUDA was probed although OpenCL was asked")


@pytest.fixture
def select():
    """``select_backend`` with an empty environment unless one is given."""
    from PyFWI.backends.selection import select_backend

    def run(inpa=None, env=None, **kwargs):
        return select_backend(inpa or {}, environ=env or {}, **kwargs)
    return run


# ---------------------------------------------------------------------------------- what is asked

def test_precedence_inpa_then_environment_then_default():
    from PyFWI.backends.selection import requested_backend
    assert requested_backend({}, {}) == ("opencl", "default")
    assert requested_backend({}, {"PYFWI_BACKEND": " CUDA "}) == ("cuda", "PYFWI_BACKEND")
    assert requested_backend({"backend": "auto"}, {"PYFWI_BACKEND": "cuda"}) == ("auto", "inpa['backend']")
    assert requested_backend({"backend": None}, {"PYFWI_BACKEND": "cuda:strict"}) == ("cuda:strict", "PYFWI_BACKEND")


def test_an_unknown_backend_lists_the_choices(select):
    with pytest.raises(ValueError, match="Unknown backend 'metal'.*opencl, cuda, cuda:strict, auto"):
        select({"backend": "metal"})


def test_numeric_mode_precedence_and_defaults():
    # Step 2.4: inpa['numerics'], else PYFWI_NUMERICS, else the default of the backend.
    from PyFWI.backends.selection import select_numerics
    assert select_numerics({}, "opencl", {}) == "legacy" and select_numerics({}, "cuda", {}) == "fixed"
    assert select_numerics({}, "opencl", {"PYFWI_NUMERICS": " Fixed "}) == "fixed"
    assert select_numerics({"numerics": "legacy"}, "cuda", {"PYFWI_NUMERICS": "fixed"}) == "legacy"
    with pytest.raises(ValueError, match="Unknown numeric mode 'exact'; choose one of legacy, fixed"):
        select_numerics({"numerics": "exact"}, "opencl", {})
    assert select_numerics({}, "opencl") == os.environ.get("PYFWI_NUMERICS", "legacy").strip().lower()


def test_opencl_never_probes_cuda(select):
    assert select({"backend": "opencl"}, probe=never).backend == "opencl"
    assert select(probe=never).describe() == "opencl"


# ---------------------------------------------------------------------------------- the choice

def test_cuda_uses_the_gpu_of_the_chosen_opencl_device(select):
    # Two GPUs: the OpenCL device is on bus 4, which is CUDA device 1.
    selection = select({"backend": "cuda"}, device=nvidia_device(4), probe=usable("0000:01:00.0", "0000:04:00.0"))
    assert (selection.backend, selection.cuda_ordinal, selection.reason) == ("cuda", 1, "")
    assert selection.describe() == "cuda device 1 (cuda asked by inpa['backend'])"


def test_auto_follows_an_nvidia_opencl_device_only(select, caplog):
    selection = select({"backend": "auto"}, device=nvidia_device(4), probe=usable("0000:04:00.0"))
    assert (selection.backend, selection.cuda_ordinal) == ("cuda", 0)
    with caplog.at_level(logging.INFO, logger="PyFWI"):
        selection = select({"backend": "auto", "platform": 1}, device=CPU, probe=usable("0000:04:00.0"))
    assert selection.backend == "opencl" and "is not a CUDA GPU" in selection.reason
    assert caplog.records[-1].levelno == logging.INFO and "CUDA cannot run" in caplog.records[-1].getMessage()


def test_cuda_with_a_cpu_named_in_inpa_falls_back_with_a_warning(select, caplog):
    with caplog.at_level(logging.INFO, logger="PyFWI"):
        selection = select({"backend": "cuda", "platform": 1, "device": 0}, device=CPU, probe=usable("0000:04:00.0"))
    assert selection.backend == "opencl" and "pthread-AMD Ryzen" in selection.reason
    assert caplog.records[-1].levelno == logging.WARNING


def test_cuda_without_a_named_device_takes_gpu_0(select):
    # inpa names no device and the default OpenCL device is a CPU: CUDA device 0.
    selection = select({"backend": "cuda"}, device=CPU, probe=usable("0000:04:00.0", "0000:05:00.0"))
    assert (selection.backend, selection.cuda_ordinal) == ("cuda", 0)


def test_cuda_falls_back_to_opencl_when_cuda_cannot_run(select, caplog):
    with caplog.at_level(logging.WARNING, logger="PyFWI"):
        selection = select({}, env={"PYFWI_BACKEND": "cuda"}, device=nvidia_device(), probe=unusable("no CUDA GPU"))
    assert (selection.backend, selection.reason) == ("opencl", "no CUDA GPU")
    assert selection.describe() == "opencl (cuda asked by PYFWI_BACKEND; CUDA cannot run: no CUDA GPU)"
    assert "no CUDA GPU" in caplog.records[-1].getMessage()


def test_cuda_strict_fails_with_the_reason(select):
    with pytest.raises(RuntimeError, match=r"required \(inpa\['backend'\] = 'cuda:strict'\).*driver is too old"):
        select({"backend": "cuda:strict"}, device=nvidia_device(), probe=unusable("the driver is too old"))
    assert select({"backend": "cuda:strict"}, device=nvidia_device(), probe=usable("0000:04:00.0")).backend == "cuda"


# ---------------------------------------------------------------------------------- probing CUDA

def fake_cupy(count=1, driver=13000, runtime=13020, fails=False):
    """A module that looks like CuPy to ``probe_cuda``."""
    def device_count():
        if fails:
            raise RuntimeError("cudaErrorInsufficientDriver")
        return count
    return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        getDeviceCount=device_count, driverGetVersion=lambda: driver, runtimeGetVersion=lambda: runtime,
        deviceGetPCIBusId=lambda i: f"0000:0{4 + i}:00.0")))


@pytest.mark.parametrize("cupy, reason", [
    (None, "CuPy is not installed"),                                     # import blocked
    (fake_cupy(count=0), "no CUDA GPU is visible"),
    (fake_cupy(driver=12080), "driver supports CUDA 12.8, this CuPy needs CUDA 13"),
    (fake_cupy(fails=True), "CUDA does not work: RuntimeError: cudaErrorInsufficientDriver"),
])
def test_probe_says_why_cuda_cannot_run(monkeypatch, cupy, reason):
    from PyFWI.backends.selection import probe_cuda
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    status = probe_cuda()
    assert not status.usable and reason in status.reason


def test_probe_needs_a_cuda_backend_in_pyfwi(monkeypatch):
    from PyFWI.backends import selection
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy(count=2))
    monkeypatch.setattr(selection, "has_cuda_backend", lambda: False)
    status = selection.probe_cuda()
    assert not status.usable and "no CUDA backend" in status.reason
    monkeypatch.setattr(selection, "has_cuda_backend", lambda: True)
    assert selection.probe_cuda() == selection.CudaStatus(True, "", ("0000:04:00.0", "0000:05:00.0"))


def test_has_cuda_backend_looks_for_the_module():
    import PyFWI
    from PyFWI.backends.selection import has_cuda_backend
    module = os.path.join(os.path.dirname(PyFWI.__file__), "backends", "cuda.py")
    assert has_cuda_backend() == os.path.exists(module)


def test_without_opencl_auto_stays_on_opencl(select, monkeypatch):
    import PyFWI._opencl_device as opencl_device

    def no_platform(*args, **kwargs):
        raise RuntimeError("No OpenCL platform found.")
    monkeypatch.setattr(opencl_device, "choose_devices", no_platform)
    selection = select({"backend": "auto"}, probe=usable("0000:04:00.0"))
    assert selection.backend == "opencl" and selection.reason == "no OpenCL device to match with a CUDA GPU"


def test_opencl_bus_ids():
    from PyFWI.backends.selection import opencl_bus_id
    assert opencl_bus_id(nvidia_device(4)) == "0000:04:00.0"
    assert opencl_bus_id(nvidia_device(0x41)) == "0000:41:00.0"
    assert opencl_bus_id(CPU) is None and opencl_bus_id(None) is None
    no_extension = SimpleNamespace(name="NVIDIA", platform=SimpleNamespace(name="NVIDIA CUDA"))
    assert opencl_bus_id(no_extension) is None


# ---------------------------------------------------------------------------------- real machine

def test_propagators_follow_the_selection(opencl_platform, monkeypatch, caplog):
    from PyFWI.backends.selection import probe_cuda
    cuda = probe_cuda().usable  # CuPy, a GPU and a recent enough driver (the CUDA backend exists since 2.3)
    monkeypatch.setenv("PYFWI_BACKEND", "cuda")
    with caplog.at_level(logging.WARNING, logger="PyFWI"):
        W = tiny_case().W
    assert W.backend_selection.backend == W._backend.name == ("cuda" if cuda else "opencl")
    if cuda:
        assert W.ctx is None and W.queue is None  # OpenCL objects
        assert tiny_case(inpa_extra={"backend": "cuda:strict"}).W._backend.name == "cuda"
    else:  # the fallback is logged, and cuda:strict fails
        assert any("CUDA cannot run" in r.getMessage() for r in caplog.records)
        with pytest.raises(RuntimeError, match="cuda:strict"):
            tiny_case(inpa_extra={"backend": "cuda:strict"})
    with pytest.raises(ValueError, match="Unknown backend"):
        tiny_case(inpa_extra={"backend": "vulkan"})


def test_the_rtx_matches_its_cuda_device(nvidia_opencl_platform, cuda_device):
    import cupy
    import pyopencl as cl

    from PyFWI.backends.selection import opencl_bus_id, probe_cuda
    device = cl.get_platforms()[nvidia_opencl_platform].get_devices()[0]
    assert opencl_bus_id(device) == cupy.cuda.runtime.deviceGetPCIBusId(0).lower()
    assert opencl_bus_id(device) in probe_cuda().bus_ids


def test_every_branch_is_covered_without_a_gpu(tmp_path):
    # Roadmap criterion of step 2.2: every branch of the choice covered by tests that run in the
    # CPU CI, i.e. without a GPU (CUDA_VISIBLE_DEVICES empty) and without the hardware tests.
    env = {**os.environ, "COVERAGE_FILE": str(tmp_path / ".coverage"), "CUDA_VISIBLE_DEVICES": ""}
    here = os.path.dirname(__file__)
    run = subprocess.run([sys.executable, "-m", "coverage", "run", "--branch", "--source=PyFWI.backends.selection",
                          "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(here, "test_backend_selection.py"),
                          "-k", "not real_machine and not the_rtx and not propagators_follow and not every_branch"],
                         cwd=os.path.dirname(here), env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    report = subprocess.run([sys.executable, "-m", "coverage", "report", "-m", "--fail-under=100"],
                            cwd=os.path.dirname(here), env=env, capture_output=True, text=True, check=False)
    assert report.returncode == 0, report.stdout


def test_kernel_set_selection():
    # The kernels of the CUDA backend (Phase 3): native in the fixed mode, unless PYFWI_CUDA_KERNELS=parity.
    from PyFWI.backends.selection import select_cuda_kernels
    assert select_cuda_kernels("fixed", {}) == "native"
    assert select_cuda_kernels("fixed", {"PYFWI_CUDA_KERNELS": "Parity "}) == "parity"
    assert select_cuda_kernels("legacy", {}) == "parity"  # the native kernels have the fixed arithmetic
    with pytest.raises(ValueError, match="Unknown PYFWI_CUDA_KERNELS"):
        select_cuda_kernels("fixed", {"PYFWI_CUDA_KERNELS": "fast"})
