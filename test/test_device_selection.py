"""Choice of the OpenCL device (roadmap step 1.2): ``PyFWI._opencl_device.choose_devices``.

PyFWI 0.1.10 let ``pyopencl.create_some_context()`` choose the device. The fork chooses it itself,
with the same rules, so that the same ``inpa`` and the same ``PYOPENCL_CTX`` give the same device,
but without PyOpenCL's failure on single-device platforms and without ever asking the user.

* Unit tests use *simulated* platforms (no OpenCL needed): one platform with one device, one with
  two devices, and edge cases (no platform, a platform without devices).
* ``test_same_choice_as_pyopencl`` runs PyOpenCL's own ``choose_devices`` on the same simulated
  platforms for many ``PYOPENCL_CTX`` values: wherever PyOpenCL succeeds, the choice is the same.
* The tests at the end build a real ``WavePropagator`` on every OpenCL platform of the machine
  (also in the reference environment, with the published 0.1.10, where they apply).

The unit tests are skipped in the reference environment, which has no ``PyFWI._opencl_device``.
"""
import builtins
import os
import sys

import pytest

from conftest import _opencl_platforms
from helpers import running_reference, tiny_case

#: True in the reference environment (published 0.1.10).
REFERENCE = running_reference()


class FakeDevice:
    """A simulated OpenCL device: only the ``name`` is used by the selection."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"FakeDevice({self.name!r})"


class FakePlatform:
    """A simulated OpenCL platform with the given device names."""

    def __init__(self, name, device_names):
        self.name = name
        self.devices = [FakeDevice(n) for n in device_names]

    def get_devices(self):
        return self.devices


#: A machine like the RTX one: NVIDIA with one GPU (index 0), POCL with two CPU devices (index 1).
MACHINE = [FakePlatform("NVIDIA CUDA", ["NVIDIA GeForce RTX 3060"]),
           FakePlatform("Portable Computing Language", ["pthread-AMD Ryzen 7 5700X", "pthread-Intel Xeon"])]
NVIDIA, POCL = MACHINE
RTX = NVIDIA.devices[0]
CPU0, CPU1 = POCL.devices


@pytest.fixture
def choose():
    """``choose_devices`` on the simulated ``MACHINE``, with an empty environment unless given."""
    module = pytest.importorskip("PyFWI._opencl_device", reason="the published 0.1.10 has no device selector")

    def run(platform=None, device=None, ctx=None):
        environ = {} if ctx is None else {"PYOPENCL_CTX": ctx}
        return module.choose_devices(platform, device, environ=environ, platforms=MACHINE)
    return run


def test_default_is_the_first_device_of_the_first_platform(choose):
    assert choose() == [RTX]


@pytest.mark.parametrize("ctx, expected", [
    ("", [RTX]),                    # empty: the defaults
    ("0", [RTX]),
    ("0:0", [RTX]),                 # single-device platform with a device answer: failed with PyOpenCL 2022
    ("1", [CPU0]),
    ("1:1", [CPU1]),
    ("1:0,1", [CPU0, CPU1]),        # several devices in one context, as PyOpenCL allows
    ("nvidia", [RTX]),              # part of the platform name, any case
    ("portable:xeon", [CPU1]),      # part of the device name
    ("1:pthread", [CPU0]),          # several devices match: the first one
])
def test_pyopencl_ctx(choose, ctx, expected):
    assert choose(ctx=ctx) == expected


def test_inpa_device_on_a_single_device_platform(choose):
    # The bug of 0.1.10 with PyOpenCL 2022 ("left over" device answer).
    assert choose(platform=0, device=0) == [RTX]


@pytest.mark.parametrize("platform, device, expected", [
    (1, 1, [CPU1]),
    (None, 0, [RTX]),               # inpa['device'] without inpa['platform']: platform 0, as in 0.1.10
    (1, None, [CPU0]),              # inpa['platform'] alone: device 0 (0.1.10 ignored the platform)
    ("portable", 1, [CPU1]),        # 0.1.10 also accepted part of the platform name
])
def test_inpa_keys(choose, platform, device, expected):
    assert choose(platform=platform, device=device) == expected


def test_inpa_keys_take_precedence_over_pyopencl_ctx(choose):
    assert choose(platform=0, device=0, ctx="1:1") == [RTX]


@pytest.mark.parametrize("platform, device, message", [
    (2, 0, r"platform '2' does not exist\. Available: \[0\] NVIDIA CUDA, \[1\] Portable"),
    (0, 1, r"device '1' does not exist\. Available: \[0\] NVIDIA GeForce RTX 3060\.$"),
    (1, 2, r"device '2' does not exist\. Available: \[0\] pthread-AMD Ryzen 7 5700X, \[1\] pthread-Intel Xeon\.$"),
    (1, -1, r"device '-1' does not exist"),
    ("amd", 0, r"platform 'amd' does not exist"),
])
def test_a_device_that_does_not_exist_lists_the_choices(choose, platform, device, message):
    # 0.1.10 checked inpa['device'] against the devices of platform 0 whatever the platform was.
    with pytest.raises(ValueError, match=message):
        choose(platform=platform, device=device)


def test_without_platforms_or_devices_the_error_says_what_to_install(monkeypatch):
    module = pytest.importorskip("PyFWI._opencl_device", reason="the published 0.1.10 has no device selector")
    cl = pytest.importorskip("pyopencl")
    monkeypatch.setattr(cl, "get_platforms", list)  # list() == []: a machine where PyOpenCL finds no platform
    with pytest.raises(RuntimeError, match="No OpenCL platform found.*pocl-binary-distribution"):
        module.choose_devices(environ={})
    with pytest.raises(RuntimeError, match="'Empty' has no devices"):
        module.choose_devices(environ={}, platforms=[FakePlatform("Empty", [])])


def test_never_asks_the_user(choose, monkeypatch):
    # In a terminal, create_some_context() asked "Choose platform:" when nothing was configured.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail(f"asked the user: {prompt}"))
    assert choose() == [RTX]


@pytest.mark.parametrize("ctx", ["", "0", "1", "1:0", "1:1", "1:0,1", "nvidia", "portable", "portable:xeon",
                                 "1:pthread", "computing:1", "2", "1:2", "intel", "0:0", "1:-1", "0:0:0"])
def test_same_choice_as_pyopencl(choose, monkeypatch, ctx):
    # PyOpenCL's own selection (non-interactive) on the same simulated platforms. "0:0" is the
    # single-device case: PyOpenCL 2026 accepts it (2022 did not), so it is compared here too.
    cl = pytest.importorskip("pyopencl")
    if not hasattr(cl, "choose_devices"):
        pytest.skip("this PyOpenCL has no choose_devices (older than 2024)")
    monkeypatch.setattr(cl, "get_platforms", lambda: MACHINE)
    monkeypatch.delenv("PYOPENCL_TEST", raising=False)
    try:
        expected = cl.choose_devices(interactive=False, answers=ctx.split(":"))
    except (RuntimeError, cl.Error):  # PyOpenCL's RuntimeError is its own class, not the built-in one
        with pytest.raises(ValueError):
            choose(ctx=ctx)
    else:
        assert choose(ctx=ctx) == expected


# ---------------------------------------------------------------------------------------------
# Real devices: the same calls as in 0.1.10 pick the same device
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def clean_environment(monkeypatch):
    """No ``PYOPENCL_CTX`` during the test, and the variables PyFWI writes restored afterwards."""
    for name in ("PYOPENCL_CTX", "PYOPENCL_COMPILER_OUTPUT"):
        monkeypatch.setenv(name, "placeholder")  # makes monkeypatch remember the original value
        monkeypatch.delenv(name)
    platforms = _opencl_platforms()
    if not platforms:
        pytest.skip("no OpenCL platform")
    return platforms


def _device(platform, device=0):
    """The ``pyopencl.Device`` with the given indices on this machine."""
    import pyopencl as cl
    return cl.get_platforms()[platform].get_devices()[device]


def test_real_default_device(clean_environment):
    case = tiny_case()
    assert case.W.ctx.devices == [_device(0)]


def test_real_pyopencl_ctx_selects_each_platform(clean_environment, monkeypatch):
    for index, _name, _n_devices in clean_environment:
        monkeypatch.setenv("PYOPENCL_CTX", str(index))
        assert tiny_case().W.ctx.devices == [_device(index)], f"platform {index}"


def test_real_inpa_selects_each_platform_and_is_remembered(clean_environment):
    # In the reference environment, inpa['device'] fails on single-device platforms with PyOpenCL
    # 2022 (test_known_bugs.py), so only platforms with several devices can be checked there.
    platforms = [p for p in clean_environment if not (REFERENCE and p[2] == 1)]
    if not platforms:
        pytest.skip("only single-device platforms, where 0.1.10 fails (test_known_bugs.py)")
    for index, _name, _n_devices in platforms:
        case = tiny_case(inpa_extra={"platform": index, "device": 0})
        assert case.W.ctx.devices == [_device(index)], f"platform {index}"
        # As in 0.1.10, the choice is written to PYOPENCL_CTX, so that a later WavePropagator
        # created without inpa['device'] runs on the same device.
        assert os.environ["PYOPENCL_CTX"] == f"{index}:0"
        assert tiny_case().W.ctx.devices == [_device(index)]
