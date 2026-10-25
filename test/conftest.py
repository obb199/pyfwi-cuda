"""Shared pytest configuration and fixtures.

pytest loads this file automatically before the tests in ``test/``. It:

* puts the repository root on ``sys.path`` so tests can ``import tools.*``;
* provides fixtures that **skip** a test when the hardware it needs is absent,
  so the same suite runs on the RTX machine and on GPU-less CI runners.

Markers (declared in ``pyproject.toml``): ``gpu`` needs CuPy + an NVIDIA GPU +
cuda.core; ``opencl`` needs the NVIDIA OpenCL platform.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

#: Absolute path of the repository root (the parent of ``test/``).
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)


def _nvidia_opencl_platform():
    """Return the index of the NVIDIA OpenCL platform, or None if there is none.

    The index is what ``PYOPENCL_CTX`` expects. Any error (PyOpenCL missing, no
    ICD loader, no platforms) means "no NVIDIA platform".

    Used by the ``nvidia_opencl_platform`` fixture and by
    ``test_check_env.py::test_check_env_cli_runs``.
    """
    try:
        import pyopencl as cl
        for i, p in enumerate(cl.get_platforms()):
            if "NVIDIA" in p.name:
                return i
    except Exception:  # noqa: BLE001
        return None
    return None


def _opencl_platforms():
    """Return ``[(index, name, number_of_devices), ...]`` for every OpenCL platform (empty on error)."""
    try:
        import pyopencl as cl
        return [(i, p.name, len(p.get_devices())) for i, p in enumerate(cl.get_platforms())]
    except Exception:  # noqa: BLE001
        return []


def _pyopencl_ctx(index, n_devices):
    """Value of ``PYOPENCL_CTX`` that selects the first device of platform ``index``.

    With a single device, PyOpenCL does not consume a device answer, so only the
    platform index may be given ("left over" error otherwise, the 0.1.10 bug).
    """
    return f"{index}" if n_devices == 1 else f"{index}:0"


def _has_cuda():
    """Return True if CuPy is installed and sees at least one GPU. Used by ``cuda_device``."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def nvidia_opencl_platform():
    """Index of the NVIDIA OpenCL platform; skips the test when there is none."""
    idx = _nvidia_opencl_platform()
    if idx is None:
        pytest.skip("no NVIDIA OpenCL platform")
    return idx


@pytest.fixture
def opencl_platform(monkeypatch):
    """Select an OpenCL device for PyFWI and return its platform index; skip if there is none.

    The NVIDIA platform is preferred (fast); otherwise the first platform, which
    on a CI machine is POCL (OpenCL on the CPU). The choice is passed to PyFWI
    through ``PYOPENCL_CTX``, set with ``monkeypatch`` so it is restored after the test.

    Used by the device tests of ``test_known_bugs.py`` and ``test_oob_guard.py``.
    """
    platforms = _opencl_platforms()
    if not platforms:
        pytest.skip("no OpenCL platform")
    chosen = next((p for p in platforms if "NVIDIA" in p[1]), platforms[0])
    monkeypatch.setenv("PYOPENCL_CTX", _pyopencl_ctx(chosen[0], chosen[2]))
    return chosen[0]


@pytest.fixture
def single_device_platform(monkeypatch):
    """Index of an OpenCL platform with exactly one device; skip if there is none.

    ``PYOPENCL_CTX`` is removed for the test (and restored afterwards), so that
    PyFWI's own device selection is what gets tested.
    """
    platform = next((p for p in _opencl_platforms() if p[2] == 1), None)
    if platform is None:
        pytest.skip("no OpenCL platform with a single device")
    monkeypatch.delenv("PYOPENCL_CTX", raising=False)
    return platform[0]


@pytest.fixture(scope="session")
def cuda_device():
    """CUDA device ordinal (always 0); skips the test without CuPy, a GPU or cuda.core."""
    if not _has_cuda():
        pytest.skip("no CuPy or no NVIDIA GPU")
    try:
        import cuda.core  # noqa: F401
    except ImportError:
        pytest.skip("no cuda.core")
    return 0


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory):
    """Wheel and source distribution of this repository, built once per test session.

    They are built from a copy of the files a build reads (``pyproject.toml``, ``README.md``,
    ``LICENSE`` and ``src/``), so that the build artefacts (``build/``, ``*.egg-info``) never touch
    the working tree. The build uses the setuptools of the running environment, without network.

    Returns ``{"wheel": path, "sdist": path}`` (``pathlib.Path``). Skips the test when setuptools
    is not installed (the reference environment has none).

    Used by ``test_packaging.py`` and ``test_known_bugs.py::test_wheel_contains_torch_and_data``.
    """
    pytest.importorskip("setuptools", reason="setuptools is needed to build the wheel")
    source = tmp_path_factory.mktemp("source")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(os.path.join(REPO, name), source)
    shutil.copytree(os.path.join(REPO, "src"), source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    build = ("import json; from setuptools import build_meta as b; "
             "json.dump({'wheel': b.build_wheel('dist'), 'sdist': b.build_sdist('dist')}, open('built.json', 'w'))")
    subprocess.run([sys.executable, "-c", build], cwd=source, check=True, capture_output=True, text=True)
    with open(source / "built.json") as f:
        return {kind: source / "dist" / name for kind, name in json.load(f).items()}
