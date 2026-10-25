"""Shared pytest configuration and fixtures.

pytest loads this file automatically before the tests in ``test/``. It:

* puts the repository root on ``sys.path`` so tests can ``import tools.*``;
* provides fixtures that **skip** a test when the hardware it needs is absent,
  so the same suite runs on the RTX machine and on GPU-less CI runners.

Markers (declared in ``pyproject.toml``): ``gpu`` needs CuPy + an NVIDIA GPU +
cuda.core; ``opencl`` needs the NVIDIA OpenCL platform.
"""
import os
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
