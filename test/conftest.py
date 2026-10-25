import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)  # para importar tools.*


def _nvidia_opencl_platform():
    try:
        import pyopencl as cl
        for i, p in enumerate(cl.get_platforms()):
            if "NVIDIA" in p.name:
                return i
    except Exception:  # noqa: BLE001
        return None
    return None


def _has_cuda():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def nvidia_opencl_platform():
    idx = _nvidia_opencl_platform()
    if idx is None:
        pytest.skip("sem plataforma OpenCL NVIDIA")
    return idx


@pytest.fixture(scope="session")
def cuda_device():
    if not _has_cuda():
        pytest.skip("sem CuPy ou GPU NVIDIA")
    try:
        import cuda.core  # noqa: F401
    except ImportError:
        pytest.skip("sem cuda.core")
    return 0
