"""The CUDA backend (roadmap step 2.3): ``PyFWI.backends.cuda.CUDABackend``.

The same checks as ``test_backends.py`` for OpenCL, with a kernel written in PyFWI's OpenCL C
(``__kernel``, ``__global``, ``get_global_id``) that the backend compiles as CUDA, plus what is
specific to CUDA: the bounds guard of ``portable_source`` (the grid of blocks is rounded up), and
loading a module compiled by an NVRTC newer than the driver.
"""
import numpy as np
import pytest

from helpers import running_reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no backends")]

#: A kernel in PyFWI's style: global size (Nz, Nx), the row from dimension 0, the column from 1.
SOURCE = """
#define Nz 5
#define Nx 37
__kernel void mark(__global float *a, __global float *b, float offset, __global float *before) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  b[i * Nx + j] = 2.0f * a[i * Nx + j] + offset;
  if (i == 0 && j == 0) before[0] = a[-1];
}
"""


@pytest.fixture
def backend(cuda_device):
    from PyFWI.backends.cuda import CUDABackend
    return CUDABackend(cuda_device)


def test_portable_source_guards_every_kernel():
    from PyFWI.backends.cuda import GUARD, portable_source
    two = SOURCE + SOURCE.replace("mark", "mark2").replace("#define Nz 5\n#define Nx 37\n", "")
    cuda = portable_source(two)
    assert cuda.count(GUARD.strip()) == 2 and '#define __kernel extern "C" __global__' in cuda
    with pytest.raises(ValueError, match="cannot guard"):
        portable_source("__kernel void k(__global float *a) { a[get_global_id(0)] = 0; }")


def test_allocate_puts_the_data_between_margins(backend):
    host = np.arange(10, dtype=np.float32)
    buffer, allocation = backend.allocate(host, 512, fill=-1.0)
    assert backend.nbytes(buffer) == 40 and backend.nbytes(allocation) == 40 + 1024
    whole = np.empty(allocation.size, np.float32)
    backend.to_host(whole, allocation)
    np.testing.assert_array_equal(whole[128:138], host)
    assert (whole[:128] == -1).all() and (whole[138:] == -1).all()


def test_copies(backend):
    buffer = backend.allocate_empty(64)
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    backend.to_device(buffer, data)
    out = np.zeros((4, 4), np.float32)
    backend.wait([backend.to_host(out, buffer, blocking=False)])
    np.testing.assert_array_equal(out, data)
    target = backend.allocate_empty(16)
    backend.copy(target, buffer, 16, offset=8 * 4)  # values 8 to 11
    small = np.zeros(4, np.float32)
    backend.to_host(small, target)
    np.testing.assert_array_equal(small, [8, 9, 10, 11])


def test_kernels_run_exactly_on_the_opencl_work_items(backend):
    # Nz x Nx = 5 x 37 is not a multiple of the 8 x 32 blocks: the guard must stop the threads
    # beyond it, or they would write past the end of b (into its margin, filled with -7).
    program = backend.build(SOURCE)
    kernel = backend.kernel(program, "mark")
    a, _ = backend.allocate(np.arange(5 * 37, dtype=np.float32), 512, fill=3.0)
    b, b_allocation = backend.allocate(np.zeros(5 * 37, np.float32), 512, fill=-7.0)
    before = backend.allocate_empty(4)
    backend.set_args(kernel, a, b, np.float32(0), before)
    backend.set_arg(kernel, 2, np.float32(1))
    backend.launch(kernel, (5, 37))
    backend.finish()
    whole = np.empty(b_allocation.size, np.float32)
    backend.to_host(whole, b_allocation)
    np.testing.assert_array_equal(whole[128:128 + 185], 2 * np.arange(185) + 1)
    assert (whole[:128] == -7).all() and (whole[128 + 185:] == -7).all()
    margin = np.zeros(1, np.float32)
    backend.to_host(margin, before)
    assert margin[0] == 3.0  # a read before the buffer sees the margin
    backend.call(kernel, (5, 37), a, b, np.float32(-5), before)
    backend.to_host(whole, b_allocation)
    np.testing.assert_array_equal(whole[128:128 + 185], 2 * np.arange(185) - 5)


def test_the_kernels_of_pyfwi_compile_and_load(backend):
    # The NVRTC installed by pip may be newer than the driver (13.4 against 13.0 on the RTX
    # machine): PTX would not load, the cubin that CuPy builds for the GPU does.
    from cupy.cuda import nvrtc, runtime

    from helpers import tiny_case
    sources = tiny_case(acq_type=2, inpa_extra={"backend": "opencl"}).W.kernel_caller()
    for source in sources:
        program = backend.build(source)
        assert backend.kernel(program, "injSrc").function_name == "injSrc"
    major, minor = nvrtc.getVersion()
    print(f"NVRTC {major}.{minor}, driver CUDA {runtime.driverGetVersion()}")
    assert backend.description.endswith("(CUDA device 0)") and backend.architecture.startswith("sm_")
