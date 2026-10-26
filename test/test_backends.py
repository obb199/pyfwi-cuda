"""The OpenCL backend (roadmap step 2.1): ``PyFWI.backends.opencl.OpenCLBackend``.

Each method of the interface of ``PyFWI.backends`` is checked on its own, on the OpenCL device of
the session (NVIDIA when there is one, POCL otherwise), with a small kernel written here.
"""
import numpy as np
import pytest

from helpers import running_reference

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no backends")

#: A kernel that writes 2 * a[i] + offset into b[i] and reads one value before a (its margin).
SOURCE = """
__kernel void twice(__global float *a, __global float *b, float offset, __global float *before) {
    int i = get_global_id(0);
    b[i] = 2.0f * a[i] + offset;
    if (i == 0) before[0] = a[-1];
}
"""


@pytest.fixture
def backend(opencl_platform):
    from PyFWI.backends.opencl import OpenCLBackend
    return OpenCLBackend()


def test_allocate_puts_the_data_between_margins(backend):
    host = np.arange(10, dtype=np.float32)
    margin = backend.alignment * 2
    buffer, allocation = backend.allocate(host, margin, fill=-1.0)
    assert backend.nbytes(buffer) == host.nbytes and backend.nbytes(allocation) == host.nbytes + 2 * margin
    whole = np.empty(backend.nbytes(allocation) // 4, np.float32)
    backend.to_host(whole, allocation)
    np.testing.assert_array_equal(whole[margin // 4:margin // 4 + 10], host)
    assert (whole[:margin // 4] == -1).all() and (whole[margin // 4 + 10:] == -1).all()


def test_copies_to_and_from_the_device_blocking_or_not(backend):
    buffer = backend.allocate_empty(64)
    data = np.arange(16, dtype=np.float32)
    backend.to_device(buffer, data)
    out = np.zeros(16, np.float32)
    backend.to_host(out, buffer)
    np.testing.assert_array_equal(out, data)
    backend.to_device(buffer, data * 3, blocking=False)
    events = [backend.to_host(out, buffer, blocking=False)]
    backend.wait(events)
    np.testing.assert_array_equal(out, data * 3)


def test_copy_on_the_device_from_an_offset(backend):
    source, target = backend.allocate_empty(64), backend.allocate_empty(16)
    backend.to_device(source, np.arange(16, dtype=np.float32))
    backend.copy(target, source, 16, offset=8 * 4)  # values 8 to 11
    out = np.zeros(4, np.float32)
    backend.to_host(out, target)
    np.testing.assert_array_equal(out, [8, 9, 10, 11])


def test_kernels_run_with_arguments_set_before_or_at_the_call(backend):
    program = backend.build(SOURCE)
    kernel = backend.kernel(program, "twice")
    a, _ = backend.allocate(np.arange(8, dtype=np.float32), backend.alignment, fill=7.0)
    b, before = backend.allocate_empty(32), backend.allocate_empty(4)
    backend.set_args(kernel, a, b, np.float32(0), before)
    backend.set_arg(kernel, 2, np.float32(1))  # change one argument only
    backend.launch(kernel, (8,))
    backend.finish()
    out, margin = np.zeros(8, np.float32), np.zeros(1, np.float32)
    backend.to_host(out, b)
    backend.to_host(margin, before)
    np.testing.assert_array_equal(out, 2 * np.arange(8) + 1)
    assert margin[0] == 7.0  # the read before the buffer sees the margin
    backend.call(kernel, (8,), a, b, np.float32(-5), before)
    backend.to_host(out, b)
    np.testing.assert_array_equal(out, 2 * np.arange(8) - 5)


def test_backends_on_the_same_device_share_the_context(backend):
    from PyFWI.backends.opencl import OpenCLBackend
    other = OpenCLBackend()
    assert other.context is backend.context and other.queue is not backend.queue
    assert backend.name == "opencl" and backend.alignment % 4 == 0
