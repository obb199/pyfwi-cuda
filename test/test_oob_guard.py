"""Self-tests of the guard bands (``test/oob_guard.py``).

Before trusting the guard to find out-of-bounds accesses in PyFWI, check it on
kernels whose behaviour we choose:

* ``read_shift``: ``out[i] = a[i + shift]``. With ``shift = 0`` every read is in
  bounds; with ``+1`` the last read is one element past the end of ``a``, with
  ``-1`` the first read is one element before its start. With NaN bands exactly
  those outputs must be NaN; with zero bands they must be 0.
* ``write_shift``: ``out[i + shift] = a[i]`` (and an accumulating variant with
  ``+=``). With ``shift = 0`` no band may change; with ``+1``/``-1`` one write
  lands in a band and ``GuardReport.touched()`` must report ``out``.
"""
import numpy as np
import pytest

#: OpenCL kernels of the self-tests. ``a`` holds 256 finite, non-zero values.
KERNELS = """
__kernel void read_shift(__global const float *a, __global float *out, int shift) {
    int i = get_global_id(0);
    out[i] = a[i + shift];
}
__kernel void write_shift(__global const float *a, __global float *out, int shift) {
    int i = get_global_id(0);
    out[i + shift] = a[i];
}
__kernel void accumulate_shift(__global const float *a, __global float *out, int shift) {
    int i = get_global_id(0);
    out[i + shift] += a[i];
}
"""


def _run(platform_index, guard, kernel, shift):
    """Run one self-test kernel on ``a = 1..256`` inside ``guard``; return (out, report).

    Parameters
    ----------
    platform_index : int
        OpenCL platform to use (first device).
    guard : callable
        ``oob_guard.nan_guard_bands`` or ``oob_guard.zero_guard_bands``.
    kernel : str
        Name of a kernel of ``KERNELS``.
    shift : int
        Index shift passed to the kernel.
    """
    import pyopencl as cl

    device = cl.get_platforms()[platform_index].get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, KERNELS).build()
    a = np.arange(1, 257, dtype=np.float32)
    out = np.zeros_like(a)
    with guard() as report:
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        out_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)
        getattr(program, kernel)(queue, a.shape, None, a_buf, out_buf, np.int32(shift))
        cl.enqueue_copy(queue, out, out_buf)
        queue.finish()
    return a, out, report


@pytest.mark.parametrize("fill", [np.nan, 0.0])
@pytest.mark.parametrize("shift, oob_index", [(0, None), (1, -1), (-1, 0)])
def test_guard_gives_out_of_bounds_reads_the_fill_value(opencl_platform, fill, shift, oob_index):
    from oob_guard import nan_guard_bands, zero_guard_bands

    guard = nan_guard_bands if np.isnan(fill) else zero_guard_bands
    a, out, report = _run(opencl_platform, guard, "read_shift", shift)
    expected = np.roll(a, -shift)  # a[i + shift]; the wrapped-around element is the out-of-bounds one
    if oob_index is not None:
        expected[oob_index] = fill
    np.testing.assert_array_equal(out, expected)
    assert report.touched() == []  # reading never changes a band


@pytest.mark.parametrize("kernel", ["write_shift", "accumulate_shift"])
@pytest.mark.parametrize("shift", [0, 1, -1])
def test_guard_reports_out_of_bounds_writes(opencl_platform, kernel, shift):
    from oob_guard import zero_guard_bands

    _, _, report = _run(opencl_platform, zero_guard_bands, kernel, shift)
    touched = report.touched()
    if shift == 0:
        assert touched == []
    else:
        assert [b.shape for b in touched] == [(256,)] and touched[0] is report.buffers[1]  # the "out" buffer


def test_guard_restores_pyopencl_buffer():
    import pyopencl as cl

    from oob_guard import guard_bands

    original = cl.Buffer
    with guard_bands():
        assert cl.Buffer is not original
    assert cl.Buffer is original
