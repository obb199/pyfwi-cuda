"""Self-test of the NaN guard bands (``test/oob_guard.py``).

Before trusting the guard to detect out-of-bounds reads in PyFWI, check it on a
kernel whose behaviour we choose: ``out[i] = a[i + shift]`` over every index of
``a``. With ``shift = 0`` every read is in bounds and the output must equal the
input exactly; with ``shift = +1`` the last read is one element past the end, and
with ``shift = -1`` the first read is one element before the start: those two
outputs must be NaN, and only those.
"""
import numpy as np
import pytest

#: OpenCL kernel used by the self-test: each work-item copies a[i + shift] into out[i].
SHIFT_KERNEL = """
__kernel void read_shift(__global const float *a, __global float *out, int shift) {
    int i = get_global_id(0);
    out[i] = a[i + shift];
}
"""


@pytest.mark.parametrize("shift, nan_index", [(0, None), (1, -1), (-1, 0)])
def test_oob_guard_detects_reads_outside_a_buffer(opencl_platform, shift, nan_index):
    import pyopencl as cl

    from oob_guard import nan_guard_bands

    device = cl.get_platforms()[opencl_platform].get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, SHIFT_KERNEL).build()
    a = np.arange(1, 257, dtype=np.float32)          # 256 finite, non-zero values
    out = np.zeros_like(a)
    with nan_guard_bands():
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        out_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)
        program.read_shift(queue, a.shape, None, a_buf, out_buf, np.int32(shift))
        cl.enqueue_copy(queue, out, out_buf)
        queue.finish()

    expected = np.roll(a, -shift)  # a[i + shift], wrapping around (the wrapped value is the OOB one)
    if nan_index is None:
        np.testing.assert_array_equal(out, a)
    else:
        assert np.isnan(out[nan_index])
        finite = np.ones(a.size, bool)
        finite[nan_index] = False
        np.testing.assert_array_equal(out[finite], expected[finite])


def test_oob_guard_restores_pyopencl_buffer():
    import pyopencl as cl

    from oob_guard import nan_guard_bands

    original = cl.Buffer
    with nan_guard_bands():
        assert cl.Buffer is not original
    assert cl.Buffer is original
