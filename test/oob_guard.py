"""NaN guard bands around OpenCL buffers: a deterministic detector of out-of-bounds reads.

The problem
-----------
A kernel that reads past the end (or before the start) of a buffer has undefined
behaviour: it reads whatever happens to be next to the buffer in device memory.
On the NVIDIA driver that is often a harmless finite value; on POCL it is often
garbage that becomes NaN. A test that just runs the kernel is therefore flaky.

The trick
---------
While ``nan_guard_bands()`` is active, every ``pyopencl.Buffer`` created from a
host array is allocated as one larger buffer laid out as

    [ guard band: NaN | the data, copied from the host array | guard band: NaN ]

and the caller receives a *sub-buffer* that covers only the data. Kernels see a
normal buffer, but any read just outside it returns NaN, and NaN propagates
through the arithmetic to the outputs. "Outputs are finite" then becomes a
deterministic, device-independent check that no out-of-bounds read happened
(within ``pad_bytes`` of the buffer).

It works because PyFWI 0.1.10 creates every buffer as
``cl.Buffer(ctx, flags, hostbuf=array)`` through the ``pyopencl`` module
attribute, which this context manager temporarily replaces.

Used by
-------
``test/test_known_bugs.py`` (out-of-bounds reads of the stencils) and its own
self-test ``test_oob_guard_detects_reads_outside_a_buffer``. Later steps of the
roadmap (1.7 padded buffers, 2.4 ``fixed`` mode) use it to prove the fix.
"""
import contextlib

import numpy as np
import pyopencl as cl


@contextlib.contextmanager
def nan_guard_bands(pad_bytes=1 << 16):
    """Context manager: buffers created inside it get NaN guard bands.

    Parameters
    ----------
    pad_bytes : int, optional
        Size of each guard band in bytes (rounded up to the device alignment for
        the band before the data). The default, 64 KiB, covers several rows of
        any grid used in the tests (a 140-point row of float32 is 560 bytes).

    Yields
    ------
    list
        The parent buffers created so far. They are kept alive here because a
        sub-buffer must not outlive its parent; tests do not need to use it.

    Notes
    -----
    Only buffers created with ``hostbuf`` are guarded; others are created
    normally. Buffers keep their contents and sizes, so results of code without
    out-of-bounds reads are unchanged bit for bit.
    """
    original = cl.Buffer
    parents = []  # keeps the large parent buffers alive while their sub-buffers are in use
    queues = {}   # one command queue per context, used to fill the parent buffers

    def guarded_buffer(ctx, flags, size=0, hostbuf=None):
        if hostbuf is None:
            return original(ctx, flags, size)
        host = np.ascontiguousarray(hostbuf)
        nbytes = host.nbytes
        # Sub-buffers must start at a multiple of the device base-address alignment
        # (reported in bits by OpenCL).
        align = max(ctx.devices[0].mem_base_addr_align // 8, 4)
        head = -(-pad_bytes // align) * align          # guard band before the data, rounded up
        total = head + nbytes + pad_bytes
        total += -total % 4                            # whole number of float32 values
        parent = original(ctx, cl.mem_flags.READ_WRITE, total)
        queue = queues.setdefault(ctx.int_ptr, cl.CommandQueue(ctx))
        cl.enqueue_copy(queue, parent, np.full(total // 4, np.nan, np.float32))   # everything NaN
        cl.enqueue_copy(queue, parent, host, dst_offset=head)                     # then the data
        queue.finish()
        parents.append(parent)
        return parent.get_sub_region(head, nbytes)

    cl.Buffer = guarded_buffer
    try:
        yield parents
    finally:
        cl.Buffer = original
