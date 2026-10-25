"""Guard bands around OpenCL buffers: deterministic detection of out-of-bounds reads and writes.

The problem
-----------
A kernel that reads past the end (or before the start) of a buffer has undefined
behaviour: it reads whatever happens to be next to the buffer in device memory.
On the NVIDIA driver that is often a harmless finite value, sometimes a crash
(``CL_OUT_OF_RESOURCES``); on POCL it is often garbage that becomes NaN. A kernel
that *writes* outside a buffer silently corrupts its neighbours. Tests that just
run the kernels are therefore flaky.

The trick
---------
While ``guard_bands()`` is active, every ``pyopencl.Buffer`` created from a host
array is allocated as one larger buffer laid out as

    [ guard band | the data, copied from the host array | guard band ]

with both bands filled with a chosen value, and the caller receives a
*sub-buffer* that covers only the data. Kernels see a normal buffer, but:

* **reads** just outside it return the fill value. With ``fill=nan`` any
  out-of-bounds read turns the outputs into NaN, so "outputs are finite" becomes
  a deterministic check. With ``fill=0`` out-of-bounds reads return zero, which
  gives PyFWI 0.1.10 a defined, reproducible behaviour;
* **writes** outside it land in a band instead of in another buffer, and
  ``GuardReport.touched()`` reports every buffer whose bands no longer hold the
  fill value. Use a finite, non-zero fill such as ``1.0`` for this: an
  accumulating write (``+=``) into a NaN band leaves it NaN, and a kernel that
  accumulates a value it read out of bounds from a *zero* band adds 0; both
  would go unnoticed. (PyFWI 0.1.10 does exactly the latter in ``Adj_injSrc``.)

It works because PyFWI 0.1.10 creates every buffer as
``cl.Buffer(ctx, flags, hostbuf=array)`` through the ``pyopencl`` module
attribute, which this context manager temporarily replaces. Code without
out-of-bounds accesses produces bit-for-bit the same results with or without it.

Used by
-------
``test/test_known_bugs.py`` (out-of-bounds reads and writes of the kernels) and
the self-tests in ``test/test_oob_guard.py``. Later steps of the roadmap (1.7
padded buffers, 2.4 ``fixed`` mode) use it to prove the fixes, and the golden
masters of Phase 0 are generated with ``fill=0`` (see the roadmap, step 0.4).
"""
import contextlib
from dataclasses import dataclass, field

import numpy as np
import pyopencl as cl


@dataclass
class GuardedBuffer:
    """Bookkeeping for one guarded allocation.

    Attributes
    ----------
    parent : pyopencl.Buffer
        The large buffer: band + data + band. Kept here so it outlives its sub-buffer.
    queue : pyopencl.CommandQueue
        Queue of the same context, used to read the bands back.
    head : int
        Size in bytes of the band before the data (also the offset of the data).
    nbytes : int
        Size in bytes of the data, i.e. of the buffer the caller asked for.
    total : int
        Size in bytes of the parent buffer.
    shape : tuple
        Shape of the host array the buffer was created from (for error messages).
    """
    parent: object
    queue: object
    head: int
    nbytes: int
    total: int
    shape: tuple


@dataclass
class GuardReport:
    """What ``guard_bands()`` yields: the guarded buffers and a way to inspect their bands.

    Attributes
    ----------
    fill : float
        The value the bands were filled with.
    buffers : list of GuardedBuffer
        One entry per buffer created inside the ``with`` block, in creation order.
    """
    fill: float
    buffers: list = field(default_factory=list)

    def touched(self):
        """Return the guarded buffers whose bands no longer hold the fill value.

        A non-empty result means some kernel wrote outside a buffer. Call it
        after the kernels have run (inside or after the ``with`` block).

        Returns
        -------
        list of GuardedBuffer
        """
        touched = []
        for b in self.buffers:
            whole = np.empty(b.total // 4, np.float32)
            cl.enqueue_copy(b.queue, whole, b.parent)
            b.queue.finish()
            bands = np.concatenate([whole[: b.head // 4], whole[(b.head + b.nbytes) // 4:]])
            intact = np.isnan(bands).all() if np.isnan(self.fill) else (bands == np.float32(self.fill)).all()
            if not intact:
                touched.append(b)
        return touched


@contextlib.contextmanager
def guard_bands(fill=np.nan, pad_bytes=1 << 16):
    """Context manager: buffers created inside it get guard bands filled with ``fill``.

    Parameters
    ----------
    fill : float, optional
        Value written into the bands. ``nan`` (default) to detect out-of-bounds
        reads through NaN in the outputs; ``1.0`` to detect out-of-bounds writes
        with ``GuardReport.touched()``; ``0.0`` to give out-of-bounds reads a
        defined, harmless value (reproducible runs of PyFWI 0.1.10).
    pad_bytes : int, optional
        Size of each band in bytes (the band before the data is rounded up to
        the device alignment). The default, 64 KiB, covers more than 100 rows of
        the grids used in the tests (a 140-point float32 row is 560 bytes).

    Yields
    ------
    GuardReport
        Filled as buffers are created; it stays usable after the block ends.

    Notes
    -----
    Only buffers created with ``hostbuf`` are guarded; others are created normally.
    """
    original = cl.Buffer
    report = GuardReport(fill=float(fill))
    queues = {}  # one command queue per context, used to fill and read the parent buffers

    def guarded_buffer(ctx, flags, size=0, hostbuf=None):
        if hostbuf is None:
            return original(ctx, flags, size)
        host = np.ascontiguousarray(hostbuf)
        nbytes = host.nbytes
        # Sub-buffers must start at a multiple of the device base-address alignment
        # (reported in bits by OpenCL).
        align = max(ctx.devices[0].mem_base_addr_align // 8, 4)
        head = -(-pad_bytes // align) * align          # band before the data, rounded up
        total = head + nbytes + pad_bytes
        total += -total % 4                            # whole number of float32 values
        parent = original(ctx, cl.mem_flags.READ_WRITE, total)
        queue = queues.setdefault(ctx.int_ptr, cl.CommandQueue(ctx))
        cl.enqueue_copy(queue, parent, np.full(total // 4, fill, np.float32))   # bands (and data area)
        cl.enqueue_copy(queue, parent, host, dst_offset=head)                   # then the data
        queue.finish()
        report.buffers.append(GuardedBuffer(parent, queue, head, nbytes, total, host.shape))
        return parent.get_sub_region(head, nbytes)

    cl.Buffer = guarded_buffer
    try:
        yield report
    finally:
        cl.Buffer = original


def nan_guard_bands(pad_bytes=1 << 16):
    """Shortcut for ``guard_bands(fill=nan)``: out-of-bounds reads turn the outputs into NaN."""
    return guard_bands(np.nan, pad_bytes)


def zero_guard_bands(pad_bytes=1 << 16):
    """Shortcut for ``guard_bands(fill=0)``: out-of-bounds reads return 0 (defined, reproducible behaviour)."""
    return guard_bands(0.0, pad_bytes)
