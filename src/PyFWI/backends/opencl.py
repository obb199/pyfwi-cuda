"""The OpenCL backend: PyOpenCL behind the interface of ``PyFWI.backends`` (roadmap step 2.1).

It gathers the OpenCL calls that ``PyFWI.wave_propagation`` made directly until step 2.1, with
the behaviour of the earlier steps of the fork:

* the device is chosen by ``PyFWI._opencl_device.choose_devices`` and the context is shared by
  every propagator on the same devices (steps 1.2 and 1.7);
* buffers live inside allocations with margins, created empty and written whole by an explicit
  copy (step 1.7);
* kernels are launched with their arguments set beforehand (step 1.3).

Every OpenCL call goes through the ``pyopencl`` module at the time of the call
(``cl.enqueue_copy``, ``cl.enqueue_nd_range_kernel``), so that the instrumentation of the tests and
tools (``tools/benchmark.py``, ``tools/opencl_profile.py``) sees it.

Used by
-------
``PyFWI.wave_propagation.WavePreparation``. Tested by ``test/test_backends.py`` and by every test
that runs a propagation.
"""
import os

import numpy as np
import pyopencl as cl

from PyFWI._opencl_device import choose_devices, shared_context


class OpenCLBackend:
    """Run PyFWI's kernels with PyOpenCL on one OpenCL device (see ``PyFWI.backends``).

    Parameters
    ----------
    platform, device : int or str, optional
        ``inpa['platform']`` and ``inpa['device']`` (see ``choose_devices``).

    Attributes
    ----------
    name : str
        ``"opencl"``.
    context : pyopencl.Context
        Shared by every backend on the same devices (``shared_context``).
    queue : pyopencl.CommandQueue
        In-order queue of this backend: commands run in the order they are enqueued.
    alignment : int
        Base-address alignment of the devices in bytes: sub-buffers must start at a multiple of it.
    description : str
        The device(s) and platform, for the log and ``PyFWI.info()``.
    """

    name = "opencl"

    def __init__(self, platform=None, device=None):
        devices = choose_devices(platform, device)
        if device is not None:
            # As in 0.1.10, the choice is also written to the environment, so that a later
            # WavePropagator created without inpa['device'] runs on the same device.
            os.environ['PYOPENCL_CTX'] = str(0 if platform is None else platform) + ':' + str(device)
            os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        # One context per set of devices for the whole process (see shared_context: releasing
        # contexts makes the NVIDIA driver fail after a few propagators); one queue per backend.
        self.context = shared_context(devices)
        self.queue = cl.CommandQueue(self.context)
        self.alignment = max(max(d.mem_base_addr_align for d in self.context.devices) // 8, 4)
        self.description = ", ".join(f"{d.name} on {d.platform.name}" for d in self.context.devices)

    # ------------------------------------------------------------------ memory

    def allocate(self, host, margin_bytes, fill):
        """A device buffer holding ``host``, inside an allocation with margins of ``fill``.

        Parameters
        ----------
        host : numpy.ndarray
            Initial content; any shape, copied as a contiguous block.
        margin_bytes : int
            Length of each margin, a multiple of ``alignment``.
        fill : float
            Value of the margins.

        Returns
        -------
        buffer : pyopencl.Buffer
            Sub-buffer of the data: what the kernels use.
        allocation : pyopencl.Buffer
            The whole allocation, margins included.
        """
        host = np.ascontiguousarray(host)
        whole = np.full((2 * margin_bytes + host.nbytes) // host.itemsize, fill, dtype=host.dtype)
        whole.view(np.uint8)[margin_bytes:margin_bytes + host.nbytes] = host.reshape(-1).view(np.uint8)
        # Allocated first, then written whole by an explicit copy. With COPY_HOST_PTR the NVIDIA
        # driver places on the device only the region of the sub-buffer the kernels use, and the
        # margins there hold whatever the device memory held. (Without hostbuf the buffer is also
        # never wrapped by the guard bands of the tests, test/oob_guard.py: nesting their
        # sub-buffers in this one would make overlapping sub-buffers, which crash the driver.)
        allocation = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, whole.nbytes)
        cl.enqueue_copy(self.queue, allocation, whole)
        return allocation.get_sub_region(margin_bytes, host.nbytes), allocation

    def allocate_empty(self, nbytes):
        """A device buffer of ``nbytes`` bytes, without margins; its content is undefined."""
        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, nbytes)

    @staticmethod
    def nbytes(buffer):
        """Size of a buffer in bytes."""
        return buffer.size

    # ------------------------------------------------------------------ copies

    def to_device(self, buffer, host, blocking=True):
        """Copy ``host`` into ``buffer``; returns the event of the copy."""
        return cl.enqueue_copy(self.queue, buffer, host, is_blocking=blocking)

    def to_host(self, host, buffer, blocking=True):
        """Copy ``buffer`` into the contiguous array ``host``; returns the event of the copy."""
        return cl.enqueue_copy(self.queue, host, buffer, is_blocking=blocking)

    def copy(self, target, source, nbytes, offset=0):
        """Copy ``nbytes`` bytes from byte ``offset`` of ``source`` to the start of ``target``, on the device."""
        return cl.enqueue_copy(self.queue, target, source, byte_count=nbytes, src_offset=offset)

    @staticmethod
    def wait(events):
        """Wait until every event of ``events`` has completed."""
        cl.wait_for_events(list(events))

    def finish(self):
        """Wait until every command enqueued so far has completed."""
        self.queue.finish()

    # ------------------------------------------------------------------ kernels

    def build(self, source):
        """Compile an OpenCL program from its source (the ``#define`` block of PyFWI included)."""
        return cl.Program(self.context, source).build()

    @staticmethod
    def kernel(program, name):
        """The kernel ``name`` of ``program``. Retrieve each kernel once and reuse it (step 1.3)."""
        return getattr(program, name)

    @staticmethod
    def set_args(kernel, *args):
        """Set every argument of ``kernel`` (buffers and NumPy scalars, in the order of the source)."""
        kernel.set_args(*args)

    @staticmethod
    def set_arg(kernel, index, value):
        """Set argument number ``index`` of ``kernel``."""
        kernel.set_arg(index, value)

    def launch(self, kernel, global_size):
        """Run ``kernel`` on ``global_size`` work items, with the arguments set before."""
        return cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, None)

    def call(self, kernel, global_size, *args):
        """Set every argument of ``kernel`` and run it on ``global_size`` work items."""
        return kernel(self.queue, global_size, None, *args)
