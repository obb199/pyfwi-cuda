"""The CUDA backend: PyFWI's OpenCL C kernels compiled with NVRTC and run with CuPy (roadmap step 2.3).

The *parity* backend: the same kernel files as the OpenCL backend (``elastic.cl``...), compiled
as CUDA C++ after the portability header ``PyFWI/kernels/opencl_on_cuda.h``, in the ``legacy``
numeric mode (the arithmetic of 0.1.10, float literals without ``f`` included). Its results must
agree with OpenCL on the same GPU to a relative L2 error of 1e-6 (the compilers may contract
multiplications and additions differently); the optimised kernels come in Phase 3.

* ``portable_source`` turns an OpenCL source into CUDA C++: the header, and at the start of every
  kernel a return for the threads beyond the OpenCL global size (the grid of blocks is rounded up).
* Buffers are CuPy arrays: an allocation ``[margin | data | margin]`` and a view of its data
  (step 1.7), so that the out-of-bounds accesses of the kernels stay inside the allocation.
* NVRTC compiles for the architecture of the GPU and CuPy loads the cubin, which also works when the
  NVRTC installed by pip is newer than the driver (PTX would not load).
* Copies and launches run on the current CUDA stream, in order. A non-blocking copy goes through a
  pinned (page-locked) host buffer (``_StagingPool``): a copy from pageable NumPy memory would make
  the host wait for the GPU (device to host) or for the driver's staging (host to device), and the
  launches of the time loop would no longer run ahead of the GPU (step 2.7: the gradient of the
  small benchmark scenario was 8 % slower than on OpenCL because of it).

Used by
-------
``PyFWI.wave_propagation.WavePreparation`` when ``PyFWI.backends.selection`` chooses CUDA. Tested
by ``test/test_cuda_backend.py`` and by the parity tests.
"""
import os
import re
from collections import deque

import cupy as cp
import cupyx
import numpy as np

#: The portability header, prepended to every kernel source.
HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "opencl_on_cuda.h")

#: Threads per block: 32 columns (x, contiguous in memory) by up to 8 rows. ``block`` takes fewer
#: rows (4, then 2) for grids too small to give every streaming multiprocessor ``BLOCKS_PER_SM``
#: blocks; the results do not depend on it (every work item is independent).
BLOCK = (32, 8, 1)
BLOCK_ROWS = (8, 4, 2)
#: Blocks per streaming multiprocessor below which a launch takes shorter blocks. Measured on the
#: RTX 3060 (28 SMs), legacy mode, gradient of the small benchmark scenario (140 x 140 grid): 0.80 s
#: with 32 x 8 blocks (90 blocks), 0.73 s with 32 x 4, 0.64 s with 32 x 2 (350 blocks); on the
#: medium one (440 x 440) 32 x 8 and 32 x 4 are within 2 %. Step 3.1 tried to tune the block of each
#: kernel at run time instead, timing it alone with every shape: in the time loops the chosen blocks
#: were slower than this rule (fixed mode, small scenario 10 %, medium forward 8 to 13 %; a kernel run
#: alone over and over finds its data in the L2 cache, which the loops do not), so the rule stays.
BLOCKS_PER_SM = 8

#: Pinned staging buffers kept for each copy size (``_StagingPool``): a buffer is reused, after its
#: previous copy has completed, only once this many later copies of the same size have started.
#: The time loops copy five fields at a time, so 16 leaves three checkpoints in flight.
STAGING_DEPTH = 16

#: Inserted before the first statement of every kernel: only the work items of the OpenCL launch,
#: whose global size is (Nz, Nx), run (``Nz`` and ``Nx`` are the grid size of PyFWI's #define block).
GUARD = "if (get_global_id(0) >= Nz || get_global_id(1) >= Nx) return;\n  "


def portable_source(source):
    """CUDA C++ source of an OpenCL source of PyFWI: the header, and the bounds guard in every kernel.

    Every kernel of PyFWI starts with ``int i = get_global_id(0)...``; the guard goes just before.

    Raises
    ------
    ValueError
        If some kernel does not start that way (the guard would be missing).
    """
    with open(HEADER) as f:
        header = f.read()
    guarded, count = re.subn(r"(\bint\s+i\s*=\s*get_global_id\(0\))", GUARD.replace("\\", "\\\\") + r"\1", source)
    kernels = len(re.findall(r"__kernel\s+void", source))
    if count != kernels:
        raise ValueError(f"{kernels} kernels but {count} start with 'int i = get_global_id(0)': cannot guard them")
    return header + "\n" + guarded


def forced_block(text):
    """The block ``(columns, rows, 1)`` of a ``PYFWI_CUDA_BLOCK`` value such as ``"64x4"``.

    Raises
    ------
    ValueError
        If ``text`` is not two positive integers joined by ``x`` with at most 1,024 threads.
    """
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", text)
    columns, rows = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    if not (columns > 0 and rows > 0 and columns * rows <= 1024):
        raise ValueError(f"PYFWI_CUDA_BLOCK={text!r}: expected '<columns>x<rows>' with at most 1024 threads, "
                         "for example '32x4'")
    return (columns, rows, 1)


class CUDAKernel:
    """A compiled kernel and the arguments set for its next launch."""

    def __init__(self, name, function):
        self.function_name, self.function, self.args = name, function, []


class CUDAProgram:
    """A compiled module: gives a kernel for each name, as ``pyopencl.Program`` does."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return CUDAKernel(name, self.module.get_function(name))


class _Done:
    """Event of a copy that has already completed."""

    @staticmethod
    def wait():
        pass


class _Copy:
    """Event of a copy through a pinned staging buffer (returned by the non-blocking copies).

    Parameters
    ----------
    event : cupy.cuda.Event
        Recorded on the stream just after the copy.
    staging, destination : numpy.ndarray, optional
        For a copy to the host: the pinned buffer the GPU copies into, and the array the data is
        for. ``wait`` moves the data from one to the other once the GPU copy has completed.
    """

    def __init__(self, event, staging=None, destination=None):
        self.event, self.staging, self.destination = event, staging, destination

    def wait(self):
        """Wait for the GPU copy; for a copy to the host, then fill the destination (once)."""
        self.event.synchronize()
        if self.destination is not None:
            np.copyto(self.destination, self.staging)
            self.destination = None


class _StagingPool:
    """Pinned host buffers that make the non-blocking copies of ``CUDABackend`` asynchronous.

    For each copy size, up to ``depth`` buffers in turn, created at first use. A buffer is reused
    only after its previous copy has completed (``_Copy.wait``), so a copy to the host reaches its
    destination at the latest when its buffer comes round again, or at ``complete``.

    Parameters
    ----------
    depth : int
        Buffers per size (``STAGING_DEPTH``).
    """

    def __init__(self, depth=STAGING_DEPTH):
        self.depth = depth
        #: ``{nbytes: deque of [pinned uint8 array, its last copy or None]}``.
        self.rings = {}

    def take(self, nbytes):
        """A free pinned buffer of ``nbytes`` bytes, as a ``[array, copy]`` slot to fill in."""
        ring = self.rings.setdefault(nbytes, deque())
        if len(ring) < self.depth:
            slot = [cupyx.empty_pinned(nbytes, np.uint8), None]
        else:
            slot = ring.popleft()
            if slot[1] is not None:
                slot[1].wait()
        ring.append(slot)
        return slot

    def complete(self):
        """Complete every copy still in flight (copies to the host reach their destination)."""
        for ring in self.rings.values():
            for slot in ring:
                if slot[1] is not None:
                    slot[1].wait()


class CUDABackend:
    """Run PyFWI's kernels with CuPy on one CUDA GPU (see ``PyFWI.backends`` and the module docstring).

    Parameters
    ----------
    ordinal : int
        CUDA device number (``PyFWI.backends.selection`` matches it to the OpenCL device by PCI bus ID).

    Attributes
    ----------
    name : str
        ``"cuda"``.
    device : cupy.cuda.Device
        The GPU.
    description : str
        Its name and number, for the log and ``PyFWI.info()``.
    alignment : int
        Offset alignment of the data inside an allocation (CuPy allocates on 256-byte boundaries).
    context, queue : None
        OpenCL objects, which this backend has not (``W.ctx`` and ``W.queue`` are None).
    """

    name = "cuda"
    alignment = 256
    context = queue = None

    def __init__(self, ordinal=0):
        self.device = cp.cuda.Device(ordinal)
        self.device.use()
        #: Pinned buffers of the non-blocking copies.
        self._staging = _StagingPool()
        #: The stream of every copy and launch (made current here): a stream of its own, since the
        #: legacy default stream cannot be recorded into a CUDA graph (``capture``, step 3.7).
        self.stream = cp.cuda.Stream()
        self.stream.use()

        properties = cp.cuda.runtime.getDeviceProperties(ordinal)
        name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
        self.description = f"{name} (CUDA device {ordinal})"
        #: Number of streaming multiprocessors (``block``), and the block of each global size used.
        self.multiprocessors = int(properties["multiProcessorCount"])
        self._blocks = {}
        self.architecture = f"sm_{properties['major']}{properties['minor']}"

    # ------------------------------------------------------------------ memory

    def allocate(self, host, margin_bytes, fill):
        """A device buffer holding ``host`` inside an allocation with margins of ``fill`` (see ``PyFWI.backends``)."""
        host = np.ascontiguousarray(host)
        itemsize = host.itemsize
        whole = np.full((2 * margin_bytes + host.nbytes) // itemsize, fill, dtype=host.dtype)
        whole[margin_bytes // itemsize:(margin_bytes + host.nbytes) // itemsize] = host.reshape(-1)
        allocation = cp.asarray(whole)
        return allocation[margin_bytes // itemsize:(margin_bytes + host.nbytes) // itemsize], allocation

    @staticmethod
    def allocate_empty(nbytes, dtype=np.float32):
        """A device buffer of ``nbytes`` bytes of ``dtype`` values (float16 for the FP16 fields of step 4.3),
        content undefined."""
        return cp.empty(nbytes // np.dtype(dtype).itemsize, dtype=dtype)

    @staticmethod
    def nbytes(buffer):
        """Size of a buffer in bytes."""
        return buffer.nbytes

    # ------------------------------------------------------------------ copies

    def _record(self):
        """A CUDA event recorded now on the current stream."""
        event = cp.cuda.Event(block=False, disable_timing=True)
        event.record()
        return event

    def to_device(self, buffer, host, blocking=True):
        """Copy ``host`` into ``buffer``; returns an event (``wait()``).

        Non-blocking: ``host`` is copied at once into a pinned buffer, from which the GPU copies it
        when the stream gets there; ``host`` may change as soon as this returns.
        """
        host = np.ascontiguousarray(host).reshape(-1).view(buffer.dtype)
        if blocking:
            buffer.set(host)
            return _Done()
        slot = self._staging.take(host.nbytes)
        staging = slot[0].view(buffer.dtype)
        np.copyto(staging, host)
        buffer.set(staging)  # from pinned memory: asynchronous on the current stream
        slot[1] = _Copy(self._record())
        return slot[1]

    def to_host(self, host, buffer, blocking=True):
        """Copy ``buffer`` into the contiguous array ``host``; returns an event (``wait()``).

        Non-blocking: the GPU copies into a pinned buffer when the stream gets there, and ``host`` is
        filled when the event is waited for (``wait``, ``finish``), at the latest.
        """
        destination = host.reshape(-1).view(buffer.dtype)
        if blocking:
            buffer.get(out=destination)
            return _Done()
        slot = self._staging.take(buffer.nbytes)
        staging = slot[0].view(buffer.dtype)
        buffer.get(out=staging, blocking=False)
        slot[1] = _Copy(self._record(), staging, destination)
        return slot[1]

    @staticmethod
    def section(buffer, offset, nbytes):
        """The ``nbytes`` bytes of ``buffer`` from byte ``offset``, as a buffer (a view: no copy)."""
        return buffer[offset // buffer.itemsize:(offset + nbytes) // buffer.itemsize]

    @staticmethod
    def checkpoint_budget():
        """Bytes of device memory the checkpoints of a propagator may take (step 3.8).

        ``PYFWI_CUDA_CHECKPOINT_BUDGET`` (bytes) when set, else 80 % of the free memory: the
        driver's, plus the blocks CuPy keeps in its memory pool for reuse. Above it, the
        checkpoints stay on the host.
        """
        forced = os.environ.get("PYFWI_CUDA_CHECKPOINT_BUDGET")
        if forced:
            return int(float(forced))
        free, _ = cp.cuda.runtime.memGetInfo()
        return int(0.8 * (free + cp.get_default_memory_pool().free_bytes()))

    @staticmethod
    def copy(target, source, nbytes, offset=0):
        """Copy ``nbytes`` bytes from byte ``offset`` of ``source`` to the start of ``target``, on the device."""
        n, start = nbytes // target.itemsize, offset // source.itemsize
        target[:n] = source[start:start + n]
        return _Done()

    @staticmethod
    def wait(events):
        """Wait for the copies of ``events`` (copies to the host then reach their destination)."""
        for event in events:
            event.wait()

    def finish(self):
        """Wait until everything launched on the GPU has completed, copies to the host included."""
        self.device.synchronize()
        self._staging.complete()

    # ------------------------------------------------------------------ kernels

    def build(self, source):
        """Compile an OpenCL source of PyFWI as CUDA C++ for this GPU (``portable_source``)."""
        module = cp.RawModule(code=portable_source(source), options=(f"--gpu-architecture={self.architecture}",))
        module.compile()
        return CUDAProgram(module)

    def build_native(self, source):
        """Compile CUDA C++ source as it is: the native kernels of Phase 3 (``PyFWI/kernels/elastic.cu``
        after the ``#define`` block), which need neither the portability header nor the guard."""
        module = cp.RawModule(code=source, options=(f"--gpu-architecture={self.architecture}",))
        module.compile()
        return CUDAProgram(module)

    @staticmethod
    def capture(record):
        """Record the launches made by ``record()`` into a CUDA graph and return it (step 3.7).

        Nothing runs while recording; ``graph.launch()`` then replays every launch, in order, on the
        current stream, at the cost of one call. The launches keep the arguments they had when
        recorded, so the kernels of a graph read what changes (the time step) from the device.
        """
        stream = cp.cuda.get_current_stream()
        stream.begin_capture()
        try:
            record()
        finally:
            graph = stream.end_capture()
        return graph

    @staticmethod
    def kernel(program, name):
        """The kernel ``name`` of ``program``. Retrieve each kernel once and reuse it."""
        return getattr(program, name)

    @staticmethod
    def set_args(kernel, *args):
        """Set every argument of ``kernel`` (CuPy arrays and NumPy scalars, in the order of the source)."""
        kernel.args = list(args)

    @staticmethod
    def set_arg(kernel, index, value):
        """Set argument number ``index`` of ``kernel``."""
        kernel.args[index] = value

    @staticmethod
    def grid(global_size, block=BLOCK):
        """Blocks of shape ``block`` covering an OpenCL global size ``(rows, columns)``, rounded up.

        A third size, ``(rows, columns, slots)``, is the number of shots of a batch (step 3.9): the
        grid then repeats the blocks of one shot ``slots`` times along z, and the native kernels take
        the slot from ``blockIdx.z``.
        """
        rows, columns, *slots = global_size
        return (-(-columns // block[0]), -(-rows // block[1]), slots[0] if slots else 1)

    def block(self, global_size):
        """The block for an OpenCL global size: 32 columns, and the most rows of ``BLOCK_ROWS`` that still
        give at least ``BLOCKS_PER_SM`` blocks per streaming multiprocessor (else the fewest rows), the
        blocks of every slot of a batch of shots counted (step 3.9).

        ``PYFWI_CUDA_BLOCK``, ``"<columns>x<rows>"`` (``"64x4"``), sets the block of every launch
        instead, to tune it for a GPU (step 3.1); the results do not depend on it.
        """
        if global_size not in self._blocks and os.environ.get("PYFWI_CUDA_BLOCK"):
            self._blocks[global_size] = forced_block(os.environ["PYFWI_CUDA_BLOCK"])
        if global_size not in self._blocks:
            choice = (BLOCK[0], BLOCK_ROWS[-1], 1)
            for rows in BLOCK_ROWS:
                columns, lines, slots = self.grid(global_size, (BLOCK[0], rows, 1))
                if columns * lines * slots >= BLOCKS_PER_SM * self.multiprocessors:
                    choice = (BLOCK[0], rows, 1)
                    break
            self._blocks[global_size] = choice
        return self._blocks[global_size]

    def launch(self, kernel, global_size):
        """Run ``kernel`` on the work items of an OpenCL launch of ``global_size``, with the arguments set before."""
        block = self.block(global_size)
        kernel.function(self.grid(global_size, block), block, tuple(kernel.args))

    def call(self, kernel, global_size, *args):
        """Set every argument of ``kernel`` and run it."""
        self.set_args(kernel, *args)
        self.launch(kernel, global_size)
