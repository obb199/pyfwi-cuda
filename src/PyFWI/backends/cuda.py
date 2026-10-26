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
* Copies and launches run on the current CUDA stream, in order; ``to_host`` waits for the copy, so
  "non-blocking" copies are blocking here (overlap comes with the optimisations of Phase 3).

Used by
-------
``PyFWI.wave_propagation.WavePreparation`` when ``PyFWI.backends.selection`` chooses CUDA. Tested
by ``test/test_cuda_backend.py`` and by the parity tests.
"""
import os
import re

import cupy as cp
import numpy as np

#: The portability header, prepended to every kernel source.
HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "opencl_on_cuda.h")

#: Threads per block: 32 columns (x, contiguous in memory) by 8 rows.
BLOCK = (32, 8, 1)

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
        properties = cp.cuda.runtime.getDeviceProperties(ordinal)
        name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
        self.description = f"{name} (CUDA device {ordinal})"
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
    def allocate_empty(nbytes):
        """A device buffer of ``nbytes`` bytes (float32 values), content undefined."""
        return cp.empty(nbytes // 4, dtype=np.float32)

    @staticmethod
    def nbytes(buffer):
        """Size of a buffer in bytes."""
        return buffer.nbytes

    # ------------------------------------------------------------------ copies

    @staticmethod
    def to_device(buffer, host, blocking=True):
        """Copy ``host`` into ``buffer``; returns a completed event."""
        buffer.set(np.ascontiguousarray(host).reshape(-1).view(buffer.dtype))
        return _Done()

    @staticmethod
    def to_host(host, buffer, blocking=True):
        """Copy ``buffer`` into the contiguous array ``host``; returns a completed event."""
        buffer.get(out=host.reshape(-1).view(buffer.dtype))
        return _Done()

    @staticmethod
    def copy(target, source, nbytes, offset=0):
        """Copy ``nbytes`` bytes from byte ``offset`` of ``source`` to the start of ``target``, on the device."""
        n, start = nbytes // target.itemsize, offset // source.itemsize
        target[:n] = source[start:start + n]
        return _Done()

    @staticmethod
    def wait(events):
        """Every copy of this backend has completed when it returns: nothing to wait for."""

    def finish(self):
        """Wait until everything launched on the GPU has completed."""
        self.device.synchronize()

    # ------------------------------------------------------------------ kernels

    def build(self, source):
        """Compile an OpenCL source of PyFWI as CUDA C++ for this GPU (``portable_source``)."""
        module = cp.RawModule(code=portable_source(source), options=(f"--gpu-architecture={self.architecture}",))
        module.compile()
        return CUDAProgram(module)

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
    def grid(global_size):
        """Blocks covering an OpenCL global size ``(rows, columns)``, rounded up."""
        rows, columns = global_size
        return (-(-columns // BLOCK[0]), -(-rows // BLOCK[1]), 1)

    def launch(self, kernel, global_size):
        """Run ``kernel`` on the work items of an OpenCL launch of ``global_size``, with the arguments set before."""
        kernel.function(self.grid(global_size), BLOCK, tuple(kernel.args))

    def call(self, kernel, global_size, *args):
        """Set every argument of ``kernel`` and run it."""
        self.set_args(kernel, *args)
        self.launch(kernel, global_size)
