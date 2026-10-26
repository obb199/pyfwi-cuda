"""Choice of the backend of a propagator (roadmap step 2.2).

What is asked, in order of precedence:

1. ``inpa['backend']``;
2. the ``PYFWI_BACKEND`` environment variable;
3. ``DEFAULT`` (``"opencl"`` in Phase 2: CUDA comes in as an option).

The values:

* ``"opencl"``: the OpenCL engine of PyFWI 0.1.10;
* ``"cuda"``: CUDA if it can run here, otherwise OpenCL, with the reason in the log (warning);
* ``"cuda:strict"``: CUDA, or an error that says why it cannot run;
* ``"auto"``: CUDA when the OpenCL device that would be used is an NVIDIA GPU and CUDA can run,
  otherwise OpenCL (the reason in the log, at INFO level). A CPU or a non-NVIDIA GPU chosen with
  ``inpa['platform']``/``inpa['device']`` therefore keeps ``auto`` on OpenCL.

CUDA can run when CuPy imports, sees a GPU, the NVIDIA driver supports the CUDA version of CuPy,
and this PyFWI has a CUDA backend (``PyFWI.backends.cuda``, step 2.3). The CUDA device is the one
with the PCI bus ID of the chosen OpenCL device when that is an NVIDIA GPU, so that
``inpa['platform']``/``inpa['device']`` keep their meaning; otherwise CUDA device 0.

Used by
-------
``PyFWI.wave_propagation.WavePreparation`` and ``PyFWI.info``. Tested by
``test/test_backend_selection.py``, with simulated CuPy, GPUs, drivers and OpenCL devices.
"""
import importlib.util
import logging
import os
from dataclasses import dataclass

#: Values of ``inpa['backend']`` and ``PYFWI_BACKEND``.
CHOICES = ("opencl", "cuda", "cuda:strict", "auto")

#: What runs when nothing is asked.
DEFAULT = "opencl"

#: Environment variable read when ``inpa`` has no ``backend`` key.
ENVIRONMENT_VARIABLE = "PYFWI_BACKEND"

#: Where the choice and the fallbacks are logged.
logger = logging.getLogger("PyFWI")


@dataclass
class CudaStatus:
    """Whether CUDA can run here (``probe_cuda``).

    Attributes
    ----------
    usable : bool
        True when every condition holds.
    reason : str
        Why not (empty when usable).
    bus_ids : list of str
        PCI bus ID of each CUDA device, by ordinal (``"0000:04:00.0"``).
    """

    usable: bool
    reason: str = ""
    bus_ids: tuple = ()


@dataclass
class Selection:
    """The backend chosen for a propagator (``select_backend``).

    Attributes
    ----------
    backend : str
        ``"opencl"`` or ``"cuda"``: what runs.
    requested : str
        What was asked (one of ``CHOICES``).
    origin : str
        Who asked: ``"inpa['backend']"``, ``"PYFWI_BACKEND"`` or ``"default"``.
    cuda_ordinal : int or None
        CUDA device, when ``backend`` is ``"cuda"``.
    reason : str
        Why the backend differs from what was asked (empty otherwise).
    """

    backend: str
    requested: str
    origin: str
    cuda_ordinal: int = None
    reason: str = ""

    def describe(self):
        """One line for the log and ``PyFWI.info()``.

        For example ``"opencl"`` (the default), ``"cuda device 0 (cuda asked by PYFWI_BACKEND)"`` or
        ``"opencl (auto asked by inpa['backend']; CUDA cannot run: CuPy is not installed ...)"``.
        """
        text = f"cuda device {self.cuda_ordinal}" if self.backend == "cuda" else self.backend
        if self.origin != "default":
            text += f" ({self.requested} asked by {self.origin}"
            text += f"; CUDA cannot run: {self.reason})" if self.reason else ")"
        return text


def requested_backend(inpa, environ=None):
    """``(value, origin)``: what is asked, by ``inpa``, the environment or the default (see the module docstring).

    Raises
    ------
    ValueError
        A value that is not one of ``CHOICES``; the message lists them.
    """
    environ = os.environ if environ is None else environ
    if inpa.get("backend") is not None:
        value, origin = inpa["backend"], "inpa['backend']"
    elif environ.get(ENVIRONMENT_VARIABLE):
        value, origin = environ[ENVIRONMENT_VARIABLE], ENVIRONMENT_VARIABLE
    else:
        value, origin = DEFAULT, "default"
    value = str(value).strip().lower()
    if value not in CHOICES:
        raise ValueError(f"Unknown backend {value!r} ({origin}); choose one of {', '.join(CHOICES)}.")
    return value, origin


def has_cuda_backend():
    """True when this PyFWI has a CUDA backend (``PyFWI.backends.cuda``, roadmap step 2.3)."""
    return importlib.util.find_spec("PyFWI.backends.cuda") is not None


def probe_cuda():
    """Whether CUDA can run here, and the PCI bus ID of every CUDA device (``CudaStatus``).

    Never raises: any failure becomes the reason.
    """
    try:
        import cupy
    except ImportError:
        return CudaStatus(False, "CuPy is not installed (pip install 'pyfwi-cuda[cuda13]')")
    try:
        runtime = cupy.cuda.runtime
        count = runtime.getDeviceCount()
        if count == 0:
            return CudaStatus(False, "no CUDA GPU is visible")
        driver, needed = runtime.driverGetVersion(), runtime.runtimeGetVersion()
        if driver // 1000 < needed // 1000:
            return CudaStatus(False, f"the NVIDIA driver supports CUDA {driver // 1000}.{driver % 1000 // 10}, "
                                     f"this CuPy needs CUDA {needed // 1000}: update the driver or install "
                                     f"CuPy for CUDA {driver // 1000}")
        bus_ids = tuple(runtime.deviceGetPCIBusId(i).lower() for i in range(count))
    except Exception as error:  # noqa: BLE001 (a broken CUDA install is a reason, not a crash)
        return CudaStatus(False, f"CUDA does not work: {type(error).__name__}: {error}")
    if not has_cuda_backend():
        return CudaStatus(False, "this version of PyFWI has no CUDA backend yet", bus_ids)
    return CudaStatus(True, "", bus_ids)


def opencl_bus_id(device):
    """PCI bus ID (``"0000:04:00.0"``) of an NVIDIA OpenCL device, or None for other devices.

    NVIDIA OpenCL reports the domain, bus and slot (``cl_nv_device_attribute_query``).
    """
    if device is None or "NVIDIA" not in device.platform.name.upper():
        return None
    try:
        return f"{device.pci_domain_id_nv:04x}:{device.pci_bus_id_nv:02x}:{device.pci_slot_id_nv:02x}.0"
    except Exception:  # noqa: BLE001 (an old driver without the extension: no match, not a crash)
        return None


def select_backend(inpa, environ=None, device=None, probe=probe_cuda):
    """Choose the backend of a propagator (see the module docstring); logs fallbacks.

    Parameters
    ----------
    inpa : dict
        Inputs of the propagator (``backend``, ``platform`` and ``device`` are read).
    environ : mapping, optional
        Where ``PYFWI_BACKEND`` is read; ``os.environ`` by default.
    device : pyopencl.Device, optional
        The OpenCL device the propagator would use (``choose_devices``), for ``auto`` and for the
        CUDA device; looked up when not given and needed.
    probe : callable, optional
        Returns a ``CudaStatus``; ``probe_cuda`` by default (tests pass simulated ones).

    Returns
    -------
    Selection

    Raises
    ------
    ValueError
        An unknown backend.
    RuntimeError
        ``"cuda:strict"`` when CUDA cannot run; the message says why.
    """
    requested, origin = requested_backend(inpa, environ)
    if requested == "opencl":
        return Selection("opencl", requested, origin)

    status = probe()
    if device is None:
        device = _opencl_device(inpa, environ)
    bus_id = opencl_bus_id(device)
    explicit_device = inpa.get("platform") is not None or inpa.get("device") is not None
    reason = status.reason
    ordinal = None
    if status.usable:
        if bus_id is not None and bus_id in status.bus_ids:
            ordinal = status.bus_ids.index(bus_id)
        elif requested == "auto" or explicit_device:
            # auto only follows an NVIDIA OpenCL device; cuda follows inpa when it names a device.
            reason = (f"the chosen OpenCL device ({device.name} on {device.platform.name}) is not a CUDA GPU"
                      if device is not None else "no OpenCL device to match with a CUDA GPU")
        else:
            ordinal = 0

    if ordinal is not None:
        selection = Selection("cuda", requested, origin, ordinal)
        logger.info("PyFWI backend: %s", selection.describe())
        return selection
    if requested == "cuda:strict":
        raise RuntimeError(f"The CUDA backend was required ({origin} = 'cuda:strict') but cannot run: {reason}.")
    selection = Selection("opencl", requested, origin, None, reason)
    (logger.warning if requested == "cuda" else logger.info)("PyFWI backend: %s", selection.describe())
    return selection


def _opencl_device(inpa, environ):
    """The first OpenCL device a propagator with ``inpa`` would use, or None without OpenCL."""
    try:
        from PyFWI._opencl_device import choose_devices
        return choose_devices(inpa.get("platform"), inpa.get("device"), environ=environ)[0]
    except Exception:  # noqa: BLE001 (no OpenCL: nothing to match)
        return None


#: Numeric modes (step 2.4): ``legacy`` computes what PyFWI 0.1.10 computes, ``fixed`` corrects it.
NUMERICS = ("legacy", "fixed")

#: Numeric mode of each backend when none is asked: OpenCL keeps the results of 0.1.10, the CUDA
#: backend starts in the corrected mode.
DEFAULT_NUMERICS = {"opencl": "legacy", "cuda": "fixed"}


def select_numerics(inpa, backend, environ=None):
    """The numeric mode of a propagator: ``inpa['numerics']``, else ``PYFWI_NUMERICS``, else the backend's default.

    Parameters
    ----------
    inpa : dict
        Inputs of the propagator (``numerics`` is read).
    backend : str
        ``"opencl"`` or ``"cuda"``, the backend that runs (``Selection.backend``).
    environ : mapping, optional
        Where ``PYFWI_NUMERICS`` is read; ``os.environ`` by default.

    Returns
    -------
    str
        ``"legacy"`` or ``"fixed"``.

    Raises
    ------
    ValueError
        A mode that is not one of ``NUMERICS``.
    """
    environ = os.environ if environ is None else environ
    value = inpa.get("numerics") or environ.get("PYFWI_NUMERICS") or DEFAULT_NUMERICS[backend]
    value = str(value).strip().lower()
    if value not in NUMERICS:
        raise ValueError(f"Unknown numeric mode {value!r}; choose one of {', '.join(NUMERICS)}.")
    return value


#: Kernel sets of the CUDA backend (Phase 3): ``native``, the CUDA kernels of ``PyFWI/kernels/elastic.cu``
#: (fixed mode only), and ``parity``, the OpenCL files compiled for CUDA (step 2.3).
CUDA_KERNELS = ("native", "parity")


def select_cuda_kernels(numerics, environ=None):
    """The kernel set of a propagator on the CUDA backend.

    In the fixed numeric mode the native kernels, unless ``PYFWI_CUDA_KERNELS`` is ``parity`` (the
    tests compare both sets); in the legacy mode the parity kernels, the only ones with its arithmetic.

    Parameters
    ----------
    numerics : str
        The numeric mode of the propagator (``select_numerics``).
    environ : mapping, optional
        Where ``PYFWI_CUDA_KERNELS`` is read; ``os.environ`` by default.

    Returns
    -------
    str
        ``"native"`` or ``"parity"``.

    Raises
    ------
    ValueError
        A value of ``PYFWI_CUDA_KERNELS`` that is not one of ``CUDA_KERNELS``.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get("PYFWI_CUDA_KERNELS") or "native").strip().lower()
    if value not in CUDA_KERNELS:
        raise ValueError(f"Unknown PYFWI_CUDA_KERNELS {value!r}; choose one of {', '.join(CUDA_KERNELS)}.")
    return "native" if numerics == "fixed" and value == "native" else "parity"
