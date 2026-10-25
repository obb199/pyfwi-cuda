"""Choice of the OpenCL device that PyFWI runs on.

PyFWI 0.1.10 let PyOpenCL choose the device with ``pyopencl.create_some_context()``, after
writing ``inpa['platform']`` and ``inpa['device']`` into the ``PYOPENCL_CTX`` environment variable.
That had three problems, which this module removes while keeping the same choices:

* with PyOpenCL 2022, ``"<platform>:<device>"`` fails on a platform that has a single device
  ("not all provided choices were used ... left over"), so ``inpa['device']`` could not be used
  with, for example, the NVIDIA platform of a machine with one GPU;
* the range of ``inpa['device']`` was checked against the devices of the *first* platform, not of
  the chosen one;
* without ``inpa['device']`` and ``PYOPENCL_CTX``, in a terminal, PyOpenCL *asked* the user which
  platform and device to use. Now the answer is the default of that question (first platform,
  first device), without asking.

Which device is chosen (``choose_devices``), in order of precedence:

1. ``inpa['platform']`` and/or ``inpa['device']`` (keys of the ``inpa`` dictionary of
   ``WavePropagator``): that device of that platform. A missing key means 0, as in 0.1.10 (where
   ``inpa['platform']`` without ``inpa['device']`` was ignored, a bug).
2. The ``PYOPENCL_CTX`` environment variable, with PyOpenCL's syntax:
   ``"<platform>[:<device>[,<device>...]]"``, where each choice is an index or a case-insensitive
   part of the name (``"1"``, ``"nvidia"``, ``"portable:0"``).
3. The first device of the first platform.

PyOpenCL's own test variable ``PYOPENCL_TEST`` is not used.

Used by
-------
``PyFWI.wave_propagation.WavePreparation.__init__``, which builds the OpenCL context of every
``WavePropagator``. Tested by ``test/test_device_selection.py``.
"""
import os


def _platforms():
    """The OpenCL platforms of this machine, or a ``RuntimeError`` that says how to get one."""
    import pyopencl as cl
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as error:  # with an ICD loader and no driver: PLATFORM_NOT_FOUND_KHR
        platforms, cause = [], error
    else:
        cause = None
    if not platforms:
        raise RuntimeError("No OpenCL platform found. Install an OpenCL driver: the NVIDIA driver for an "
                           "NVIDIA GPU, or 'pip install pocl-binary-distribution' for the CPU.") from cause
    return platforms


def _listing(items):
    """``"[0] name, [1] name"``: the choices, for error messages."""
    return ", ".join(f"[{i}] {item.name}" for i, item in enumerate(items))


def _match(items, answer, kind, last=False):
    """The item chosen by ``answer``: an index, or a case-insensitive part of the name.

    Parameters
    ----------
    items : list
        Platforms or devices (objects with a ``name`` attribute).
    answer : str
        The choice, as written in ``PYOPENCL_CTX`` or taken from ``inpa``.
    kind : str
        "platform" or "device", for the error message.
    last : bool
        Which item wins when the name matches several: PyOpenCL takes the last matching
        platform and the first matching device.

    Raises
    ------
    ValueError
        If ``answer`` matches nothing; the message lists the choices.
    """
    try:
        index = int(answer)
    except ValueError:
        index = None
    if index is not None and 0 <= index < len(items):
        return items[index]
    # Not a valid index: PyOpenCL then looks for the answer in the names.
    matches = [item for item in items if answer.lower() in item.name.lower()]
    if matches:
        return matches[-1] if last else matches[0]
    raise ValueError(f"OpenCL {kind} {answer!r} does not exist. Available: {_listing(items)}.")


def choose_devices(platform=None, device=None, environ=None, platforms=None):
    """The OpenCL devices to run on, chosen without asking the user (see the module docstring).

    Parameters
    ----------
    platform : int or str, optional
        ``inpa['platform']``: index of the platform, or part of its name. None if not given.
    device : int or str, optional
        ``inpa['device']``: index of the device in that platform, or part of its name.
    environ : mapping, optional
        Where ``PYOPENCL_CTX`` is read; ``os.environ`` by default (tests pass a dict).
    platforms : list, optional
        The platforms to choose from; ``pyopencl.get_platforms()`` by default (tests pass
        simulated platforms: objects with ``name`` and ``get_devices()``).

    Returns
    -------
    list of pyopencl.Device
        One device, or several when ``PYOPENCL_CTX`` lists several (``"0:0,1"``), as PyOpenCL does.

    Raises
    ------
    ValueError
        A platform or device that does not exist.
    RuntimeError
        No OpenCL platform, or a platform without devices.
    """
    platforms = list(_platforms() if platforms is None else platforms)
    environ = os.environ if environ is None else environ
    if platform is None and device is None:
        platform_answer, _, device_answer = environ.get("PYOPENCL_CTX", "").partition(":")
    else:
        platform_answer = str(0 if platform is None else platform)
        device_answer = str(0 if device is None else device)

    chosen = _match(platforms, platform_answer, "platform", last=True) if platform_answer else platforms[0]
    devices = list(chosen.get_devices())
    if not devices:
        raise RuntimeError(f"The OpenCL platform {chosen.name!r} has no devices.")
    if not device_answer:
        return [devices[0]]
    return [_match(devices, answer, "device") for answer in device_answer.split(",")]
