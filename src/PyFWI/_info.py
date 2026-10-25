"""Diagnostics of the installed PyFWI (roadmap step 1.8): ``PyFWI.info()`` and the log of the first run.

When results differ between two machines, the first question is "what exactly ran?". ``info()``
answers it in one call: which package (this fork, at which git commit when it runs from a
checkout), which backend and numeric mode, which OpenCL device a propagator would use, and the
versions of the libraries that change results or speed.

The same summary is written to the ``"PyFWI"`` logger, at level INFO, by the first forward
modelling of the process that runs while that level is enabled (``log_first_run``). Python logs
nothing at INFO unless asked, e.g. with ``logging.basicConfig(level=logging.INFO)``.

Used by
-------
``PyFWI.info`` (the public function, in ``PyFWI/__init__.py``) and
``PyFWI.wave_propagation.WavePropagator.forward_modeling``. Tested by ``test/test_info.py``.
"""
import importlib.metadata
import logging
import os
import platform as python_platform
import subprocess

#: Logger of the package. The summary goes there once per process (``log_first_run``).
logger = logging.getLogger("PyFWI")

#: What this package is, for users who may confuse it with the original PyFWI.
ORIGIN = "pyfwi-cuda, a fork of PyFWI 0.1.10 (github.com/AmirMardan/PyFWI) towards a CUDA backend"

#: The engine that runs the propagation: OpenCL only, until the CUDA backend of Phase 2.
BACKEND = "opencl"

#: The numeric mode: the results of PyFWI 0.1.10. The "fixed" mode comes with step 2.4.
NUMERIC_MODE = "legacy"

#: Libraries whose versions are reported, by distribution name (some are optional).
LIBRARIES = ("numpy", "scipy", "pyopencl", "matplotlib", "hdf5storage", "segyio", "torch")

#: True once the summary has been logged in this process.
_logged = False


class Info(dict):
    """What ``info()`` returns: a plain dict, displayed as the readable summary of ``format_info``."""

    def __repr__(self):
        return format_info(self)


def _git(folder, *args):
    """Output of ``git -C folder <args>``, stripped; raises OSError or SubprocessError on failure."""
    return subprocess.run(["git", "-C", folder, *args], capture_output=True, text=True, timeout=5,
                          check=True).stdout.strip()


def _commit(folder):
    """Short git commit of the sources in ``folder`` (with ``-dirty`` if they have local changes).

    None when ``folder`` is not the ``src/PyFWI`` folder of a git checkout, e.g. for an installed
    wheel (a virtual environment inside some other repository must not report that commit).
    """
    try:
        top = _git(folder, "rev-parse", "--show-toplevel")
        if os.path.realpath(os.path.join(top, "src", "PyFWI")) != os.path.realpath(folder):
            return None
        commit = _git(folder, "rev-parse", "--short", "HEAD")
        return commit + ("-dirty" if _git(folder, "status", "--porcelain", "--", ".") else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _version(distribution):
    """Installed version of a distribution, or None if it is not installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_type(bits):
    """"GPU", "CPU", "accelerator" or "other", from the type bit field of an OpenCL device."""
    import pyopencl as cl
    for name, flag in (("GPU", cl.device_type.GPU), ("CPU", cl.device_type.CPU),
                       ("accelerator", cl.device_type.ACCELERATOR)):
        if bits & flag:
            return name
    return "other"


def _opencl(platform_choice, device_choice):
    """``(platforms, selected device, error)``: every OpenCL platform with its devices, and the choice.

    Works without any OpenCL platform or without PyOpenCL: the error says what is missing.
    """
    try:
        import pyopencl as cl

        from PyFWI._opencl_device import choose_devices
        platforms = [{"name": p.name, "version": p.version,
                      "devices": [{"name": d.name, "type": _device_type(d.type),
                                   "memory_gb": round(d.global_mem_size / 2**30, 1)} for d in p.get_devices()]}
                     for p in cl.get_platforms()]
        selected = choose_devices(platform_choice, device_choice)
    except Exception as error:  # noqa: BLE001 (any failure is reported, never raised)
        return [], None, f"{type(error).__name__}: {error}"
    return platforms, ", ".join(f"{d.name} on {d.platform.name}" for d in selected), None


def info(platform=None, device=None):
    """Summary of this PyFWI: origin, commit, backend, numeric mode, OpenCL devices and versions.

    Parameters
    ----------
    platform, device : int or str, optional
        As ``inpa['platform']`` and ``inpa['device']`` of a ``WavePropagator``: the summary says
        which device such a propagator would use. By default, what a propagator without them uses
        (``PYOPENCL_CTX``, else the first device of the first platform).

    Returns
    -------
    Info
        A dict with the keys ``pyfwi``, ``origin``, ``source``, ``commit``, ``backend``,
        ``numeric_mode``, ``device``, ``opencl_platforms``, ``opencl_error`` and ``versions``,
        displayed as a readable summary (``print(PyFWI.info())``).
    """
    from PyFWI.version import version
    folder = os.path.dirname(os.path.abspath(__file__))
    platforms, selected, error = _opencl(platform, device)
    versions = {"python": platform_version()}
    versions.update({name: _version(name) for name in LIBRARIES})
    return Info(pyfwi=version, origin=ORIGIN, source=folder, commit=_commit(folder), backend=BACKEND,
                numeric_mode=NUMERIC_MODE, device=selected, opencl_platforms=platforms, opencl_error=error,
                versions=versions)


def platform_version():
    """Python version and implementation, e.g. ``"3.12.15 (CPython)"``."""
    return f"{python_platform.python_version()} ({python_platform.python_implementation()})"


def format_info(summary):
    """The readable text of a summary returned by ``info()``."""
    lines = [f"PyFWI {summary['pyfwi']}: {summary['origin']}",
             f"  source        {summary['source']}"
             + (f" (git commit {summary['commit']})" if summary["commit"] else " (not a git checkout)"),
             f"  backend       {summary['backend']}",
             f"  numeric mode  {summary['numeric_mode']} (results of PyFWI 0.1.10)",
             f"  device        {summary['device'] or 'none: ' + str(summary['opencl_error'])}"]
    for i, p in enumerate(summary["opencl_platforms"]):
        devices = "; ".join(f"[{j}] {d['name']} ({d['type']}, {d['memory_gb']} GB)" for j, d in enumerate(p["devices"]))
        lines.append(f"  platform [{i}]  {p['name']}: {devices}")
    versions = ", ".join(f"{name} {version or 'not installed'}" for name, version in summary["versions"].items())
    lines.append(f"  versions      {versions}")
    return "\n".join(lines)


def log_first_run(devices):
    """Log the summary at INFO level, once per process, for the first run while INFO is enabled.

    Parameters
    ----------
    devices : list of pyopencl.Device
        The devices of the propagator that runs, reported instead of the default choice.
    """
    global _logged
    if _logged or not logger.isEnabledFor(logging.INFO):
        return
    summary = info()
    summary["device"] = ", ".join(f"{d.name} on {d.platform.name}" for d in devices)
    logger.info("first propagation of this process\n%s", format_info(summary))
    _logged = True


if __name__ == "__main__":  # python -m PyFWI._info
    print(info())
