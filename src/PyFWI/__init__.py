"""PyFWI: elastic full-waveform inversion and time-lapse FWI on GPUs (fork pyfwi-cuda).

Importing the package only defines its version and ``info``: the modules (``PyFWI.wave_propagation``,
``PyFWI.fwi``, ...) are imported explicitly, as in PyFWI 0.1.10, and nothing heavy (NumPy, PyOpenCL)
is loaded until they are.
"""
from PyFWI.version import version as __version__

__all__ = ['__version__', 'info']


def info(platform=None, device=None):
    """Summary of this PyFWI: origin and git commit, backend, numeric mode, OpenCL devices and versions.

    ``print(PyFWI.info())`` shows it; the result is also a dict, for programs. ``platform`` and
    ``device`` are those of ``inpa``: the summary then says which device such a propagator uses.
    See ``PyFWI._info.info`` (added in step 1.8 of the fork).
    """
    from PyFWI._info import info as _info
    return _info(platform, device)
