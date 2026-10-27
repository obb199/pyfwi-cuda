"""Shared code of the notebooks that compare the CUDA and OpenCL engines of PyFWI on one GPU.

The notebooks of this folder run the same seismic calculation on the two engines of PyFWI-CUDA and show
the difference (see ``README.md``). What they have in common is here, so that each notebook shows only its
question: the model, the acquisition, how a propagator is put on an engine, how a run is timed, the colours
of the figures.

Engines
-------
An *engine* is a pair (backend, numeric mode). PyFWI chooses both for every propagator from two entries of
``inpa``, which win over the environment variables ``PYFWI_BACKEND`` and ``PYFWI_NUMERICS``:

* ``inpa["backend"]``: ``"opencl"`` (the engine of PyFWI 0.1.10) or ``"cuda:strict"`` (CUDA, or an error
  that says why it cannot run: a comparison must never fall back to OpenCL without saying so);
* ``inpa["numerics"]``: ``"legacy"`` (the arithmetic of PyFWI 0.1.10, bugs included: the default of OpenCL)
  or ``"fixed"`` (the corrected arithmetic: the default of CUDA).

So one Python process, one notebook, can run all four engines, one propagator each, on the same GPU.

Used by
-------
The notebooks of this folder, which import it as ``comparison`` (they run from this folder). Tested by
``test/test_notebooks.py``, which runs every notebook in a reduced size.
"""
import collections
import contextlib
import gc
import os
import statistics
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

import PyFWI.acquisition as acq
import PyFWI.fwi_tools as tools
import PyFWI.processing as process
import PyFWI.wave_propagation as wave

# PyOpenCL repeats every remark of the NVIDIA OpenCL compiler about the kernels of PyFWI 0.1.10 as a
# warning; they are about unused variables and say nothing about the comparison.
with contextlib.suppress(ImportError, AttributeError):
    import pyopencl

    warnings.filterwarnings("ignore", category=pyopencl.CompilerWarning)

#: An engine: ``backend`` and ``numerics`` as ``inpa`` takes them.
Engine = collections.namedtuple("Engine", "backend numerics")

OPENCL_LEGACY = Engine("opencl", "legacy")
OPENCL_FIXED = Engine("opencl", "fixed")
CUDA_LEGACY = Engine("cuda:strict", "legacy")
CUDA_FIXED = Engine("cuda:strict", "fixed")

#: The four engines, in the order of the tables and figures.
ENGINES = (OPENCL_LEGACY, OPENCL_FIXED, CUDA_LEGACY, CUDA_FIXED)

#: What runs when nothing is set: OpenCL in the legacy mode, CUDA in the fixed mode.
DEFAULTS = (OPENCL_LEGACY, CUDA_FIXED)

#: Colours of the figures: the hue says the backend, the line (solid, dashed) the numeric mode.
OPENCL_COLOUR, CUDA_COLOUR = "#eb6834", "#2a78d6"
INK, INK_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def name(engine):
    """``"OpenCL, legacy"``, ``"CUDA, fixed"``...: the engine as the tables and figures write it."""
    return f"{'CUDA' if engine.backend.startswith('cuda') else 'OpenCL'}, {engine.numerics}"


def colour(engine):
    """The colour of an engine in the figures: orange for OpenCL, blue for CUDA."""
    return CUDA_COLOUR if engine.backend.startswith("cuda") else OPENCL_COLOUR


def line_style(engine):
    """Solid for the fixed mode, dashed for the legacy mode."""
    return "-" if engine.numerics == "fixed" else "--"


def style(ax, grid_axis="y"):
    """Quiet axes: a light grid on one axis, no top and right frame, so that the data stand out."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------- the seismic problem

def inclusion_model(nz, nx, smooth=0):
    """An elastic model with a faster circular inclusion, as ``ModelGenerator("louboutin")`` of PyFWI.

    Parameters
    ----------
    nz, nx : int
        Grid points in depth (rows) and along x (columns).
    smooth : float, optional
        0 for the true model; otherwise vp is smoothed by a Gaussian filter of ``10 * smooth`` grid points,
        a starting model that does not know where the inclusion is.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``"vp"`` and ``"vs"`` in m/s, ``"rho"`` from Gardner's relation, ``float32`` arrays ``(nz, nx)``:
        vp is 2500 m/s with 3000 m/s in a circle of radius ``nz // 10`` at the centre, vs = vp / √3.
    """
    z, x = np.mgrid[:nz, :nx]
    inside = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp = np.where(inside, 3000.0, 2500.0)
    if smooth:
        vp = gaussian_filter(vp, 10 * smooth)
    model = {"vp": vp, "vs": vp / np.sqrt(3.0), "rho": 310.0 * vp ** 0.25}
    return {key: values.astype(np.float32) for key, values in model.items()}


def survey(nz, nx, ns, t, dh=7.0, acq_type=1, amplitude=1.0):
    """The acquisition of a seismic survey over an ``nz`` × ``nx`` model, without an engine.

    Parameters
    ----------
    nz, nx : int
        Grid points of the model.
    ns : int
        Number of sources (shots).
    t : float
        Length of the recording (s).
    dh : float, optional
        Grid spacing (m).
    acq_type : int, optional
        1: sources and receivers near the surface; 0: crosswell, sources in a well on the left, receivers
        in a well on the right.
    amplitude : float, optional
        Factor of the source wavelet (a Ricker wavelet of 15 Hz). The optimisers of the legacy mode stop on
        absolute tolerances, so an inversion needs a stronger source there (see ``example/scripts/fwi.py``).

    Returns
    -------
    dict
        ``inpa`` (the input parameters of PyFWI, without ``backend`` and ``numerics``), ``src``,
        ``rec_loc``, ``n_well_rec`` and ``shape``: the arguments of ``WavePropagator`` and ``FWI``; and
        ``src_loc``, the positions of the sources (m), for the figures.
    """
    inpa = {"ns": ns, "sdo": 4, "fdom": 15, "dh": dh, "dt": 0.004, "acq_type": acq_type, "t": t, "npml": 20,
            "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh}
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(ns, dh, dh * nx, dh * nz, dh, inpa["sdo"],
                                                         acq_type=acq_type)
    if acq_type == 1:
        # One grid row above the receivers: a source on a receiver cell makes the OpenCL kernels of
        # PyFWI 0.1.10 race, and the last digits of the results change from run to run.
        src_loc[:, 1] -= dh
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])
    src.w = src.w * np.float32(amplitude)
    return {"inpa": inpa, "src": src, "rec_loc": rec_loc, "n_well_rec": n_well_rec, "shape": (nz, nx),
            "src_loc": src_loc}


def on_engine(inpa, engine):
    """A copy of ``inpa`` that asks for ``engine``: the only change between two engines."""
    return dict(inpa, backend=engine.backend, numerics=engine.numerics)


def propagator(problem, engine, chpr=20):
    """A ``WavePropagator`` of ``problem`` (``survey``) on ``engine``.

    ``chpr`` is the share (%) of the time steps whose wave field the forward modelling keeps for the
    gradient (the *checkpoints*); 0 when no gradient follows.
    """
    return wave.WavePropagator(on_engine(problem["inpa"], engine), problem["src"], problem["rec_loc"],
                               problem["shape"], components=0, n_well_rec=problem["n_well_rec"], chpr=chpr)


def where(W):
    """What runs a propagator: ``"cuda device 0 (cuda:strict asked by inpa['backend']), fixed mode, on
    NVIDIA GeForce RTX 3060 (CUDA device 0)"``."""
    return f"{W.backend_selection.describe()}, {W.numerics} mode, on {W._backend.description}"


def check_engines():
    """Put a tiny propagator on every engine and print where it runs.

    Raises
    ------
    RuntimeError
        CUDA cannot run here: these notebooks compare the two backends on an NVIDIA GPU, so they need
        CuPy (``pip install ".[cuda13]"``) and an NVIDIA driver. The message says what is missing.
    """
    problem = survey(20, 20, 1, 0.05)
    for engine in ENGINES:
        try:
            W = propagator(problem, engine, chpr=0)
        except Exception as error:
            raise RuntimeError(f"{name(engine)} cannot run here: {error}") from error
        print(f"{name(engine):15s} -> {where(W)}")


# ---------------------------------------------------------------------------- timing

def seconds(function, *args, **kwargs):
    """``(result, seconds)``: the result of ``function(*args, **kwargs)`` and its wall-clock time.

    PyFWI returns NumPy arrays, which it copies from the GPU once the GPU has finished: when a call
    returns, its work is done, so the wall clock is the time of the calculation.
    """
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def adjoint_source(W, predicted, observed):
    """The misfit (L2) of ``predicted`` against ``observed`` and its adjoint source, the input of
    ``W.gradient``."""
    misfit, source = tools.cost_seismic(process.prepare_residual(predicted, 1), process.prepare_residual(observed, 1),
                                        fun=tools.CostFunction("l2"))
    return misfit, source


def time_forward_and_gradient(problem, engine, true, start, repeats=3):
    """Median times of one forward modelling and one gradient of ``problem`` on ``engine``.

    The data of ``true`` are the observed data; the forward modelling and the gradient are those of
    ``start``, as in one iteration of an inversion. One run comes first, untimed: it compiles the kernels
    and warms the GPU up (a GeForce card raises its clock under load).

    Returns
    -------
    dict
        ``forward`` and ``gradient`` (s, the median of ``repeats`` runs), ``steps`` (time steps of one
        shot), ``where`` (``where``), ``checkpoint_mb`` (``checkpoint_mb``), and ``last``, the results of the
        last run: ``{"seismograms": ..., "gradient": {"vp", "vs", "rho"}}``.
    """
    W = propagator(problem, engine)
    observed = W.forward_modeling(true, show=False)
    forwards, gradients = [], []
    for run in range(repeats + 1):
        predicted, forward = seconds(W.forward_modeling, start, show=False)
        _, source = adjoint_source(W, predicted, observed)
        result, gradient = seconds(W.gradient, source, show=False)
        if run:  # the first run is the warm-up
            forwards.append(forward)
            gradients.append(gradient)
    return {"forward": statistics.median(forwards), "gradient": statistics.median(gradients), "steps": int(W.nt),
            "where": where(W), "checkpoint_mb": checkpoint_mb(W),
            "last": {"seismograms": predicted, "gradient": result}}


# ---------------------------------------------------------------------------- comparing results

def relative_difference(a, b):
    """‖a − b‖ / ‖b‖ over every component of two results of PyFWI (dicts of arrays): 0 when they are equal,
    1e-6 when they agree to six digits."""
    a = np.concatenate([np.ravel(a[key]) for key in sorted(b)]).astype(np.float64)
    b = np.concatenate([np.ravel(b[key]) for key in sorted(b)]).astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------- memory

@contextlib.contextmanager
def gpu_memory_peak():
    """Measure the GPU memory that the block takes on CUDA: yields a dict, filled with ``mb`` at the end.

    PyFWI allocates its CUDA buffers from the memory pool of CuPy, which keeps every block it has
    allocated until it is told to free them. Freeing the free blocks first, the size of the pool at the end
    of the block, less what was in use before, is the peak of the block. Only for CUDA: OpenCL buffers do
    not go through CuPy.
    """
    import cupy

    gc.collect()  # the buffers of propagators no longer referenced go back to the pool
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    before = pool.used_bytes()
    measured = {}
    try:
        yield measured
    finally:
        measured["mb"] = (pool.total_bytes() - before) / 2 ** 20


@contextlib.contextmanager
def cuda_options(boundary_saving=False, fp16=False):
    """Turn on the options only CUDA has, for the propagators created in the block.

    They are environment variables (``PYFWI_BOUNDARY_SAVING``, ``PYFWI_FP16``), read when a propagator is
    created; the previous values come back at the end of the block.
    """
    names = {"PYFWI_BOUNDARY_SAVING": boundary_saving, "PYFWI_FP16": fp16}
    previous = {key: os.environ.get(key) for key in names}
    os.environ.update({key: "1" if on else "0" for key, on in names.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def checkpoint_mb(W):
    """The memory (MB) of the checkpoints of the last forward modelling of ``W``, for all its shots: five
    wave-field components (vx, vz, τxx, τzz, τxz) of the whole grid, PML included, at every checkpoint."""
    return 5 * int(W.nchp) * int(W.tnz) * int(W.tnx) * int(W.field_bytes) * int(W.ns) / 2 ** 20
