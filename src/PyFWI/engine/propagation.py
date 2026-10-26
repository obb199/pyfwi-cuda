"""The time loops of the elastic propagation, on any backend (roadmap step 2.1).

``PropagationEngine`` runs the kernels of PyFWI (the OpenCL C files of 0.1.10) in the same order
as 0.1.10 did, through a backend (``PyFWI.backends``): it never calls a GPU library itself. It
owns everything that lives on the device or follows the time loop:

* the device buffers (fields, adjoint fields, model, absorbing boundary, seismograms, residuals,
  gradients), each inside an allocation with margins (step 1.7);
* the kernels, retrieved once, with their constant arguments set once per run (step 1.3);
* the forward loop: source injection (every ``dt_scale`` steps in the ``legacy`` numeric mode,
  at every step in the ``fixed`` one, see ``internal_source``), the four field updates, the
  checkpoints of the forward field (stored as ``(ns, nchp, tnz, tnx)`` per component, step 1.5),
  and the seismograms read at the end of each shot (step 1.4);
* the gradient loop: the forward field rebuilt backwards from the checkpoints, the residual
  injected from a copy on the device (step 1.4), the adjoint updates and the gradient kernel (in
  the ``fixed`` mode in the order of the exact adjoint of the forward step, see ``_adjoint_step``);
* with the native CUDA kernels (Phase 3), the shots in batches: several shots run by the same
  launches, one per slot of the buffers (step 3.9, ``shot_batch``).

``PyFWI.wave_propagation.WavePropagator`` is the public face: it prepares the inputs (time steps,
geometry, absorbing boundary, model), turns the results into seismic sections and gradients, and
shows the engine's attributes as its own (``W.vx_b``, ``W.knl``...), as in 0.1.10.

Used by
-------
``PyFWI.wave_propagation.WavePreparation``. Tested by ``test/test_engine.py`` (with a fake NumPy
backend, without a device) and by every test that runs a propagation.
"""
import os
from types import SimpleNamespace

import numpy as np
import scipy.signal

#: The five recorded components, in the order of the seismogram and residual buffers.
COMPONENTS = ("vx", "vz", "taux", "tauz", "tauxz")

#: Names of the kernels that the engine launches (in elastic.cl; injSrc and Adj_injSrc also in
#: elastic_crosswell.cl and elastic_surface.cl, which serve the crosswell and surface acquisitions).
KERNELS = ("MakeAllZero", "MakeGradZero", "injSrc", "Adj_injSrc", "update_velx", "update_velz",
           "update_tauz", "update_tauxz", "Adj_update_tau", "Adj_update_v", "Grad")

#: The kernels of the native CUDA program (``PyFWI/kernels/elastic.cu``, Phase 3) when the sources
#: include one: every kernel but the injection ones, which still come from the OpenCL files.
NATIVE_KERNELS = tuple(name for name in KERNELS if name not in ("injSrc", "Adj_injSrc"))

#: The fused kernels of the native program, which the time loops run instead of the separate ones
#: (step 3.5): ``update_v`` (update_velx + update_velz), ``update_tau`` (update_tauz + update_tauxz)
#: and ``grad_adjoint_v`` (Grad + Adj_update_v).
FUSED_KERNELS = ("update_v", "update_tau", "grad_adjoint_v")

#: Longest CUDA graph of the native time loops, in steps (step 3.7): a run between two checkpoints is
#: replayed in graphs of at most this many steps (a graph per length, recorded once per run).
MAX_GRAPH_STEPS = 50

#: Step 3.9: the native loops run the shots in batches of at most this many grid cells (``shot_batch``),
#: each shot in a slot along the z dimension of the launches: fewer launches, and fewer blocks left over
#: at the end of each. Measured on the RTX 3060 (fixed mode, forward / gradient against one shot at a
#: time): 4 shots of 140 x 140 cells in one batch -42 % / -4 %, of 440 x 440 -11 % / -7 %, 2 shots of
#: 640 x 790 -3 % / -2 %. Above 2^20 cells the gain is small and the memory of the fields grows.
BATCH_CELLS = 1 << 20

#: The buffers with one field per slot of a batch (step 3.9): the forward and adjoint fields, and the
#: gradients and their preconditioners, which ``gradient`` sums into slot 0 at its end.
FIELD_BUFFERS = tuple(f"{par}_b" for par in COMPONENTS) + tuple(f"a{par}_b" for par in COMPONENTS)
GRADIENT_BUFFERS = ("Gmu_b", "Glam_b", "Grho_b", "g_mu_precond_b", "g_lam_precond_b", "g_rho_precond_b")

#: Value of the margins around every device buffer (step 1.7): what the out-of-bounds reads of the
#: kernels of 0.1.10 return. 0 reproduces the golden masters, which 0.1.10 computed inside zero
#: guard bands; the tests set NaN or 1.0 to see where the kernels read and write.
MARGIN_FILL = 0.0

#: Smallest margin before and after every device buffer, in bytes: the size of the guard bands
#: with which the golden masters of 0.1.10 were made (test/oob_guard.py).
MIN_MARGIN_BYTES = 1 << 16


#: The windowed-sinc filter with which the fixed numeric mode resamples the source to the internal
#: step (``internal_source``): its half-length in recorded samples and the beta of its Kaiser
#: window. Measured for ``dt_scale`` 2 to 17: the wavelet is kept within 6e-6 up to 80 % of the
#: recorded Nyquist frequency, and its images above it are attenuated by more than 100 dB.
SOURCE_FILTER_HALF_LENGTH = 16
SOURCE_FILTER_BETA = 10.0


def internal_source(source, nt, dt_scale):
    """The source to add at every internal step, in the ``fixed`` numeric mode (roadmap step 2.4).

    The source wavelet is sampled at the user's ``dt``, ``dt_scale`` internal steps per sample.
    0.1.10 adds sample ``k`` at internal step ``k * dt_scale`` only: a train of impulses, whose
    spectrum repeats the wavelet around every multiple of the recorded sampling frequency (250 Hz
    for ``dt`` = 4 ms). At grid spacings of a few metres these images excite a grid-scale mode
    that never leaves the model (a known bug). The ``fixed`` mode resamples the wavelet to the
    internal step instead, by band-limited interpolation (a windowed-sinc filter, which keeps the
    wavelet and removes the images), and adds the result at every internal step, divided by
    ``dt_scale``: every recorded interval receives the same amount of source as in 0.1.10, now
    spread over its steps.

    With ``dt_scale`` = 1 (a user ``dt`` of 0.6 ms or less) the samples are returned unchanged,
    as both modes then inject the same thing. A wavelet that does not start smoothly (for example
    ``Source.delta``) is band-limited too: the part of its interpolation before time 0 is lost.

    Parameters
    ----------
    source : callable
        ``source(k)`` returns the five samples (vx, vz, taux, tauz, tauxz) of recorded sample
        ``k``, zero after the end of the wavelet (``PyFWI.acquisition.Source``).
    nt : int
        Number of internal time steps.
    dt_scale : float
        Internal steps per recorded sample (an integer value).

    Returns
    -------
    numpy.ndarray
        float32, ``(nt, 5)``: row ``n`` is what injSrc adds to the fields at internal step ``n``.

    Used by
    -------
    ``PropagationEngine.forward``, in the ``fixed`` mode, in two parts (``recorded_source``, then
    ``resampled_source``), so that it resamples only a wavelet that changed.
    """
    return resampled_source(recorded_source(source, nt, dt_scale), nt, dt_scale)


def recorded_source(source, nt, dt_scale):
    """The recorded samples of the source that ``internal_source`` resamples, ``(samples, 5)`` float64.

    Those that the internal steps 0 to ``nt - 1`` need: up to the last step plus the half-length of
    the filter. Before sample 0 the source is zero (``resample_poly`` pads with zeros).
    """
    recorded = (nt - 1) // int(dt_scale) + SOURCE_FILTER_HALF_LENGTH + 2
    return np.array([source(k) for k in range(recorded)], np.float64)


def resampled_source(samples, nt, dt_scale):
    """``internal_source`` from the samples of ``recorded_source``: float32, ``(nt, 5)``."""
    up = int(dt_scale)
    if up == 1:
        return samples[:nt].astype(np.float32)
    # Low-pass at the recorded Nyquist frequency, designed at the internal rate. resample_poly
    # inserts up - 1 zeros between the samples, filters, multiplies by up and removes the delay of
    # the filter: the result passes through the samples at the recorded steps.
    taps = scipy.signal.firwin(2 * SOURCE_FILTER_HALF_LENGTH * up + 1, 1.0 / up,
                               window=("kaiser", SOURCE_FILTER_BETA))
    resampled = scipy.signal.resample_poly(samples, up, 1, axis=0, window=taps)
    return (resampled[:nt] / up).astype(np.float32)


def receiver_table(acq_type, tnz, tnx, nr, dxr, n_surface, n_well, surface, left, right):
    """The cell of every receiver, as the injection kernels of the OpenCL files find it (step 3.4).

    The OpenCL kernels injSrc and Adj_injSrc run on the whole grid, and the work item of each
    receiver recognises itself from the acquisition constants of the ``#define`` block. The native
    CUDA kernels run one thread per receiver instead, which reads its cell from this table. The loops
    below are those of the kernels of the fixed mode (bounded receiver loops, step 2.4): the work
    items are the rows ``i < tnz`` and columns ``j < tnx`` of the grid.

    Parameters
    ----------
    acq_type : int
        0 crosswell (``elastic_crosswell.cl``: every receiver in the right well), 1 surface
        (``elastic_surface.cl``), 2 both wells and the surface (``elastic.cl``: left well, surface,
        right well, in that order).
    tnz, tnx : int
        Grid size, absorbing layer included.
    nr : int
        Number of receivers (``Nr``).
    dxr : int
        Receiver spacing in cells.
    n_surface, n_well : int
        Receivers on the surface (``n_main_rec``) and in each well (``n_extera_rec``).
    surface, left, right : tuple of int
        ``(const, var)`` of the surface line (``rec_surface_const``: its row; ``rec_surface_var``:
        the column of its first receiver), of the left well and of the right well
        (``rec_top_*_const``: the column; ``rec_top_*_var``: the row of the first receiver).

    Returns
    -------
    cells, first, following : numpy.ndarray
        int32 arrays of ``nr`` values: the flat index of the cell of receiver ``r`` (-1 when no work
        item records it); 1 when ``r`` is the first receiver of its cell, else 0; the next receiver
        on the same cell after ``r``, or -1. With these, the residual of receivers that share a cell
        is added by one thread, in a fixed order (``inject_residual`` of ``elastic.cu``).
    """
    cells = np.full(nr, -1, np.int64)

    def well(start, const, var, count):
        for i in range(0, tnz, dxr):
            if i // dxr < count:
                cells[start + i // dxr] = (i + var) * tnx + const

    def surface_line(start):
        for j in range(0, tnx, dxr):
            if j // dxr < n_surface:
                cells[start + j // dxr] = surface[0] * tnx + j + surface[1]

    if acq_type == 0:
        well(0, right[0], right[1], nr)
    elif acq_type == 1:
        surface_line(0)
    else:
        well(0, left[0], left[1], n_well)
        surface_line(n_well)
        well(n_well + n_surface, right[0], right[1], n_well)
    first = np.zeros(nr, np.int32)
    following = np.full(nr, -1, np.int32)
    last = {}  # cell -> the last receiver seen on it
    for r, cell in enumerate(cells):
        if cell < 0:
            continue
        if cell in last:
            following[last[cell]] = r
        else:
            first[r] = 1
        last[cell] = r
    return cells.astype(np.int32), first, following


def undamped_rectangle(vdx_pml, vdz_pml):
    """Rows ``z0..z1`` and columns ``x0..x1`` (inclusive) where both damping profiles are zero (step 3.10).

    The native kernels do not read the profiles there (``damping`` of elastic.cu): the damping of
    those cells is 0 as read, so nothing changes. PyFWI's profiles are separable (``vdx_pml`` depends
    on the column only, ``vdz_pml`` on the row): the rectangle spans the columns where ``vdx_pml`` is
    zero in every row and the rows where ``vdz_pml`` is zero in every column, checked to be zero on
    both before it is used.

    Parameters
    ----------
    vdx_pml, vdz_pml : numpy.ndarray
        The damping profiles, ``(tnz, tnx)``.

    Returns
    -------
    tuple of numpy.int32
        ``(z0, z1, x0, x1)``; ``(0, -1, 0, -1)``, an empty rectangle, when there is none.
    """
    vdx, vdz = np.asarray(vdx_pml), np.asarray(vdz_pml)
    columns = np.flatnonzero(~np.any(vdx != 0, axis=0))
    rows = np.flatnonzero(~np.any(vdz != 0, axis=1))
    empty = tuple(np.int32(v) for v in (0, -1, 0, -1))
    if not (columns.size and rows.size):
        return empty
    z0, z1, x0, x1 = int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())
    inside = (slice(z0, z1 + 1), slice(x0, x1 + 1))
    if np.any(vdx[inside] != 0) or np.any(vdz[inside] != 0):  # not one rectangle: no savings, no risk
        return empty
    return tuple(np.int32(v) for v in (z0, z1, x0, x1))


def band_cells(rectangle, width, tnx):
    """The cells stored by boundary saving (step 4.2): those of ``rectangle`` within ``width`` of its edge.

    Parameters
    ----------
    rectangle : tuple of int
        ``(z0, z1, x0, x1)``, rows and columns (inclusive) where the absorbing layer does not damp
        (``undamped_rectangle``): undoing a time step is exact there.
    width : int
        Width of the band: the half order ``sdo`` of the finite differences, the farthest cell a
        stencil reads.
    tnx : int
        Grid width, for the flat indices.

    Returns
    -------
    numpy.ndarray
        int32 flat indices ``row * tnx + column``, row by row: the rows of the band at the top and the
        bottom of the rectangle, and its columns at the left and the right in the rows between. Empty
        for an empty rectangle.

    Used by ``PropagationEngine.set_pml`` with boundary saving: every cell of the rectangle off the band
    reads only cells of the rectangle, so writing back the stored band makes the rebuilt field exact.
    """
    z0, z1, x0, x1 = (int(v) for v in rectangle)
    if z1 < z0 or x1 < x0:
        return np.zeros(0, np.int32)
    rows, columns = np.mgrid[z0:z1 + 1, x0:x1 + 1]
    inner = (rows >= z0 + width) & (rows <= z1 - width) & (columns >= x0 + width) & (columns <= x1 - width)
    return (rows[~inner] * int(tnx) + columns[~inner]).astype(np.int32)


def shot_batch(ns, tnz, tnx, environ=None):
    """Number of shots that the native loops run at once (step 3.9).

    ``PYFWI_CUDA_BATCH`` when set (a positive integer), never more than ``ns``. Else as many shots as
    fit in ``BATCH_CELLS`` cells (at least one), balanced: the fewest batches of that size, with as few
    shots each as they need (8 shots of at most 5: 2 batches of 4, not 5 and 3, whose last batch would
    leave the GPU emptier). The results do not depend on it but for the order in which the gradients of
    the shots are added (``sum_slots`` of elastic.cu): the seismograms are the same bits, the gradients
    agree to float32 rounding.

    Parameters
    ----------
    ns : int
        Number of shots.
    tnz, tnx : int
        Grid size, absorbing layer included.
    environ : mapping, optional
        The environment variables (``os.environ`` when None; the tests pass their own).

    Returns
    -------
    int
        The batch size, from 1 to ``ns``.

    Used by
    -------
    ``PropagationEngine`` with the native injection of the CUDA backend.
    """
    environ = os.environ if environ is None else environ
    forced = environ.get("PYFWI_CUDA_BATCH", "").strip()
    if forced:
        if not forced.isdigit() or int(forced) < 1:
            raise ValueError(f"PYFWI_CUDA_BATCH must be a positive integer (shots run at once), not {forced!r}")
        return min(int(ns), int(forced))
    largest = max(1, BATCH_CELLS // (int(tnz) * int(tnx)))
    batches = max(1, -(-int(ns) // largest))  # rounded up
    return max(1, -(-int(ns) // batches))


def checkpoint_steps(chpr, nt):
    """Time steps at which the forward field is stored for the gradient, as in PyFWI 0.1.10.

    Parameters
    ----------
    chpr : float
        Percentage of the ``nt`` time steps to store (``chpr`` of ``WavePropagator``).
    nt : int
        Number of internal time steps.

    Returns
    -------
    numpy.ndarray
        int32 steps, evenly spread from 0 to ``nt - 1``; at least two (``[1, nt - 1]`` when
        ``chpr`` asks for fewer, the case of the default ``chpr=0``).
    """
    chp = np.linspace(0, nt - 1, int(chpr * nt / 100), dtype=np.int32)
    if len(chp) < 2:
        chp = np.array([1, nt - 1])
    return chp


def margin_bytes(tnz, tnx, rec_top_right_var, alignment):
    """Bytes of the margin before and after every buffer, enough for every out-of-bounds access (step 1.7).

    The kernels of 0.1.10 read and write outside their buffers: the stencils up to 4 grid rows
    before and after a field, the unbounded loop of the right well (``acq_type`` 0 and 2) up to
    ``rec_top_right_var`` rows past the end of a field, and up to ``tnz`` values past the end of a
    seismogram or residual. The margin is the largest of these (plus 8 rows or values to spare)
    and of ``MIN_MARGIN_BYTES``, rounded up to ``alignment``.
    """
    needed = max(MIN_MARGIN_BYTES, (int(rec_top_right_var) + 8) * int(tnx) * 4, (int(tnz) + 8) * 4)
    return -(-needed // alignment) * alignment


class PropagationEngine:
    """Forward modelling and gradient of PyFWI on one backend (see the module docstring).

    Parameters
    ----------
    backend : object
        A backend of ``PyFWI.backends`` (``OpenCLBackend``, or a fake one in the tests).
    sources : dict
        Kernel sources with their ``#define`` block: ``"elastic"``, ``"crosswell"`` and ``"surface"``
        (the OpenCL files), and optionally ``"native"`` (the native CUDA kernels of Phase 3, whose
        kernels then replace those of ``NATIVE_KERNELS``; CUDA backend only).
    injection : str
        Source of injSrc and Adj_injSrc: ``"crosswell"`` (``acq_type`` 0), ``"surface"`` (1) or
        ``"elastic"`` (2).
    tnz, tnx : int
        Grid size, absorbing boundary included.
    ns, nr : int
        Numbers of shots and of receivers.
    nt, nt_ext : int
        Numbers of internal time steps and of recorded samples.
    dt_scale : float
        Internal steps per recorded sample (an integer value): the receivers are recorded every
        ``dt_scale`` steps, and in the ``legacy`` mode the source is injected at those steps only.
    chp : numpy.ndarray
        Checkpoint steps (``checkpoint_steps``).
    dxr : numpy.int32
        Receiver spacing in grid cells.
    srcx, srcz : numpy.ndarray
        int32 grid position of every source, absorbing boundary included.
    source : callable
        ``source(t)`` returns the five source samples of recorded sample ``t``
        (``PyFWI.acquisition.Source``).
    rec_top_right_var : int
        First grid row of the right well (sizes the margins).
    numerics : str, optional
        Numeric mode of the kernel sources (``PyFWI.backends.selection.select_numerics``):
        ``"legacy"`` (default) runs the time loops of 0.1.10; ``"fixed"`` injects the source at
        every internal step and the residual at the very step where it was recorded (step 2.4).
    receivers : tuple of numpy.ndarray, optional
        ``receiver_table`` of the acquisition. With a native program, the source and the receivers
        are then handled one thread per receiver (step 3.4) instead of by the OpenCL injection
        kernels on the whole grid.
    batch : int, optional
        With native injection, the shots run at once (step 3.9); ``shot_batch`` when None. Without
        it, the shots run one by one.
    boundary_width : int, optional
        Boundary saving (step 4.2, native kernels only): instead of the checkpoints of ``chp``, the
        fields of a band of this width (the half order ``sdo``) at the edge of the undamped rectangle
        are stored at every step, and the gradient rebuilds the forward field from the last one
        (``band_cells``). None or 0: checkpoints.
    """

    def __init__(self, backend, sources, injection, *, tnz, tnx, ns, nr, nt, nt_ext, dt_scale, chp, dxr,
                 srcx, srcz, source, rec_top_right_var, numerics="legacy", receivers=None, batch=None,
                 boundary_width=None):
        self.backend = backend
        self.tnz, self.tnx, self.ns, self.nr = tnz, tnx, ns, nr
        self.nt, self.nt_ext, self.dt_scale = nt, nt_ext, dt_scale
        #: True in the fixed numeric mode; ``steps_per_sample`` is ``dt_scale`` as an int.
        self.numerics, self.fixed, self.steps_per_sample = numerics, numerics == "fixed", int(dt_scale)
        #: Fixed mode: the source of every internal step (``internal_source``), made by ``forward``
        #: from the recorded samples ``_source_samples``, again only when they change (step 3.11).
        self.source_steps = self._source_samples = None
        self.chp, self.nchp = chp, len(chp)
        self.dxr, self.srcx, self.srcz, self.src = dxr, srcx, srcz, source
        # Global size of every launch: one work item per grid point, absorbing boundary included.
        self.global_size = (int(tnz), int(tnx))

        self.prg_cw = backend.build(sources["crosswell"])
        self.prg = backend.build(sources["elastic"])
        self.prg_surf = backend.build(sources["surface"])
        programs = {"elastic": self.prg, "crosswell": self.prg_cw, "surface": self.prg_surf}
        if injection != "elastic":
            # As in 0.1.10, the program of the acquisition provides injSrc and Adj_injSrc.
            self.prg.injSrc = backend.kernel(programs[injection], "injSrc")
            self.prg.Adj_injSrc = backend.kernel(programs[injection], "Adj_injSrc")
        # The kernel objects, retrieved once and reused at every launch (step 1.3); injSrc and
        # Adj_injSrc are those assigned to self.prg just above.
        self.knl = SimpleNamespace(**{name: backend.kernel(self.prg, name) for name in KERNELS})
        #: The native CUDA program (step 3.2), or None: its kernels have the names and arguments of
        #: those of elastic.cl, so that the time loops launch either set the same way.
        self.prg_native = backend.build_native(sources["native"]) if "native" in sources else None
        if self.prg_native is not None:
            for name in NATIVE_KERNELS + FUSED_KERNELS:
                setattr(self.knl, name, backend.kernel(self.prg_native, name))
        #: Step 3.5: with a native program the loops launch its fused kernels.
        self.fused = self.prg_native is not None
        #: Step 3.4: the native kernels inject and record one thread per receiver, reading the cells
        #: of ``receivers`` (``receiver_table``); launched on a grid of 1 row and ``nr`` columns.
        self.native_injection = self.prg_native is not None and receivers is not None
        #: Step 3.9: the shots run at once by the native loops (``shot_batch``), each in a slot of the
        #: buffers of the fields, adjoint fields, gradients, seismograms and residuals; 1 without
        #: native injection. ``slots`` is the number of shots of the batch being run (the z size of
        #: its launches, ``_size``), ``_last_slot`` the slot of the last shot run.
        self.batch = 1
        if self.native_injection:
            self.batch = shot_batch(ns, tnz, tnx) if batch is None else max(1, min(int(ns), int(batch)))
        self.slots, self._last_slot = 1, 0
        #: Values of one field (``tnz * tnx``), and from the start of a slot to the start of the next
        #: (a field and one row, see ``_slots_host``).
        self._plane, self._stride = int(tnz) * int(tnx), (int(tnz) + 1) * int(tnx)
        if self.native_injection:
            for name in ("inject_and_record", "inject_residual", "advance", "sum_slots", "copy_checkpoint"):
                setattr(self.knl, name, backend.kernel(self.prg_native, name))
            # Read-only tables: no margins (they would hold the float MARGIN_FILL in int32 buffers).
            self.receiver_cells_b, self.receiver_first_b, self.receiver_following_b = (
                self._table(table) for table in receivers)
            self.receiver_size = (1, max(int(nr), 1))
            #: Step 3.6: the internal step on the device (``advance`` moves it), the cell of the source
            #: of the shot (of every slot of a batch, step 3.9), and the source of every step (nt x 4:
            #: vx, vz, taux, tauz), read by the kernels, so that every launch of a step has the same
            #: arguments.
            self.step_b = self._table(np.zeros(1, np.int32))
            self.source_cell_b = self._table(np.zeros(self.batch, np.int32))
            self.source_steps_b = backend.allocate_empty(int(nt) * 4 * 4)
        #: Step 4.2: boundary saving (``boundary_width`` cells), with the native kernels only. Its kernels,
        #: the shot of every slot (``band_shot_b``), and only the last field as checkpoint: the gradient
        #: starts from it. The band and its store come with the absorbing layer (``set_pml``).
        self.boundary_width = int(boundary_width or 0)
        if self.boundary_width:
            if not self.native_injection:
                raise ValueError("boundary saving needs the native CUDA kernels: the CUDA backend in the fixed numeric mode")
            for name in ("save_band", "remove_source"):
                setattr(self.knl, name, backend.kernel(self.prg_native, name))
            # Two kernel objects, two argument lists: the stresses, then the velocities of the band.
            self.knl.restore_stresses = backend.kernel(self.prg_native, "restore_band")
            self.knl.restore_velocities = backend.kernel(self.prg_native, "restore_band")
            self.band_shot_b = self._table(np.zeros(self.batch, np.int32))
            self.chp, self.nchp = np.array([int(nt) - 1], np.int32), 1
        self.band_b, self.n_band, self._band_store, self._band_cells = None, 0, None, None
        #: Step 3.7: the native time loops replay CUDA graphs of several steps (backend ``capture``),
        #: unless ``PYFWI_CUDA_GRAPHS=0`` or a ``show=True`` run observes every step. The graphs are kept
        #: from run to run while the buffers they use stay (``_set_buffer``, step 3.11).
        self.graphs = (self.native_injection and getattr(backend, "capture", None) is not None
                       and os.environ.get("PYFWI_CUDA_GRAPHS", "1") != "0")
        self._graph_cache = {}

        self.margin_bytes = margin_bytes(tnz, tnx, rec_top_right_var, backend.alignment)
        # The allocation (margins included) of every device buffer, by attribute name.
        self._allocations = {}

        res = np.zeros(int(nr), np.float32)
        for par in COMPONENTS:
            self._set_buffer(f"res_{par}_b", res)
        # The residual of one shot (of every slot of a batch), every time sample, per component
        # (residual_to_device); created at the first gradient and reused while its size does not
        # change. ``residual_rows``: the time samples of one shot in it.
        self.res_shot_b, self.residual_rows = {}, 0

        field = self._slots_host()  # one field per slot of a batch (step 3.9)
        for par in COMPONENTS:
            self._set_buffer(f"{par}_b", field)

        seismogram = np.zeros((self.batch, nt_ext, nr), np.float32)  # one per slot (step 3.9)
        for par in COMPONENTS:
            self._set_buffer(f"seismogramid_{par}_b", seismogram)

        #: The seismograms of the last forward modelling, ``{component: (nt_ext, nr, ns)}``.
        self.seismogram = {par: np.zeros((nt_ext, nr, ns), np.float32) for par in COMPONENTS}

        #: The checkpoints of the forward field. Those of the first ``_device_shots`` shots stay on the
        #: device, as many as the backend has room for (step 3.8: ``checkpoint_budget``), in one buffer
        #: of ``(5 components, _device_shots, nchp, tnz, tnx)`` values, stored and restored by device
        #: copies; those of the other shots on the host (see ``checkpoints``).
        per_shot = 5 * self.nchp * self._plane * 4  # bytes of the checkpoints of one shot
        budget = backend.checkpoint_budget() if hasattr(backend, "checkpoint_budget") else 0
        self._device_shots = min(int(ns), budget // per_shot)
        self._device_checkpoints = backend.allocate_empty(self._device_shots * per_shot) if self._device_shots else None
        self._host_checkpoints = None if self._device_shots == int(ns) else {
            par: np.zeros((ns, self.nchp, tnz, tnx), np.float32) for par in COMPONENTS}

    @property
    def checkpoints(self):
        """The checkpoints on the host, ``{component: (ns, nchp, tnz, tnx)}`` (``WavePropagator.W``).

        The field of one shot at one checkpoint is a contiguous block, copied straight from and to
        the device without blocking (step 1.5). When checkpoints are on the device (step 3.8), the
        first access copies them here and the host copy becomes the storage of every shot, which the
        next runs use: writing into it changes the gradient, as in 0.1.10.
        """
        if self._device_checkpoints is not None:
            self.backend.finish()
            if self._host_checkpoints is None:
                self._host_checkpoints = {par: np.empty((self.ns, self.nchp, self.tnz, self.tnx), np.float32)
                                          for par in COMPONENTS}
            block = self._device_shots * self.nchp * self._plane * 4  # Python ints: above 2 GB
            for comp, par in enumerate(COMPONENTS):
                self.backend.to_host(self._host_checkpoints[par][:self._device_shots],
                                     self.backend.section(self._device_checkpoints, comp * block, block))
            self._device_checkpoints, self._device_shots = None, 0
        return self._host_checkpoints

    @checkpoints.setter
    def checkpoints(self, stores):
        self._device_checkpoints, self._device_shots = None, 0  # the host copy is the storage from now on
        self._host_checkpoints = stores

    def _checkpoint_section(self, comp, s, chpc):
        """The part of the device checkpoints holding component ``comp`` of shot ``s`` at checkpoint ``chpc``."""
        offset = ((comp * self._device_shots + int(s)) * self.nchp + int(chpc)) * self._plane * 4  # no int32 overflow
        return self.backend.section(self._device_checkpoints, offset, self._plane * 4)

    # ------------------------------------------------------------------ buffers

    def _set_buffer(self, name, host):
        """Create the device buffer ``self.<name>`` holding ``host``, with margins (step 1.7).

        The allocation is ``[margin | the data of host | margin]``, the margins holding
        ``MARGIN_FILL``; ``self.<name>`` is the buffer of the data, which the kernels use, and
        ``self._allocations[name]`` the whole allocation (``margins_changed`` reads it). Replacing a
        buffer releases its old allocation. With CUDA graphs (step 3.11), a buffer of the same size
        keeps its allocation, filled exactly as a new one (margins included), so that the graphs
        that use it stay valid; a new allocation empties the graph cache (``_replay``).
        """
        host = np.ascontiguousarray(host)
        old = self._allocations.get(name)
        if self.graphs and old is not None and self.backend.nbytes(old) == host.nbytes + 2 * self.margin_bytes:
            margin = self.margin_bytes // host.itemsize
            whole = np.full(host.size + 2 * margin, MARGIN_FILL, host.dtype)
            whole[margin:margin + host.size] = host.reshape(-1)
            self.backend.to_device(old, whole)
            return
        buffer, self._allocations[name] = self.backend.allocate(host, self.margin_bytes, MARGIN_FILL)
        setattr(self, name, buffer)
        self._graph_cache = {}

    def _table(self, host):
        """A device copy of the int32 array ``host``, without margins (the receiver tables)."""
        host = np.ascontiguousarray(host, np.int32)
        buffer = self.backend.allocate_empty(max(host.nbytes, 4))
        if host.size:
            self.backend.to_device(buffer, host)
        return buffer

    def margins_changed(self):
        """Names of the device buffers whose margins no longer hold ``MARGIN_FILL`` (step 1.7).

        A kernel wrote outside that buffer (or, with a NaN fill, accumulated into it). Used by the
        tests of the out-of-bounds accesses; it reads every allocation back from the device.
        """
        changed = []
        margin = self.margin_bytes
        for name, allocation in self._allocations.items():
            whole = np.empty(self.backend.nbytes(allocation) // 4, np.float32)
            self.backend.to_host(whole, allocation)
            data_bytes = self.backend.nbytes(getattr(self, name))
            margins = np.concatenate([whole[:margin // 4], whole[(margin + data_bytes) // 4:]])
            intact = np.isnan(margins).all() if np.isnan(MARGIN_FILL) else (margins == np.float32(MARGIN_FILL)).all()
            if not intact:
                changed.append(name)
        return changed

    def _field_buffers(self):
        """The five device buffers of the forward field, in the order of ``COMPONENTS``."""
        return (self.vx_b, self.vz_b, self.taux_b, self.tauz_b, self.tauxz_b)

    def _slots_host(self):
        """The initial content of a buffer of ``FIELD_BUFFERS`` or ``GRADIENT_BUFFERS``.

        One shot at a time: a zero field, ``(tnz, tnx)``. A batch (step 3.9): ``batch`` zero fields,
        each followed by one row of ``MARGIN_FILL`` that no kernel writes. The stencils of the last
        rows read one row past the field (a known bug of both numeric modes); with a single shot that
        row is the margin after the buffer, and with a batch this row, so that every slot computes
        what a single shot computes (``slot_values`` of elastic.cu).
        """
        if self.batch == 1:
            return np.zeros((self.tnz, self.tnx), np.float32)
        host = np.full((self.batch, self.tnz + 1, self.tnx), MARGIN_FILL, np.float32)
        host[:, :self.tnz] = 0
        return host

    def _slot(self, buffer, slot):
        """The field of slot ``slot`` in a buffer of ``FIELD_BUFFERS`` or ``GRADIENT_BUFFERS`` (step 3.9).

        The buffer itself without batches: only the CUDA backend, which runs them, has ``section``.
        """
        if self.batch == 1:
            return buffer
        return self.backend.section(buffer, slot * self._stride * 4, self._plane * 4)

    def set_pml(self, vdx_pml, vdz_pml):
        """Device buffers of the absorbing boundary (damping times the largest velocity), per forward run.

        Also the rectangle where both profiles are zero, which the native kernels do not read there
        (``undamped_rectangle``, step 3.10).
        """
        self._set_buffer("vdx_pml_b", vdx_pml)
        self._set_buffer("vdz_pml_b", vdz_pml)
        undamped = undamped_rectangle(vdx_pml, vdz_pml)
        if undamped != getattr(self, "undamped", None):
            self._graph_cache = {}  # an argument of the graphs (_replay)
        self.undamped = undamped
        if self.boundary_width:
            self._set_band(band_cells(undamped, self.boundary_width, self.tnx))

    def _set_band(self, cells):
        """The band of boundary saving and the store of every shot (step 4.2), for the cells ``cells``.

        The store holds, for every shot, ``nt`` steps of the 5 fields at the ``n_band`` cells: memory
        proportional to the perimeter of the model, not to its area. Kept while the band is the same.
        """
        if self._band_cells is not None and np.array_equal(cells, self._band_cells):
            return
        if cells.size == 0:
            raise ValueError("boundary saving needs a region without absorbing layer")
        nbytes = int(self.ns) * int(self.nt) * 5 * int(cells.size) * 4  # Python ints: no int32 overflow
        budget = self.backend.checkpoint_budget() if hasattr(self.backend, "checkpoint_budget") else nbytes
        if nbytes > budget:
            raise MemoryError(f"boundary saving needs {nbytes / 1e9:.2f} GB on the GPU for the {self.ns} shots, "
                              f"more than the {budget / 1e9:.2f} GB available (PYFWI_CUDA_CHECKPOINT_BUDGET)")
        self._band_cells, self.n_band = cells, int(cells.size)
        self.band_b = self._table(cells)
        self._band_store = None  # release the old one first
        self._band_store = self.backend.allocate_empty(nbytes)
        self.band_size = (1, self.n_band)
        self._graph_cache = {}  # arguments of the graphs (_replay)

    def _band_args(self):
        """Arguments of the band kernels that follow the fields: the store, the band, the shot of every
        slot, the step on the device."""
        return self._band_store, self.band_b, self.band_shot_b, self.step_b

    def set_model(self, rho_inverse, mu, lam):
        """Device buffers of the model (1 / density and the Lame parameters), per forward run."""
        self._set_buffer("rho_b", rho_inverse)
        self._set_buffer("mu_b", mu)
        self._set_buffer("lam_b", lam)

    def prepare_gradient(self):
        """Create the gradient, preconditioner and adjoint-field buffers, zeroed, for a new gradient."""
        field = self._slots_host()  # one field per slot of a batch (step 3.9)
        for name in ("Gmu_b", "Glam_b", "Grho_b", "g_mu_precond_b", "g_lam_precond_b", "g_rho_precond_b",
                     "avx_b", "avz_b", "ataux_b", "atauz_b", "atauxz_b"):
            self._set_buffer(name, field)

    def read_field(self, name, host):
        """Copy the device buffer ``self.<name>`` (e.g. ``"vx_b"``) into ``host`` and return it.

        With a batch of shots (step 3.9) the buffers of the fields and adjoint fields hold one field per
        slot: the field of the last shot run is read, as 0.1.10's buffers hold it; those of the
        gradients, slot 0, which holds the sum of every shot after ``gradient``.
        """
        buffer = getattr(self, name)
        if name in FIELD_BUFFERS or name in GRADIENT_BUFFERS:
            buffer = self._slot(buffer, self._last_slot if name in FIELD_BUFFERS else 0)
        self.backend.to_host(host, buffer)
        return host

    # ------------------------------------------------------------------ arguments and launches

    def _size(self, size, slots=None):
        """The global size of a launch over ``size`` for every slot of a batch (step 3.9).

        ``size`` itself for one slot, else ``(*size, slots)``: the slots are the z dimension of the CUDA
        grid. ``slots`` defaults to those of the batch being run (``self.slots``).
        """
        slots = self.slots if slots is None else slots
        return size if slots == 1 else (*size, slots)

    def _launch(self, kernel):
        """Launch ``kernel`` on the whole grid (of every slot of the batch) with the arguments set before (step 1.3)."""
        self.backend.launch(kernel, self._size(self.global_size))

    def _set_update_args(self, coeff):
        """Set the arguments of the four update kernels of the elastic fields.

        ``coeff`` (int32) is +1 to propagate forward in time (forward modelling) and -1 to go
        backward (the forward field rebuilt during the gradient). Called at the start of each run,
        after the model and boundary buffers of that run exist.
        """
        set_args, k = self.backend.set_args, self.knl
        set_args(k.update_velx, coeff, self.vx_b, self.taux_b, self.tauxz_b, self.rho_b,
                 self.vdx_pml_b, self.vdz_pml_b)
        set_args(k.update_velz, coeff, self.vz_b, self.tauz_b, self.tauxz_b, self.rho_b,
                 self.vdx_pml_b, self.vdz_pml_b)
        set_args(k.update_tauz, coeff, self.vx_b, self.vz_b, self.taux_b, self.tauz_b,
                 self.lam_b, self.mu_b, self.vdx_pml_b, self.vdz_pml_b)
        set_args(k.update_tauxz, coeff, self.vx_b, self.vz_b, self.tauxz_b, self.mu_b,
                 self.vdx_pml_b, self.vdz_pml_b)
        if self.fused:  # with the rectangle without damping (step 3.10)
            set_args(k.update_v, coeff, self.vx_b, self.vz_b, self.taux_b, self.tauz_b, self.tauxz_b, self.rho_b,
                     self.vdx_pml_b, self.vdz_pml_b, *self.undamped)
            set_args(k.update_tau, coeff, *self._field_buffers(), self.lam_b, self.mu_b, self.vdx_pml_b,
                     self.vdz_pml_b, *self.undamped)

    def _set_injection_args(self, s):
        """Set the arguments of injSrc that stay the same for every time step of shot ``s``.

        The last five (the four source samples and the time index) change at each injection and
        are set by ``_inject``; zeros are placeholders until then.
        """
        zero = np.float32(0)
        seismograms = (self.seismogramid_vx_b, self.seismogramid_vz_b, self.seismogramid_taux_b,
                       self.seismogramid_tauz_b, self.seismogramid_tauxz_b)
        if self.native_injection:  # inject_and_record of elastic.cu (step 3.4)
            source_cell = np.int32(int(self.srcz[s]) * int(self.tnx) + int(self.srcx[s]))
            self.backend.set_args(self.knl.inject_and_record, *self._field_buffers(), *seismograms,
                                  self.receiver_cells_b, source_cell, zero, zero, zero, zero, np.int32(0))
        else:
            self.backend.set_args(self.knl.injSrc, *self._field_buffers(), *seismograms,
                                  self.dxr, self.srcx[s], self.srcz[s], zero, zero, zero, zero, np.int32(0))

    def _set_adjoint_args(self):
        """Set the arguments of Adj_injSrc, Adj_update_tau, Adj_update_v and Grad for a gradient.

        The residual buffers keep their identity during a gradient; only their content changes
        (``residual_sample``).
        """
        set_args, k = self.backend.set_args, self.knl
        adjoint = (self.avx_b, self.avz_b, self.ataux_b, self.atauz_b, self.atauxz_b)
        residuals = (self.res_vx_b, self.res_vz_b, self.res_taux_b, self.res_tauz_b, self.res_tauxz_b)
        if not self.native_injection:  # inject_residual: _set_native_residual_args, once per shot
            set_args(k.Adj_injSrc, *adjoint, *residuals, self.dxr)
        native = (*self.undamped,) if self.prg_native is not None else ()  # elastic.cu: the rectangle (3.10)
        set_args(k.Adj_update_tau, *adjoint, self.rho_b, self.vdx_pml_b, self.vdz_pml_b, *native)
        set_args(k.Adj_update_v, *adjoint, self.lam_b, self.mu_b, self.vdx_pml_b, self.vdz_pml_b)
        set_args(k.Grad, *self._field_buffers(), *adjoint, self.lam_b, self.mu_b, self.rho_b,
                 self.Gmu_b, self.g_mu_precond_b, self.Glam_b, self.g_lam_precond_b,
                 self.Grho_b, self.g_rho_precond_b)
        if self.fused:
            set_args(k.grad_adjoint_v, *self._field_buffers(), *adjoint, self.lam_b, self.mu_b, self.rho_b,
                     self.vdx_pml_b, self.vdz_pml_b, self.Gmu_b, self.g_mu_precond_b, self.Glam_b,
                     self.g_lam_precond_b, self.Grho_b, self.g_rho_precond_b, *self.undamped)

    def _inject(self, samples, row):
        """Add ``samples`` to the fields at the source and record the receivers in ``row`` (injSrc).

        Parameters
        ----------
        samples : array_like
            The five source samples (vx, vz, taux, tauz, tauxz); the fifth is not used by the
            kernel, as in 0.1.10.
        row : int
            Row of the seismograms where the receivers are recorded. -1 (fixed mode only): inject
            without recording.
        """
        src_kv_x, src_kv_z, src_kt_x, src_kt_z, _ = np.float32(samples)
        # The source samples and the row are the last five arguments (the others are set once per
        # shot by _set_injection_args): 13 to 17 of injSrc, 12 to 16 of inject_and_record.
        kernel = self.knl.inject_and_record if self.native_injection else self.knl.injSrc
        start = 12 if self.native_injection else 13
        for index, value in enumerate((src_kv_x, src_kv_z, src_kt_x, src_kt_z, np.int32(row)), start=start):
            self.backend.set_arg(kernel, index, value)
        if self.native_injection:
            self.backend.launch(kernel, self.receiver_size)
        else:
            self._launch(kernel)

    def _set_native_residual_args(self):
        """Arguments of inject_residual for a shot (step 3.6): the residual of the shot on the device
        (``residual_to_device``; of every shot of a batch, step 3.9), read at the sample of the step on
        the device."""
        self.backend.set_args(self.knl.inject_residual, self.avx_b, self.avz_b, self.ataux_b, self.atauz_b,
                              self.atauxz_b, *(self.res_shot_b[par] for par in COMPONENTS), self.receiver_cells_b,
                              self.receiver_first_b, self.receiver_following_b, self.step_b,
                              np.int32(self.steps_per_sample), np.int32(self.residual_rows))

    def _inject_residual(self):
        """Add the residual sample set by ``residual_sample`` to the adjoint fields (Adj_injSrc)."""
        if self.native_injection:
            self.backend.launch(self.knl.inject_residual, self.receiver_size)
        else:
            self._launch(self.knl.Adj_injSrc)

    # ------------------------------------------------------------------ forward modelling

    def forward(self, observe=None):
        """Forward modelling of every shot; returns the seismograms (``self.seismogram``).

        Parameters
        ----------
        observe : callable, optional
            ``observe(s, t)`` is called after every internal step ``t`` of shot ``s`` (the
            ``show=True`` path of ``WavePropagator`` plots the wavefield from it).
        """
        coeff = np.int32(+1)
        # The arguments that do not change during the time loop are set once (step 1.3).
        self._set_update_args(coeff)
        if self.fixed:
            # The wavelet may have changed since the last forward modelling: its samples are read
            # every time, and resampled when they changed (0.4 ms of host time on the small scenario,
            # with the GPU idle, before step 3.11).
            samples = recorded_source(self.src, self.nt, self.dt_scale)
            if self._source_samples is None or not np.array_equal(samples, self._source_samples):
                self._source_samples = samples
                self.source_steps = resampled_source(samples, self.nt, self.dt_scale)
                if self.native_injection:
                    self.backend.to_device(self.source_steps_b, np.ascontiguousarray(self.source_steps[:, :4]))
        if self.native_injection:
            self._set_native_injection_args()
            self.backend.set_args(self.knl.advance, self.step_b, np.int32(+1))
        for shots in self._batches(observe):
            self.slots, self._last_slot = len(shots), len(shots) - 1
            self.backend.call(self.knl.MakeAllZero, self._size(self.global_size), *self._field_buffers())
            if self.native_injection:
                self._start_native_batch(shots, 0)
                self._forward_batch_native(shots, observe)
                continue
            self._set_injection_args(shots[0])  # one shot per batch without native injection
            self._forward_shot(shots[0], observe)
        # The checkpoint copies did not block: wait for them before the checkpoints can be read.
        self.backend.finish()
        return self.seismogram

    def _forward_shot(self, s, observe):
        """The time loop of the forward modelling of shot ``s``."""
        chpc = 0
        t_src = 0
        for t in np.arange(self.nt):
            if self.fixed:
                # The source at every step. The receivers at every dt_scale steps, in the row of
                # that recorded sample when it exists: rounding in nt and nt_ext can put the last
                # recorded step one row past the seismograms, where 0.1.10 writes it.
                sample, offset = divmod(int(t), self.steps_per_sample)
                self._inject(self.source_steps[t], sample if offset == 0 and sample < self.nt_ext else -1)
            elif t % self.dt_scale == 0:
                self._inject(self.src(t_src), t_src)
                t_src += 1

            self._forward_step()

            if t in self.chp:
                # Straight into the contiguous checkpoint storage, without blocking (step 1.5). The
                # queue runs in order, so the next updates start after the copies.
                self._store_checkpoint(s, chpc)
                chpc += 1

            if observe is not None:
                observe(s, t)
        self.make_seismogram(s)

    # ------------------------------------------------------------------ native time loops (3.6, 3.7)

    def _set_native_injection_args(self):
        """Arguments of inject_and_record for a whole run (step 3.6): the step comes from the device; with
        boundary saving, of save_band too (step 4.2)."""
        self.backend.set_args(self.knl.inject_and_record, *self._field_buffers(),
                              self.seismogramid_vx_b, self.seismogramid_vz_b, self.seismogramid_taux_b,
                              self.seismogramid_tauz_b, self.seismogramid_tauxz_b, self.receiver_cells_b,
                              self.source_cell_b, self.source_steps_b, self.step_b,
                              np.int32(self.steps_per_sample), np.int32(self.nt_ext))
        if self.boundary_width:
            self.backend.set_args(self.knl.save_band, *self._field_buffers(), *self._band_args(),
                                  np.int32(self.n_band), np.int64(int(self.nt) * 5 * self.n_band))

    def _set_rebuild_args(self):
        """Arguments of the kernels that rebuild the forward field with boundary saving (step 4.2).

        The gradient step ``t`` undoes the forward step ``t + 1`` (``_gradient_batch_native``): the
        band and the source of that step (``delta`` 1).
        """
        shot_values = np.int64(int(self.nt) * 5 * self.n_band)
        for kernel, stresses in ((self.knl.restore_stresses, 1), (self.knl.restore_velocities, 0)):
            self.backend.set_args(kernel, *self._field_buffers(), *self._band_args(), np.int32(1), np.int32(stresses),
                                  np.int32(self.n_band), shot_values)
        self.backend.set_args(self.knl.remove_source, self.vx_b, self.vz_b, self.taux_b, self.tauz_b,
                              self.source_cell_b, self.source_steps_b, self.step_b, np.int32(1))

    def _batches(self, observe):
        """The shots in batches of ``batch`` (step 3.9), as ranges: slot b of a batch runs its shot b.

        The last batch may be shorter. One shot per batch when ``observe`` sees every step (the
        ``show=True`` path reads the field of the shot being run).
        """
        size = 1 if observe is not None else self.batch
        return [range(s, min(s + size, self.ns)) for s in range(0, self.ns, size)]

    def _start_native_batch(self, shots, step):
        """Put the source cells of ``shots`` (slot b: ``shots[b]``) and the first internal step of their
        loop on the device."""
        cells = np.zeros(self.batch, np.int32)  # the whole table: a copy has the size of its buffer
        for b, s in enumerate(shots):
            cells[b] = int(self.srcz[s]) * int(self.tnx) + int(self.srcx[s])
        self.backend.to_device(self.source_cell_b, cells)
        if self.boundary_width:  # the store of every slot's shot (step 4.2)
            band_shots = np.zeros(self.batch, np.int32)
            band_shots[:len(shots)] = list(shots)
            self.backend.to_device(self.band_shot_b, band_shots)
        self.backend.to_device(self.step_b, np.array([step], np.int32))

    def _replay(self, kind, step, count):
        """Run ``count`` repetitions of ``step()`` as CUDA graphs of at most ``MAX_GRAPH_STEPS`` steps.

        A graph records the launches of ``count`` steps once (backend ``capture``) and replays them
        with one call (step 3.7); every step reads its time from the device, so one graph serves
        every run of the same length. The graphs are kept in ``_graph_cache`` by kind, length, slots
        of the batch (step 3.9: a shorter last batch has launches of its own) and kernel set, from
        run to run (step 3.11): recording one took 0.2 to 1.1 ms of host time, a few per run. A graph
        keeps the arguments of its launches, so the cache is emptied when a buffer it may use is
        replaced (``_set_buffer``, ``residual_to_device``) or the undamped rectangle changes
        (``set_pml``); the other arguments are the same at every run of a kind.
        """
        while count > 0:
            length = min(count, MAX_GRAPH_STEPS)
            key = (kind, length, self.slots, self.fused)
            graph = self._graph_cache.get(key)
            if graph is None:
                graph = self.backend.capture(lambda n=length: [step() for _ in range(n)])
                self._graph_cache[key] = graph
            graph.launch()
            count -= length

    def _native_forward_step(self):
        """One forward step with the native kernels: source and receivers, the band of boundary saving
        (step 4.2), the updates, the next step."""
        self.backend.launch(self.knl.inject_and_record, self._size(self.receiver_size))
        if self.boundary_width:
            self.backend.launch(self.knl.save_band, self._size(self.band_size))
        self._forward_step()
        self.backend.launch(self.knl.advance, (1, 1))

    def _store_checkpoint(self, s, chpc, slot=0):
        """Copy the field of slot ``slot`` into checkpoint ``chpc`` of shot ``s``, without blocking (step
        1.5): on the device when the checkpoints of the shot are there (step 3.8), else on the host."""
        fields = [self._slot(buffer, slot) for buffer in self._field_buffers()]
        if s < self._device_shots:
            for comp, field in enumerate(fields):
                self.backend.copy(self._checkpoint_section(comp, s, chpc), field, self._plane * 4)
            return
        for par, field in zip(COMPONENTS, fields):  # the storage, not the property
            self.backend.to_host(self._host_checkpoints[par][s, chpc], field, blocking=False)

    def _forward_batch_native(self, shots, observe):
        """The forward time loop of the batch ``shots`` with the native kernels and the step on the device.

        Shot ``shots[b]`` runs in slot b (step 3.9), every launch covering every slot. The checkpoints
        are copied after the updates of their step, as in ``_forward_shot``. Without ``observe``, the
        steps between two checkpoints are replayed as CUDA graphs (``_replay``); with it
        (``show=True``, one shot per batch), step by step, the same kernels.
        """
        checkpoints = [int(c) for c in self.chp]
        if self.graphs and observe is None:
            t = 0
            for chpc, c in enumerate(checkpoints):
                if c >= t:
                    self._replay("forward", self._native_forward_step, c - t + 1)
                    t = c + 1
                self._store_checkpoints(shots, chpc)
            self._replay("forward", self._native_forward_step, self.nt - t)
        else:
            stored = {c: chpc for chpc, c in enumerate(checkpoints)}
            for t in range(self.nt):
                self._native_forward_step()
                if t in stored:
                    self._store_checkpoints(shots, stored[t])
                if observe is not None:
                    observe(shots[0], t)  # the only shot of an observed batch (_batches)
        self._read_seismograms(shots)

    def _read_seismograms(self, shots):
        """Read the seismograms of the batch ``shots`` into ``self.seismogram``: one read per component.

        ``make_seismogram`` for every shot read each slot of each component apart, waiting after
        each shot; the slots of a batch follow each other in the seismogram buffers, so one read of
        the first ``len(shots)`` slots does (step 3.11: 20 reads of the small scenario, 0.6 ms with
        the GPU idle between them, became 5).
        """
        count = len(shots)
        buffers = (self.seismogramid_vx_b, self.seismogramid_vz_b, self.seismogramid_taux_b,
                   self.seismogramid_tauz_b, self.seismogramid_tauxz_b)
        if count < self.batch:  # only CUDA, which runs batches, has section
            nbytes = count * int(self.nt_ext) * int(self.nr) * 4
            buffers = [self.backend.section(buffer, 0, nbytes) for buffer in buffers]
        host = {par: np.empty((count, self.nt_ext, self.nr), np.float32) for par in COMPONENTS}
        events = [self.backend.to_host(host[par], buffer, blocking=False) for par, buffer in zip(COMPONENTS, buffers)]
        self.backend.wait(events)
        for par in COMPONENTS:
            self.seismogram[par][:, :, shots[0]:shots[-1] + 1] = host[par].transpose(1, 2, 0)

    def _store_checkpoints(self, shots, chpc):
        """Store checkpoint ``chpc`` of every shot of the batch ``shots``, each from its slot: those kept on
        the device in one launch (``_copy_checkpoints``), the others as ``_store_checkpoint``."""
        on_device = self._copy_checkpoints(shots, chpc, restore=0)
        for slot in range(on_device, len(shots)):
            self._store_checkpoint(shots[slot], chpc, slot)

    def _restore_checkpoints(self, shots, chpc):
        """Restore checkpoint ``chpc`` of every shot of the batch ``shots``, each into its slot: those kept
        on the device in one launch (``_copy_checkpoints``), the others as ``_restore_checkpoint``."""
        on_device = self._copy_checkpoints(shots, chpc, restore=1)
        for slot in range(on_device, len(shots)):
            self._restore_checkpoint(shots[slot], chpc, slot)

    def _copy_checkpoints(self, shots, chpc, restore):
        """Copy the field of the slots of ``shots`` whose checkpoints are on the device into checkpoint
        ``chpc`` (``restore`` 0) or back (1), with one launch of ``copy_checkpoint`` (step 3.9).

        One copy per component and shot cost about 6 us of host time each, which made most of the
        time of a small grid. The device holds the checkpoints of the first ``_device_shots`` shots
        (step 3.8), so they are the first slots of a batch. Returns how many slots were copied.
        """
        count = max(0, min(len(shots), self._device_shots - shots[0]))
        if count:
            per_shot = self.nchp * self._plane  # values; Python ints: no int32 overflow
            offset = (shots[0] * self.nchp + int(chpc)) * self._plane
            self.backend.call(self.knl.copy_checkpoint, self._size(self.global_size, count), *self._field_buffers(),
                              self._device_checkpoints, np.int64(offset), np.int64(per_shot),
                              np.int64(self._device_shots * per_shot), np.int32(restore))
        return count

    def _native_adjoint_step(self, backward):
        """One gradient step with the native kernels (step 3.6): the forward field one step back (unless
        it was just restored), the gradient and the adjoint updates, the residual of the step, the
        previous step. The order of ``_adjoint_step``."""
        if backward:
            self._backward_step()
        if not self.fused:  # the separate kernels (tests): Grad first, as in _adjoint_step
            self._launch(self.knl.Grad)
        self._adjoint_updates()  # fused: grad_adjoint_v (the gradient, then the adjoint velocities), Adj_update_tau
        self.backend.launch(self.knl.inject_residual, self._size(self.receiver_size))
        self.backend.launch(self.knl.advance, (1, 1))

    def _restore_checkpoint(self, s, chpc, slot=0):
        """Copy checkpoint ``chpc`` of shot ``s`` back into the field of slot ``slot``, without blocking
        (step 1.5): from the device when the checkpoints of the shot are there (step 3.8), else from
        the host."""
        fields = [self._slot(buffer, slot) for buffer in self._field_buffers()]
        if s < self._device_shots:
            for comp, field in enumerate(fields):
                self.backend.copy(field, self._checkpoint_section(comp, s, chpc), self._plane * 4)
            return
        for par, field in zip(COMPONENTS, fields):  # the storage, not the property
            self.backend.to_device(field, self._host_checkpoints[par][s, chpc], blocking=False)

    def _gradient_batch_native(self, shots, observe):
        """The backward time loop of the batch ``shots`` with the native kernels and the step on the device.

        Steps nt - 1 down to 1, as ``_gradient_shot``: at a checkpoint step the forward field of
        every shot is restored into its slot, otherwise rebuilt one step back. Without ``observe``
        the runs of steps between two checkpoints are replayed as CUDA graphs; with it (one shot per
        batch), step by step.
        """
        restored = {int(c): chpc for chpc, c in enumerate(self.chp)}
        if self.graphs and observe is None:
            t = self.nt - 1
            while t >= 1:
                if t in restored:
                    self._restore_checkpoints(shots, restored[t])
                    self._replay("restored", lambda: self._native_adjoint_step(False), 1)
                    t -= 1
                    continue
                count = 0
                while t - count >= 1 and t - count not in restored:
                    count += 1
                self._replay("adjoint", lambda: self._native_adjoint_step(True), count)
                t -= count
            return
        t_src = np.int32(self.nt_ext)
        for t in range(self.nt - 1, 0, -1):
            if t in restored:
                self._restore_checkpoints(shots, restored[t])
            self._native_adjoint_step(t not in restored)
            sample, offset = divmod(t, self.steps_per_sample)
            if offset == 0 and sample < self.nt_ext:
                t_src = np.int32(sample)
            if observe is not None:
                observe(shots[0], t, t_src)  # the only shot of an observed batch (_batches)

    def _forward_step(self):
        """The four field updates of one time step; arguments set by ``_set_update_args``.

        The velocities from the stresses (V: update_velx, update_velz), then the stresses from the
        new velocities (T: update_tauz, update_tauxz): with the source added before, the step is
        u(t + 1) = T(V(u(t) + source(t))). With the native kernels, one launch for each half
        (update_v, update_tau: step 3.5).
        """
        if self.fused:
            self._launch(self.knl.update_v)
            self._launch(self.knl.update_tau)
            return
        self._launch(self.knl.update_velx)
        self._launch(self.knl.update_velz)
        self._launch(self.knl.update_tauz)
        self._launch(self.knl.update_tauxz)

    def _backward_step(self):
        """Undo one forward step (``_set_update_args`` with coeff = -1): the stresses, then the velocities.

        With boundary saving (step 4.2), the stored band is written back after each half, and the source
        of the step is removed: the field is then the forward field of the previous step inside the
        undamped rectangle, but for rounding.
        """
        if self.fused:
            self._launch(self.knl.update_tau)
            if self.boundary_width:
                self.backend.launch(self.knl.restore_stresses, self._size(self.band_size))
            self._launch(self.knl.update_v)
            if self.boundary_width:
                self.backend.launch(self.knl.restore_velocities, self._size(self.band_size))
                self.backend.launch(self.knl.remove_source, self._size((1, 1)))
            return
        self._launch(self.knl.update_tauxz)
        self._launch(self.knl.update_tauz)
        self._launch(self.knl.update_velz)
        self._launch(self.knl.update_velx)

    def make_seismogram(self, s, slot=0):
        """Read the seismograms of shot ``s``, run in slot ``slot``, into ``self.seismogram[...][:, :, s]``.

        The five reads are enqueued without blocking and waited for once (step 1.4). With batches
        (step 3.9) the seismogram buffers hold those of every slot, one after the other.
        """
        buffers = (self.seismogramid_vx_b, self.seismogramid_vz_b, self.seismogramid_taux_b,
                   self.seismogramid_tauz_b, self.seismogramid_tauxz_b)
        if self.batch > 1:
            nbytes = int(self.nt_ext) * int(self.nr) * 4
            buffers = [self.backend.section(buffer, slot * nbytes, nbytes) for buffer in buffers]
        host = {par: np.empty((self.nt_ext, self.nr), np.float32) for par in COMPONENTS}
        events = [self.backend.to_host(host[par], buffer, blocking=False) for par, buffer in zip(COMPONENTS, buffers)]
        self.backend.wait(events)
        for par in COMPONENTS:
            self.seismogram[par][:, :, s] = host[par]

    # ------------------------------------------------------------------ gradient

    def gradient(self, res, observe=None):
        """Adjoint-state gradient of every shot into the gradient buffers (read them with ``read_gradient``).

        Parameters
        ----------
        res : dict
            Residual of every shot, ``{component: (recorded samples, nr * ns)}``.
        observe : callable, optional
            ``observe(s, t, t_src)`` is called after every internal step ``t`` of shot ``s``
            (``t_src``: the last recorded sample injected, ``nt_ext`` before the first).
        """
        self.backend.call(self.knl.MakeGradZero, self._size(self.global_size, self.batch),  # every slot
                          self.Gmu_b, self.Glam_b, self.Grho_b,
                          self.g_mu_precond_b, self.g_lam_precond_b, self.g_rho_precond_b)

        # The arguments of the time loop do not change during the gradient: set them once (step
        # 1.3). The update kernels rebuild the forward field backwards (coeff = -1).
        self._set_update_args(np.int32(-1))
        self._set_adjoint_args()

        if self.native_injection:
            self.backend.set_args(self.knl.advance, self.step_b, np.int32(-1))
            if self.boundary_width:
                self._set_rebuild_args()
        for shots in self._batches(observe):
            self.slots, self._last_slot = len(shots), len(shots) - 1
            self.backend.call(self.knl.MakeAllZero, self._size(self.global_size), *self._field_buffers())
            self.backend.call(self.knl.MakeAllZero, self._size(self.global_size),
                              self.avx_b, self.avz_b, self.ataux_b, self.atauz_b, self.atauxz_b)
            self.residual_to_device(res, shots[0], len(shots))
            if self.native_injection:
                self._set_native_residual_args()
                self._start_native_batch(shots, self.nt - 1)
                self._gradient_batch_native(shots, observe)
                continue
            self._gradient_shot(shots[0], observe)  # one shot per batch without native injection
        if self.batch > 1:
            # Step 3.9: every slot accumulated the gradient of its shots; slot 0 gets the sum, added in
            # the order of the slots (the same bits at every run).
            self.backend.call(self.knl.sum_slots, self.global_size, self.Gmu_b, self.g_mu_precond_b, self.Glam_b,
                              self.g_lam_precond_b, self.Grho_b, self.g_rho_precond_b, np.int32(self.batch))

    def _gradient_shot(self, s, observe):
        """The backward time loop of the gradient of shot ``s``."""
        chpc = self.nchp - 1
        t_src = np.int32(self.nt_ext)
        for t in range(self.nt - 1, 0, -1):
            if t == self.chp[chpc]:
                # Straight from the contiguous checkpoint storage, without blocking (step 1.5): the
                # checkpoints do not change during the gradient.
                self._restore_checkpoint(s, chpc)
                chpc -= 1
            else:
                # Backward propagation; arguments (coeff = -1) set by _set_update_args.
                self._backward_step()

            # Adjoint modelling; arguments set by _set_adjoint_args.
            if self.fixed:
                t_src = self._adjoint_step(t, t_src)
            else:
                # As in 0.1.10: the residual, the adjoint updates, then the gradient.
                if t % self.dt_scale == 0:
                    self.residual_sample(t_src)
                    self._inject_residual()
                    t_src -= 1
                self._adjoint_updates()
                self._launch(self.knl.Grad)

            if observe is not None:
                observe(s, t, t_src)

    def _adjoint_updates(self):
        """The adjoint field one step back in time; arguments set by ``_set_adjoint_args``.

        The transpose of the forward step T(V(...)) of ``_forward_step`` is V^T(T^T(...)): first the
        adjoint velocities from the adjoint stresses (Adj_update_v, the transpose of T), then the
        adjoint stresses from the new adjoint velocities (Adj_update_tau, the transpose of V). The
        fixed mode runs them in that order. 0.1.10 (the legacy mode) runs Adj_update_tau first: its
        adjoint step is not the transpose of the forward one (measured on random fields, one step:
        <A u, v> and <u, A^T v> differ by 4e-3 in relative terms, against 1e-7 in that order). The
        native kernels run Adj_update_v fused with Grad (grad_adjoint_v, step 3.5): with them this
        also accumulates the gradient, which ``_adjoint_step`` then does not launch separately.
        """
        if self.fused:  # Grad + Adj_update_v in one launch (step 3.5): the gradient accumulates too
            self._launch(self.knl.grad_adjoint_v)
            self._launch(self.knl.Adj_update_tau)
        elif self.fixed:
            self._launch(self.knl.Adj_update_v)
            self._launch(self.knl.Adj_update_tau)
        else:
            self._launch(self.knl.Adj_update_tau)
            self._launch(self.knl.Adj_update_v)

    def _adjoint_step(self, t, t_src):
        """Step ``t`` of the adjoint modelling in the fixed numeric mode (step 2.4); returns ``t_src``.

        The forward modelling computes u(t + 1) = A(u(t) + source(t)), A = T V (``_forward_step``),
        and records the receivers R u(t) at the recorded steps. The adjoint field of the misfit then
        obeys lambda(t) = A^T lambda(t + 1) + R^T residual(t), and the gradient adds, for every step,
        lambda(t + 1) times the derivative of A with respect to the model, taken at u(t). When this
        runs, the forward field has been rebuilt to u(t + 1) and the adjoint field holds
        lambda(t + 1), so:

        * Grad first: lambda(t + 1) with u(t + 1), whose velocities are those that updated the
          stresses at step t (the terms of the Lame parameters are those of the exact gradient;
          the density term pairs the stresses one step later than exact);
        * then A^T (``_adjoint_updates``);
        * then the residual recorded at step t: residual sample k at step k * dt_scale. 0.1.10
          counts the samples down from nt_ext - 1 at the last multiple of dt_scale: one sample off
          when rounding makes nt_ext one more (dt = 4 ms, t = 0.4 s) or one less (dt = 5 ms,
          t = 0.25 s) than the recorded steps.

        0.1.10 injects the residual first and runs Grad last: it pairs u(t + 1) with the adjoint
        field of about two steps earlier. Measured (``tools/math_baseline.py``, vp): directional
        derivatives 6.9 % from finite differences in 0.1.10, 1.9 % with this order.

        Parameters
        ----------
        t : int
            Internal step, from nt - 1 down to 1.
        t_src : numpy.int32
            The last recorded sample injected (``nt_ext`` before the first).
        """
        if not self.fused:  # the fused kernel of _adjoint_updates accumulates the gradient itself
            self._launch(self.knl.Grad)
        self._adjoint_updates()
        sample, offset = divmod(t, self.steps_per_sample)
        if offset == 0 and sample < self.nt_ext:
            self.residual_sample(sample + 1)
            self._inject_residual()
            t_src = np.int32(sample)
        return t_src

    def residual_to_device(self, res, s, count=1):
        """Copy the residual of shots ``s`` to ``s + count - 1``, all time samples, to the device (step 1.4).

        0.1.10 copied one time sample from the host before each injection (``make_residual``):
        five blocking copies every ``dt_scale`` steps. Now the five components of the shot are
        copied once, and ``residual_sample`` copies each sample on the device (the native kernels
        read it in place). With native injection, the shots of a batch (step 3.9) are copied one
        after the other (slot b: shot ``s + b``), each with the rows that the kernels read
        (``nt_ext`` at most: ``residual_rows``), into buffers of ``batch`` shots.
        """
        nr = int(self.nr)
        rows = len(res["vx"])
        if self.native_injection:
            rows = min(rows, int(self.nt_ext))
        self.residual_rows = rows
        for par in COMPONENTS:
            # Columns s*nr to (s+count)*nr, as float32 and contiguous, shot after shot: the values
            # make_residual sent.
            columns = np.asarray(res[par])[:rows, s * nr:(s + count) * nr]
            shots = np.ascontiguousarray(columns.reshape(rows, count, nr).transpose(1, 0, 2), dtype=np.float32)
            nbytes = self.batch * shots[0].nbytes
            if self.res_shot_b.get(par) is None or self.backend.nbytes(self.res_shot_b[par]) != nbytes:
                self.res_shot_b[par] = self.backend.allocate_empty(nbytes)
                self._graph_cache = {}  # an argument of the graphs (_replay)
            target = self.res_shot_b[par]
            if count < self.batch:  # the last batch: the first slots of the buffer
                target = self.backend.section(target, 0, shots.nbytes)
            self.backend.to_device(target, shots)

    def residual_sample(self, t):
        """Copy sample ``t - 1`` of the shot's residual into the buffers that Adj_injSrc reads.

        The copies stay on the device. Adj_injSrc keeps reading ``res_vx_b``... of ``nr`` values,
        as in 0.1.10, so that what its unbounded receiver loops read past their end is unchanged.

        Parameters
        ----------
        t : int
            Time index used by 0.1.10 in ``make_residual(res, s, t)``: sample ``t - 1`` is copied.
        """
        row = int(self.nr) * 4  # bytes of one time sample of one component
        sample = int(t) - 1
        if sample < 0:  # as NumPy indexing in make_residual: -1 is the last sample
            sample += self.backend.nbytes(self.res_shot_b["vx"]) // row
        targets = (self.res_vx_b, self.res_vz_b, self.res_taux_b, self.res_tauz_b, self.res_tauxz_b)
        for par, target in zip(COMPONENTS, targets):
            self.backend.copy(target, self.res_shot_b[par], row, sample * row)

    def make_residual(self, res, s, t):
        """0.1.10: copy sample ``t - 1`` of the residual of shot ``s`` from the host. Kept, unused."""
        targets = (self.res_vx_b, self.res_vz_b, self.res_taux_b, self.res_tauz_b, self.res_tauxz_b)
        for par, target in zip(COMPONENTS, targets):
            sample = res[par][np.int32(t - 1), s * self.nr:(s + 1) * self.nr].astype(np.float32, order='C')
            self.backend.to_device(target, sample)

    def read_gradient(self, preconditioners=True):
        """Copy the gradient buffers to the host.

        Returns
        -------
        tuple of numpy.ndarray
            ``(g_mu, g_lam, g_rho, mu_precond, lam_precond, rho_precond)``, each ``(tnz, tnx)``.
            Without ``preconditioners`` the last three are arrays of ones (no energy balancing).
        """
        shape = (self.tnz, self.tnx)
        g_mu, g_lam, g_rho = (np.zeros(shape, np.float32) for _ in range(3))
        # Slot 0, which holds the sum of the shots of every slot (step 3.9).
        self.backend.to_host(g_mu, self._slot(self.Gmu_b, 0))
        self.backend.to_host(g_lam, self._slot(self.Glam_b, 0))
        self.backend.to_host(g_rho, self._slot(self.Grho_b, 0))
        if not preconditioners:
            return g_mu, g_lam, g_rho, np.ones(shape, np.float32), np.ones(shape, np.float32), np.ones(shape, np.float32)
        precond = [np.zeros(shape, np.float32) for _ in range(3)]
        self.backend.to_host(precond[0], self._slot(self.g_mu_precond_b, 0))
        self.backend.to_host(precond[1], self._slot(self.g_lam_precond_b, 0))
        self.backend.to_host(precond[2], self._slot(self.g_rho_precond_b, 0))
        return (g_mu, g_lam, g_rho, *precond)
