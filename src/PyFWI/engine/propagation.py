"""The time loops of the elastic propagation, on any backend (roadmap step 2.1).

``PropagationEngine`` runs the kernels of PyFWI (the OpenCL C files of 0.1.10) in the same order
as 0.1.10 did, through a backend (``PyFWI.backends``): it never calls a GPU library itself. It
owns everything that lives on the device or follows the time loop:

* the device buffers (fields, adjoint fields, model, absorbing boundary, seismograms, residuals,
  gradients), each inside an allocation with margins (step 1.7);
* the kernels, retrieved once, with their constant arguments set once per run (step 1.3);
* the forward loop: source injection every ``dt_scale`` steps, the four field updates, the
  checkpoints of the forward field (stored as ``(ns, nchp, tnz, tnx)`` per component, step 1.5),
  and the seismograms read at the end of each shot (step 1.4);
* the gradient loop: the forward field rebuilt backwards from the checkpoints, the residual
  injected from a copy on the device (step 1.4), the adjoint updates and the gradient kernel.

``PyFWI.wave_propagation.WavePropagator`` is the public face: it prepares the inputs (time steps,
geometry, absorbing boundary, model), turns the results into seismic sections and gradients, and
shows the engine's attributes as its own (``W.vx_b``, ``W.knl``...), as in 0.1.10.

Used by
-------
``PyFWI.wave_propagation.WavePreparation``. Tested by ``test/test_engine.py`` (with a fake NumPy
backend, without a device) and by every test that runs a propagation.
"""
from types import SimpleNamespace

import numpy as np

#: The five recorded components, in the order of the seismogram and residual buffers.
COMPONENTS = ("vx", "vz", "taux", "tauz", "tauxz")

#: Names of the kernels that the engine launches (in elastic.cl; injSrc and Adj_injSrc also in
#: elastic_crosswell.cl and elastic_surface.cl, which serve the crosswell and surface acquisitions).
KERNELS = ("MakeAllZero", "MakeGradZero", "injSrc", "Adj_injSrc", "update_velx", "update_velz",
           "update_tauz", "update_tauxz", "Adj_update_tau", "Adj_update_v", "Grad")

#: Value of the margins around every device buffer (step 1.7): what the out-of-bounds reads of the
#: kernels of 0.1.10 return. 0 reproduces the golden masters, which 0.1.10 computed inside zero
#: guard bands; the tests set NaN or 1.0 to see where the kernels read and write.
MARGIN_FILL = 0.0

#: Smallest margin before and after every device buffer, in bytes: the size of the guard bands
#: with which the golden masters of 0.1.10 were made (test/oob_guard.py).
MIN_MARGIN_BYTES = 1 << 16


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
        Kernel sources with their ``#define`` block: ``"elastic"``, ``"crosswell"`` and ``"surface"``.
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
        Internal steps per recorded sample: the source is injected every ``dt_scale`` steps.
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
    """

    def __init__(self, backend, sources, injection, *, tnz, tnx, ns, nr, nt, nt_ext, dt_scale, chp, dxr,
                 srcx, srcz, source, rec_top_right_var):
        self.backend = backend
        self.tnz, self.tnx, self.ns, self.nr = tnz, tnx, ns, nr
        self.nt, self.nt_ext, self.dt_scale = nt, nt_ext, dt_scale
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

        self.margin_bytes = margin_bytes(tnz, tnx, rec_top_right_var, backend.alignment)
        # The allocation (margins included) of every device buffer, by attribute name.
        self._allocations = {}

        res = np.zeros(int(nr), np.float32)
        for par in COMPONENTS:
            self._set_buffer(f"res_{par}_b", res)
        # The residual of one shot, every time sample, per component (residual_to_device);
        # created at the first gradient and reused while its size does not change.
        self.res_shot_b = {}

        field = np.zeros((tnz, tnx), np.float32)
        for par in COMPONENTS:
            self._set_buffer(f"{par}_b", field)

        seismogram = np.zeros((nt_ext, nr), np.float32)
        for par in COMPONENTS:
            self._set_buffer(f"seismogramid_{par}_b", seismogram)

        #: The seismograms of the last forward modelling, ``{component: (nt_ext, nr, ns)}``.
        self.seismogram = {par: np.zeros((nt_ext, nr, ns), np.float32) for par in COMPONENTS}

        #: The checkpoints of the forward field, ``{component: (ns, nchp, tnz, tnx)}``: the field of
        #: one shot at one checkpoint is a contiguous block, copied straight from and to the device
        #: without blocking (step 1.5). ``WavePropagator.W`` shows them as in 0.1.10.
        self.checkpoints = {par: np.zeros((ns, self.nchp, tnz, tnx), np.float32) for par in COMPONENTS}

    # ------------------------------------------------------------------ buffers

    def _set_buffer(self, name, host):
        """Create the device buffer ``self.<name>`` holding ``host``, with margins (step 1.7).

        The allocation is ``[margin | the data of host | margin]``, the margins holding
        ``MARGIN_FILL``; ``self.<name>`` is the buffer of the data, which the kernels use, and
        ``self._allocations[name]`` the whole allocation (``margins_changed`` reads it). Replacing a
        buffer releases its old allocation.
        """
        buffer, self._allocations[name] = self.backend.allocate(host, self.margin_bytes, MARGIN_FILL)
        setattr(self, name, buffer)

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

    def set_pml(self, vdx_pml, vdz_pml):
        """Device buffers of the absorbing boundary (damping times the largest velocity), per forward run."""
        self._set_buffer("vdx_pml_b", vdx_pml)
        self._set_buffer("vdz_pml_b", vdz_pml)

    def set_model(self, rho_inverse, mu, lam):
        """Device buffers of the model (1 / density and the Lame parameters), per forward run."""
        self._set_buffer("rho_b", rho_inverse)
        self._set_buffer("mu_b", mu)
        self._set_buffer("lam_b", lam)

    def prepare_gradient(self):
        """Create the gradient, preconditioner and adjoint-field buffers, zeroed, for a new gradient."""
        field = np.zeros((self.tnz, self.tnx), np.float32)
        for name in ("Gmu_b", "Glam_b", "Grho_b", "g_mu_precond_b", "g_lam_precond_b", "g_rho_precond_b",
                     "avx_b", "avz_b", "ataux_b", "atauz_b", "atauxz_b"):
            self._set_buffer(name, field)

    def read_field(self, name, host):
        """Copy the device buffer ``self.<name>`` (e.g. ``"vx_b"``) into ``host`` and return it."""
        self.backend.to_host(host, getattr(self, name))
        return host

    # ------------------------------------------------------------------ arguments and launches

    def _launch(self, kernel):
        """Launch ``kernel`` on the whole grid with the arguments set before (step 1.3)."""
        self.backend.launch(kernel, self.global_size)

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

    def _set_injection_args(self, s):
        """Set the arguments of injSrc that stay the same for every time step of shot ``s``.

        The last five (the four source samples and the time index) change at each injection and
        are set by ``_inject``; zeros are placeholders until then.
        """
        zero = np.float32(0)
        self.backend.set_args(self.knl.injSrc, *self._field_buffers(),
                              self.seismogramid_vx_b, self.seismogramid_vz_b,
                              self.seismogramid_taux_b, self.seismogramid_tauz_b, self.seismogramid_tauxz_b,
                              self.dxr, self.srcx[s], self.srcz[s], zero, zero, zero, zero, np.int32(0))

    def _set_adjoint_args(self):
        """Set the arguments of Adj_injSrc, Adj_update_tau, Adj_update_v and Grad for a gradient.

        The residual buffers keep their identity during a gradient; only their content changes
        (``residual_sample``).
        """
        set_args, k = self.backend.set_args, self.knl
        adjoint = (self.avx_b, self.avz_b, self.ataux_b, self.atauz_b, self.atauxz_b)
        set_args(k.Adj_injSrc, *adjoint, self.res_vx_b, self.res_vz_b,
                 self.res_taux_b, self.res_tauz_b, self.res_tauxz_b, self.dxr)
        set_args(k.Adj_update_tau, *adjoint, self.rho_b, self.vdx_pml_b, self.vdz_pml_b)
        set_args(k.Adj_update_v, *adjoint, self.lam_b, self.mu_b, self.vdx_pml_b, self.vdz_pml_b)
        set_args(k.Grad, *self._field_buffers(), *adjoint, self.lam_b, self.mu_b, self.rho_b,
                 self.Gmu_b, self.g_mu_precond_b, self.Glam_b, self.g_lam_precond_b,
                 self.Grho_b, self.g_rho_precond_b)

    def _inject(self, t):
        """Inject recorded sample ``t`` (int32) of the source and record the receivers (injSrc)."""
        src_kv_x, src_kv_z, src_kt_x, src_kt_z, _ = np.float32(self.src(t))
        # Arguments 13 to 17: the source samples and the time index (the others are set once per
        # shot by _set_injection_args). The fifth sample is not used by the kernel, as in 0.1.10.
        for index, value in enumerate((src_kv_x, src_kv_z, src_kt_x, src_kt_z, t), start=13):
            self.backend.set_arg(self.knl.injSrc, index, value)
        self._launch(self.knl.injSrc)

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
        for s in range(self.ns):
            self.backend.call(self.knl.MakeAllZero, self.global_size, *self._field_buffers())
            self._set_injection_args(s)
            self._forward_shot(s, observe)
        # The checkpoint copies did not block: wait for them before the checkpoints can be read.
        self.backend.finish()
        return self.seismogram

    def _forward_shot(self, s, observe):
        """The time loop of the forward modelling of shot ``s``."""
        chpc = 0
        t_src = 0
        for t in np.arange(self.nt):
            if t % self.dt_scale == 0:
                self._inject(np.int32(t_src))
                t_src += 1

            # The four field updates; arguments set by _set_update_args.
            self._launch(self.knl.update_velx)
            self._launch(self.knl.update_velz)
            self._launch(self.knl.update_tauz)
            self._launch(self.knl.update_tauxz)

            if t in self.chp:
                # Straight into the contiguous checkpoint storage, without blocking (step 1.5). The
                # queue runs in order, so the next updates start after the copies.
                for par, buffer in zip(COMPONENTS, self._field_buffers()):
                    self.backend.to_host(self.checkpoints[par][s, chpc], buffer, blocking=False)
                chpc += 1

            if observe is not None:
                observe(s, t)
        self.make_seismogram(s)

    def make_seismogram(self, s):
        """Read the seismograms of shot ``s`` from the device into ``self.seismogram[...][:, :, s]``.

        The five reads are enqueued without blocking and waited for once (step 1.4).
        """
        buffers = (self.seismogramid_vx_b, self.seismogramid_vz_b, self.seismogramid_taux_b,
                   self.seismogramid_tauz_b, self.seismogramid_tauxz_b)
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
            (``t_src``: the recorded sample to inject next).
        """
        self.backend.call(self.knl.MakeGradZero, self.global_size,
                          self.Gmu_b, self.Glam_b, self.Grho_b,
                          self.g_mu_precond_b, self.g_lam_precond_b, self.g_rho_precond_b)

        # The arguments of the time loop do not change during the gradient: set them once (step
        # 1.3). The update kernels rebuild the forward field backwards (coeff = -1).
        self._set_update_args(np.int32(-1))
        self._set_adjoint_args()

        for s in range(self.ns):
            self.backend.call(self.knl.MakeAllZero, self.global_size, *self._field_buffers())
            self.backend.call(self.knl.MakeAllZero, self.global_size,
                              self.avx_b, self.avz_b, self.ataux_b, self.atauz_b, self.atauxz_b)
            self.residual_to_device(res, s)
            self._gradient_shot(s, observe)

    def _gradient_shot(self, s, observe):
        """The backward time loop of the gradient of shot ``s``."""
        chpc = self.nchp - 1
        t_src = np.int32(self.nt_ext)
        for t in range(self.nt - 1, 0, -1):
            if t == self.chp[chpc]:
                # Straight from the contiguous checkpoint storage, without blocking (step 1.5): the
                # checkpoints do not change during the gradient.
                for par, buffer in zip(COMPONENTS, self._field_buffers()):
                    self.backend.to_device(buffer, self.checkpoints[par][s, chpc], blocking=False)
                chpc -= 1
            else:
                # Backward propagation; arguments (coeff = -1) set by _set_update_args.
                self._launch(self.knl.update_tauxz)
                self._launch(self.knl.update_tauz)
                self._launch(self.knl.update_velz)
                self._launch(self.knl.update_velx)

            # Adjoint modelling; arguments set by _set_adjoint_args.
            if t % self.dt_scale == 0:
                self.residual_sample(t_src)
                self._launch(self.knl.Adj_injSrc)
                t_src -= 1

            self._launch(self.knl.Adj_update_tau)
            self._launch(self.knl.Adj_update_v)
            self._launch(self.knl.Grad)

            if observe is not None:
                observe(s, t, t_src)

    def residual_to_device(self, res, s):
        """Copy the residual of shot ``s``, all time samples, to the device (step 1.4).

        0.1.10 copied one time sample from the host before each injection (``make_residual``):
        five blocking copies every ``dt_scale`` steps. Now the five components of the shot are
        copied once, and ``residual_sample`` copies each sample on the device.
        """
        for par in COMPONENTS:
            # Columns s*nr to (s+1)*nr, as float32 and contiguous: the values make_residual sent.
            shot = np.ascontiguousarray(res[par][:, s * self.nr:(s + 1) * self.nr], dtype=np.float32)
            if self.res_shot_b.get(par) is None or self.backend.nbytes(self.res_shot_b[par]) != shot.nbytes:
                self.res_shot_b[par] = self.backend.allocate_empty(shot.nbytes)
            self.backend.to_device(self.res_shot_b[par], shot)

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
        self.backend.to_host(g_mu, self.Gmu_b)
        self.backend.to_host(g_lam, self.Glam_b)
        self.backend.to_host(g_rho, self.Grho_b)
        if not preconditioners:
            return g_mu, g_lam, g_rho, np.ones(shape, np.float32), np.ones(shape, np.float32), np.ones(shape, np.float32)
        precond = [np.zeros(shape, np.float32) for _ in range(3)]
        self.backend.to_host(precond[0], self.g_mu_precond_b)
        self.backend.to_host(precond[1], self.g_lam_precond_b)
        self.backend.to_host(precond[2], self.g_rho_precond_b)
        return (g_mu, g_lam, g_rho, *precond)
