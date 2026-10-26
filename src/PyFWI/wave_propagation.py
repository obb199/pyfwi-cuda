import os
from types import SimpleNamespace
import numpy as np
# from numpy.core.shape_base import block
import pyopencl as cl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import PyFWI.fwi_tools as tools
from PyFWI.backends.opencl import OpenCLBackend
import PyFWI._info as _info
from PyFWI.fwi_tools import expand_model
import PyFWI.acquisition as acq
from PyFWI.processing import prepare_residual
from PyFWI.grad_switcher import grad_lmd_to_vd


#: The five recorded components, in the order of the seismogram and residual buffers.
COMPONENTS = ("vx", "vz", "taux", "tauz", "tauxz")

#: Value of the margins around every device buffer (step 1.7 of the fork): what the out-of-bounds
#: reads of the kernels of 0.1.10 return. 0 reproduces the golden masters, which 0.1.10 computed
#: inside zero guard bands; the tests set NaN or 1.0 to see where the kernels read and write.
MARGIN_FILL = 0.0

#: Smallest margin before and after every device buffer, in bytes: the size of the guard bands
#: with which the golden masters of 0.1.10 were made (test/oob_guard.py).
MIN_MARGIN_BYTES = 1 << 16

#: Names of the OpenCL kernels that PyFWI launches (in elastic.cl, and injSrc and Adj_injSrc
#: also in elastic_crosswell.cl and elastic_surface.cl). WavePreparation retrieves each of them
#: once, in ``self.knl``.
KERNELS = ("MakeAllZero", "MakeGradZero", "injSrc", "Adj_injSrc", "update_velx", "update_velz",
           "update_tauz", "update_tauxz", "Adj_update_tau", "Adj_update_v", "Grad")


class WavePreparation:

    def __init__(self, inpa, src, 
                 rec_loc, model_shape, components=0, 
                 n_well_rec=0, chpr=10):
        '''
        A class to prepare the variable and basic functions for wave propagation.

        '''
        #TODO: work on how ypu specify the acq_type, getting n_well_rec, using that again fpr two .cl files
        keys = [*inpa]

        self.dt_scale = np.ceil(inpa['dt']/0.0006)
        self.t = inpa['t']
        self.dt_ext = inpa['dt']
        self.dt = inpa['dt'] / self.dt_scale
        self.fn = 1.0/2.0/0.0006

        self.nt = int(1 + self.t // self.dt)
        self.nt_ext = int(1 + self.t // self.dt_ext)

        self.nx = np.int32(model_shape[1])
        self.nz = np.int32(model_shape[0])

        if 'seimogram_shape' in keys:
            self.seismo_shape  = inpa['seimogram_shape']
        else: # if is not defined, make seismogram as [nt, nr * ns]
            self.seismo_shape = '2d'
            
        if 'g_smooth' in keys:
            self.g_smooth = inpa['g_smooth']
        else:
            self.g_smooth = 0

        if 'npml' in keys:
            self.npml = inpa['npml']
            pmlR = inpa['pmlR']
            pml_dir = inpa['pml_dir']
        else:
            self.npml = 0
            pmlR = 0
            pml_dir = 2
            
        if 'sdo' in keys:
            self.sdo = np.int32(inpa['sdo'] / 2)
        else:
            self.sdo = 2

        if 'rec_dis' in keys:
            rec_dis = inpa['rec_dis']
        else:
            rec_dis = (rec_loc[1,:] - rec_loc[0,:]).max()
        
        if 'acq_type' in keys:
            self.acq_type = inpa['acq_type']
        else:
            self.acq_type = 1
            
        if 'energy_balancing' in keys:
            self.energy_balancing = inpa['energy_balancing']
        else:
            self.energy_balancing = False
            
            
        # Number of samples in x- and z- direction by considering pml
        self.tnx = np.int32(self.nx + 2 * self.npml)
        self.tnz = np.int32(self.nz + 2 * self.npml)

        self.dh = np.float32(inpa['dh'])

        self.srcx = np.int32(src.i + self.npml)
        self.srcz = np.int32(src.j + self.npml)
        src_loc = np.vstack((self.srcx, self.srcz)).T

        self.src = src
        self.ns = np.int32(src.i.size)
        
        self.dxr = np.int32(rec_dis / self.dh)

        chp = int(chpr * self.nt / 100)
        self.chp = np.linspace(0, self.nt-1, chp, dtype=np.int32)
        if (len(self.chp) < 2): #& (chpr != 0)
            self.chp = np.array([1, self.nt-1])

        self.nchp = len(self.chp)
        # Take chpr into account

        self.rec_loc = rec_loc
        self.nr = rec_loc.shape[0]
        self.n_well_rec = n_well_rec
        self.n_surface_rec = self.nr - 2 * n_well_rec

        if n_well_rec ==0 and self.acq_type == 2:
            raise Exception(" Number of geophons in the wells is not defined")

        # The matrix containg the geometry of acquisittion (Never used really)
        data_guide = acq.acquisition_plan(self.ns, self.nr, src_loc, self.rec_loc, self.acq_type, n_well_rec, self.dh)

        self.data_guide_sampling = acq.discretized_acquisition_plan(data_guide, self.dh, self.npml)

        self.rec_top_left_const = np.int32(0)
        self.rec_top_left_var = np.int32(0)
        self.rec_top_right_const = np.int32(0)
        self.rec_top_right_var = np.int32(0)
        self.rec_surface_const = np.int32(0)
        self.rec_surface_var = np.int32(0)

        if self.acq_type == 0:
            self.rec_top_right_const = np.int32(rec_loc[0, 0] / self.dh + self.npml)
            self.rec_top_right_var = np.int32(rec_loc[0, 1] / self.dh + self.npml)
            self.src_cts = src.i[0]

        elif self.acq_type == 1:
            self.rec_surface_const = np.int32(rec_loc[0, 1] / self.dh + self.npml)
            self.rec_surface_var = np.int32(rec_loc[0, 0] / self.dh + self.npml)
            self.src_cts = src.j[0]

        elif self.acq_type == 2:
            a = np.int32(rec_loc[-self.n_well_rec, :]/ self.dh + self.npml)
            self.rec_top_right_const = np.copy(a[0])
            self.rec_top_right_var = np.copy(a[1])

            a = np.int32(rec_loc[self.n_well_rec, :]/ self.dh + self.npml)
            self.rec_surface_const = np.copy(a[1])
            self.rec_surface_var = np.copy(a[0])

            self.rec_top_left_const = np.int32(rec_loc[0, 0] / self.dh + self.npml)
            self.rec_top_left_var = np.int32(rec_loc[:, 1] / self.dh + self.npml)[0]
            self.src_cts = src.j[0]

        # ======== Parameters Boundary condition ======
        self.dx_pml, self.dz_pml = tools.pml_counstruction(self.tnz, self.tnx, self.dh, self.npml,
                                                     pmlR, pml_dir)

        # Checkpoints of the forward field: for each component an array (ns, nchp, tnz, tnx), so
        # that the field of one shot at one checkpoint is a contiguous block, copied straight from
        # and to the device without blocking (step 1.5 of the fork). 0.1.10 stored them as
        # (tnz, tnx, ns, nchp), where that field is scattered with a stride of ns * nchp values;
        # the attribute W still shows them in that layout (see the W property).
        self._checkpoints = {par: np.zeros((self.ns, self.nchp, self.tnz, self.tnx), dtype=np.float32)
                             for par in COMPONENTS}

        self.D = tools.Fdm(order=self.sdo * 2)

        self.components = components

        # To call openCl
        # Every OpenCL call goes through the backend (step 2.1 of the fork, PyFWI.backends.opencl).
        # The device: inpa['platform'] and inpa['device'] (0 when missing), otherwise the
        # PYOPENCL_CTX environment variable, otherwise the first device of the first platform; one
        # context per device for the whole process, one queue per propagator.
        self._backend = OpenCLBackend(inpa.get("platform"), inpa.get("device"))
        self.ctx = self._backend.context
        self.queue = self._backend.queue

        kernel, kernel_crosswell, kernel_surface = self.kernel_caller()

        self.prg_cw = self._backend.build(kernel_crosswell)
        self.prg = self._backend.build(kernel)
        self.prg_surf = self._backend.build(kernel_surface)

        if self.acq_type == 0:
            self.prg.injSrc = self.prg_cw.injSrc
            self.prg.Adj_injSrc = self.prg_cw.Adj_injSrc
        elif self.acq_type == 1:
            self.prg.injSrc = self.prg_surf.injSrc
            self.prg.Adj_injSrc = self.prg_surf.Adj_injSrc

        # The kernel objects, retrieved once and reused at every launch (step 1.3 of the fork).
        # 0.1.10 wrote ``self.prg.update_velx(...)`` at every launch, and each ``self.prg.<name>``
        # created a new kernel object (clCreateKernel), several times per time step; PyOpenCL
        # 2026 also warns about it (RepeatedKernelRetrieval). injSrc and Adj_injSrc come from the
        # program of the acquisition (assigned to self.prg just above for acq_type 0 and 1).
        self.knl = SimpleNamespace(**{name: self._backend.kernel(self.prg, name) for name in KERNELS})
        # Global size of every launch: one work item per grid point, absorbing boundary included.
        self.global_size = (int(self.tnz), int(self.tnx))

        self.mf = cl.mem_flags

        # Margins around every device buffer (step 1.7 of the fork). The kernels of 0.1.10 read and
        # write outside their buffers: the stencils up to 4 grid rows before and after a field, the
        # unbounded loop of the right well (acq_type 0 and 2) up to rec_top_right_var rows past the
        # end of a field, and up to tnz values past the end of a seismogram or residual. In 0.1.10
        # those accesses hit whatever lay next to the buffer (garbage, NaN on POCL, or a
        # CL_OUT_OF_RESOURCES crash); now they hit margins of MARGIN_FILL inside the allocation.
        # Sub-buffers must start at a multiple of the base-address alignment of the device.
        align = self._backend.alignment
        needed = max(MIN_MARGIN_BYTES, (int(self.rec_top_right_var) + 8) * int(self.tnx) * 4,
                     (int(self.tnz) + 8) * 4)
        self.margin_bytes = -(-needed // align) * align
        # The allocation (margins included) of every device buffer, by attribute name.
        self._allocations = {}

        # Buffer for residuals
        res = np.zeros((np.int32(self.nr))).astype(np.float32, order='C')

        self._set_buffer("res_vx_b", res)
        self._set_buffer("res_vz_b", res)
        self._set_buffer("res_taux_b", res)
        self._set_buffer("res_tauz_b", res)
        self._set_buffer("res_tauxz_b", res)
        # The residual of one shot, every time sample, per component (residual_to_device);
        # created at the first gradient and reused while its size does not change.
        self.res_shot_b = {}


        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        # Buffer for forward modelling
        self._set_buffer("vx_b", v)
        self._set_buffer("vz_b", v)
        self._set_buffer("taux_b", v)
        self._set_buffer("tauz_b", v)
        self._set_buffer("tauxz_b", v)

        # Buffer for seismograms
        seismogram_id = np.zeros((self.nt_ext, self.nr)).astype(np.float32, order='C')

        self._set_buffer("seismogramid_vx_b", seismogram_id)
        self._set_buffer("seismogramid_vz_b", seismogram_id)
        self._set_buffer("seismogramid_taux_b", seismogram_id)
        self._set_buffer("seismogramid_tauz_b", seismogram_id)
        self._set_buffer("seismogramid_tauxz_b", seismogram_id)

        # Make a list for seismograms
        self.seismogram = {
            'vx': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'vz': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'taux': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'tauz': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'tauxz': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
        }

    def _set_buffer(self, name, host):
        """Create the device buffer ``self.<name>`` holding ``host``, with margins (step 1.7 of the fork).

        The allocation is ``[margin | the data of host | margin]``, the margins filled with
        ``MARGIN_FILL`` and ``margin_bytes`` long; ``self.<name>`` is the sub-buffer of the data,
        which the kernels use as before, and ``self._allocations[name]`` the whole allocation
        (``margins_changed`` reads it). Replacing a buffer releases its old allocation.

        Parameters
        ----------
        name : str
            Attribute name, e.g. ``"vx_b"``.
        host : numpy.ndarray
            Initial content (float32 in PyFWI), copied to the device.
        """
        buffer, self._allocations[name] = self._backend.allocate(host, self.margin_bytes, MARGIN_FILL)
        setattr(self, name, buffer)

    def margins_changed(self):
        """Names of the device buffers whose margins no longer hold ``MARGIN_FILL`` (step 1.7).

        A kernel wrote outside that buffer (or, with a NaN fill, accumulated into it). Used by the
        tests of the out-of-bounds accesses; it reads every allocation back from the device.
        """
        changed = []
        margin = self.margin_bytes
        for name, allocation in self._allocations.items():
            whole = np.empty(self._backend.nbytes(allocation) // 4, np.float32)
            self._backend.to_host(whole, allocation)
            data_bytes = self._backend.nbytes(getattr(self, name))
            margins = np.concatenate([whole[:margin // 4], whole[(margin + data_bytes) // 4:]])
            intact = np.isnan(margins).all() if np.isnan(MARGIN_FILL) else (margins == np.float32(MARGIN_FILL)).all()
            if not intact:
                changed.append(name)
        return changed

    @property
    def W(self):
        """Checkpoints of the forward field, ``{component: array (tnz, tnx, ns, nchp)}``, as in 0.1.10.

        ``W['vx'][:, :, s, c]`` is the field ``vx`` of shot ``s`` at checkpoint ``c`` (time step
        ``chp[c]``), absorbing boundary included. Since step 1.5 the arrays are views of the
        checkpoint storage (``_checkpoints``, layout ``(ns, nchp, tnz, tnx)``): same shape, dtype,
        axis order and values as in 0.1.10, and writing into them changes the checkpoints, as
        before. The dictionary itself is new at each access, so replacing one of its entries has
        no effect; assign ``W`` instead (setter below).
        """
        return {par: store.transpose(2, 3, 0, 1) for par, store in self._checkpoints.items()}

    @W.setter
    def W(self, value):
        """Replace the checkpoints with ``{component: array (tnz, tnx, ns, nchp)}``, the 0.1.10 layout.

        Used by ``PyFWI.optimization`` (``FO_waves.W = copy.deepcopy(W)``). The arrays are copied
        into the storage layout, as float32.
        """
        self._checkpoints = {par: np.ascontiguousarray(np.asarray(array, dtype=np.float32).transpose(2, 3, 0, 1))
                             for par, array in value.items()}

    def adjoint_buffer_preparing(self):
        # Buffer for gradient
        g_mu = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self._set_buffer("Gmu_b", g_mu)

        g_lam = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self._set_buffer("Glam_b", g_lam)

        g_rho = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self._set_buffer("Grho_b", g_rho)

        g_mu_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self._set_buffer("g_mu_precond_b", g_mu_precond)

        self._set_buffer("g_lam_precond_b", g_mu_precond)

        self._set_buffer("g_rho_precond_b", g_mu_precond)

        # Buffer for backward modelling
        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self._set_buffer("avx_b", v)
        self._set_buffer("avz_b", v)
        self._set_buffer("ataux_b", v)
        self._set_buffer("atauz_b", v)
        self._set_buffer("atauxz_b", v)

    def gradient_reading(self):
        g_mu = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        g_lam = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        g_rho = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        self._backend.to_host(g_mu, self.Gmu_b)
        self._backend.to_host(g_lam, self.Glam_b)
        self._backend.to_host(g_rho, self.Grho_b)

        if self.energy_balancing:
            g_mu_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_lam_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_rho_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

            self._backend.to_host(g_mu_precond, self.g_mu_precond_b)
            self._backend.to_host(g_lam_precond, self.g_lam_precond_b)
            self._backend.to_host(g_rho_precond, self.g_rho_precond_b)
        else:
            g_mu_precond = np.ones((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_lam_precond = np.ones((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_rho_precond = np.ones((self.tnz, self.tnx)).astype(np.float32, order='C')

        def denom2factor(precond0):
            precond = np.copy(precond0)

            factor = np.zeros((self.nz, self.nx), np.float32)

            denom = np.sqrt(precond)
            if self.acq_type == 0:
                denom = denom[self.npml + sdo:self.tnz - self.npml - sdo, self.npml + sdo + self.src_cts + 2:self.tnx - self.npml - sdo]
                factor[sdo:-sdo, sdo + self.src_cts + 2:-sdo] = 1 / denom
            else:
                denom = denom[self.npml + sdo + self.src_cts + 2:self.tnz - self.npml - sdo, self.npml + sdo:self.tnx - self.npml - sdo]
                factor[sdo + self.src_cts + 2:-sdo, sdo:-sdo] = 1 / denom

            factor = factor / np.abs(factor).max()

            return np.copy(factor)

        sdo = self.sdo

        factor_rho = denom2factor(g_rho_precond)
        grho = g_rho[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml] * factor_rho

        factor_mu = denom2factor(g_mu_precond)
        gmu = g_mu[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml] * factor_mu

        factor_lam = denom2factor(g_lam_precond)
        glam = g_lam[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml] * factor_lam

        return glam, gmu, grho


    def make_seismogram(self, s, t):
        """
        This function read the seismogram buffer and put its value in the
        right place in main seismogram matrix based on source and the
        time step.

        Parameters
        ----------
            s : int
                Number of current acive source.

            t : float
                Current time step.
        """
        # The five reads are enqueued without blocking and waited for once (step 1.4 of the
        # fork); 0.1.10 waited after each of them.
        buffers = (self.seismogramid_vx_b, self.seismogramid_vz_b, self.seismogramid_taux_b,
                   self.seismogramid_tauz_b, self.seismogramid_tauxz_b)
        host = {par: np.empty((self.nt_ext, self.nr), np.float32) for par in COMPONENTS}
        events = [self._backend.to_host(host[par], buffer, blocking=False)
                  for par, buffer in zip(COMPONENTS, buffers)]
        self._backend.wait(events)
        for par in COMPONENTS:
            self.seismogram[par][:, :, s] = host[par]

    def make_residual(self, res, s, t):
        """
        This function reads the inject the residual to residual buffer based on source and the
        time step.

        Parameters
        ----------
            res: list
                list containing the residual of all parameters

            s : int
                Number of current acive source.

            t : float
                Current time step.
        """
        # Injection data into opencl
        res_src_vx = (res['vx'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        self._backend.to_device(self.res_vx_b, res_src_vx)

        res_src_vz = (res['vz'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        self._backend.to_device(self.res_vz_b, res_src_vz)

        res_src_taux = (res['taux'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        self._backend.to_device(self.res_taux_b, res_src_taux)

        res_src_tauz = (res['tauz'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        self._backend.to_device(self.res_tauz_b, res_src_tauz)

        res_src_tauxz = (res['tauxz'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        self._backend.to_device(self.res_tauxz_b, res_src_tauxz)

    def residual_to_device(self, res, s):
        """Copy the residual of shot ``s``, all time samples, to the device (step 1.4 of the fork).

        0.1.10 copied one time sample of the residual from the host before each injection
        (``make_residual``): five blocking copies every ``dt_scale`` steps, 750 per shot in the
        example of the documentation. Now the five components of the shot are copied once, and
        ``residual_sample`` copies each time sample on the device.

        Parameters
        ----------
        res : dict
            Residual of every shot, ``{component: array (time samples, nr * ns)}``, as prepared
            by ``gradient`` (``prepare_residual``).
        s : int
            Index of the shot.
        """
        for par in COMPONENTS:
            # Columns s*nr to (s+1)*nr, as float32 and contiguous: the same values make_residual sent.
            shot = np.ascontiguousarray(res[par][:, s * self.nr:(s + 1) * self.nr], dtype=np.float32)
            if self.res_shot_b.get(par) is None or self._backend.nbytes(self.res_shot_b[par]) != shot.nbytes:
                self.res_shot_b[par] = self._backend.allocate_empty(shot.nbytes)
            self._backend.to_device(self.res_shot_b[par], shot)

    def residual_sample(self, t):
        """Copy time sample ``t - 1`` of the shot's residual into the buffers that Adj_injSrc reads.

        The copies stay on the device and do not make the host wait. Adj_injSrc keeps reading
        ``res_vx_b``... of ``nr`` values, as in 0.1.10, so that what its unbounded receiver loops
        read past the end of them is unchanged (``test_known_bugs.py``).

        Parameters
        ----------
        t : int
            Time index used by 0.1.10 in ``make_residual(res, s, t)``: the sample ``t - 1`` is copied.
        """
        row = int(self.nr) * 4  # bytes of one time sample of one component
        sample = int(t) - 1
        if sample < 0:  # as NumPy indexing in make_residual: -1 is the last sample
            sample += self._backend.nbytes(self.res_shot_b["vx"]) // row
        targets = (self.res_vx_b, self.res_vz_b, self.res_taux_b, self.res_tauz_b, self.res_tauxz_b)
        for par, target in zip(COMPONENTS, targets):
            self._backend.copy(target, self.res_shot_b[par], row, sample * row)

    def pml_preparation(self, v_max):

        vdx_pml = self.dx_pml * v_max
        vdz_pml = self.dz_pml * v_max
        self._set_buffer("vdx_pml_b", vdx_pml)
        self._set_buffer("vdz_pml_b", vdz_pml)

    def kernel_caller(self):
        '''
            This function is used to specify the constants for use in openCl's
            files and choose the asked openCl source to run based on the assumptions
            of problem.

            The coeeficients are based on Lavendar, 1988 and Hasym et al., 2014.

            Returns
            -------

                kernel_source : str
                    macro plus the openCl's source.
        '''
        if self.sdo == 2:  # For O4
            c1 = 9 / 8
            c2 = -1 / 24
            c3 = 0
            c4 = 0
        elif self.sdo == 4:  # For O8
            c1 = 1715 / 1434  #1.2257  #
            c2 = -114 / 1434  # - 0.099537  #
            c3 = 14 / 1434  # 0.018063  #
            c4 = -1 / 1434  # - 0.0026274  #

        macro = """
            #define Nz	   %d
            #define Nx     %d
            #define Ns     %d
            #define Nr     %d
            #define dt     %f
            #define dx     %f
            #define dz     %f
            #define npml   %d
            #define n_main_rec %d
            #define n_extera_rec %d

            #define sdo     %d
            #define c1     %f
            #define c2     %f
            #define c3     %f
            #define c4     %f

            #define center	   (i)*Nx + (j)

            #define below    (i+1)*Nx + (j)
            #define below2   (i+2)*Nx + (j)
            #define below3   (i+3)*Nx + (j)
            #define below4   (i+4)*Nx + (j)

            #define above     (i-1)*Nx + (j)
            #define above2    (i-2)*Nx + (j)
            #define above3    (i-3)*Nx + (j)
            #define above4    (i-4)*Nx + (j)

            #define right     (i)*Nx + (j+1)
            #define right2    (i)*Nx + (j+2)
            #define right3    (i)*Nx + (j+3)
            #define right4    (i)*Nx + (j+4)

            #define left    (i)*Nx + (j-1)
            #define left2   (i)*Nx + (j-2)
            #define left3   (i)*Nx + (j-3)
            #define left4   (i)*Nx + (j-4)

            #define src_depth %d
            #define rec_top_left_const  %d
            #define rec_top_left_var %d
            #define rec_top_right_const %d
            #define rec_top_right_var %d
            #define rec_surface_const %d
            #define rec_surface_var %d


                //float Diff_Forward(float , float , float , float , int );
                //float Diff_Backward(float , float , float , float , int );
        """ % (self.tnz, self.tnx, self.ns, self.nr, self.dt, self.dh, self.dh, self.npml,
               self.n_surface_rec, self.n_well_rec, self.sdo,
               c1, c2, c3, c4, self.src_cts,
               self.rec_top_left_const, self.rec_top_left_var,
               self.rec_top_right_const, self.rec_top_right_var,
               self.rec_surface_const, self.rec_surface_var)

        # Decide on openCl file based on medium type
        cl_file_name = "elastic.cl"

        # Call the openCl file
        path = os.path.dirname(__file__) + "/"
        f = open(path + cl_file_name, 'r')
        f_cw = open(path + 'elastic_crosswell.cl', 'r')
        f_surf = open(path + 'elastic_surface.cl', 'r')

        fstr = "".join(f.readlines())
        fstr_cw = "".join(f_cw.readlines())
        fstr_surf = "".join(f_surf.readlines())

        kernel_source = macro + fstr
        kernel_src_crosswell = macro + fstr_cw
        kernel_surface = macro + fstr_surf

        return kernel_source , kernel_src_crosswell, kernel_surface

    def initial_wavefield_plot(self, model, plot_type="Forward"):
        """
        A function to initialize the the plot for wave
        propagation visulizing

        Parameters
        ----------
            plot_type: string optional = "Forward"
                Specify if we want to show Forward modelloing or Backward.

        """
        key = [*model][0]
        show_purpose = np.zeros((self.tnz - 2 * self.npml, self.tnx - 2 * self.npml))
        if plot_type == "Forward":
            fig, ax = plt.subplots(1, 1)
            self.__im0 = ax.imshow(model[key][self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml], cmap='jet',
                                   vmin=show_purpose.min())
            fig.colorbar(self.__im0, extend='both', label=key, shrink=0.3)
            self.__im0 = ax.imshow(show_purpose, alpha=.65, cmap="seismic")

        elif plot_type == "Backward":
            fig, ax = plt.subplots(1, 2)
            self.__im1 = ax[1].imshow(show_purpose)
            self.__im0 = ax[0].imshow(show_purpose)
        self.__stitle = fig.suptitle('')

    def plot_propagation(self, wave1, t, wave2=None):
        """
        This function is used to shpw the propagation wave with time.

        Parameters
        ----------

        wave1 : float32
            The wave that we are going to show

        t : float32
            Time step

        wave2 : float32, optional = None
            The second wave which is used when we want to show the propagation
            of backward wave as wave1 and adjoint wave as wave2.

        """

        wave1 = wave1[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml]

        self.__im0.set_data(wave1)  # [self.Npml:self.TNz - self.Npml, self.Npml:self.TNx - self.Npml])
        self.__im0.set_clim(wave1.min(), wave1.max())

        if wave2 is not None:
            wave2 = wave2[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml]
            self.__im1.set_data(wave2)
            self.__im1.set_clim(wave2.min() / 20, wave2.max() / 20)
        self.__stitle.set_text(
            't = {0:6.3f} (st. no {1:d}/{2:d})'.format(t * self.dt, t + 1, self.nt_ext))
        plt.pause(0.1)


    def elastic_buffers(self, model):
        '''
        Model hast contain vp, vs, and rho
        '''

        self.mu = model['rho'] * (model['vs'] ** 2)

        self.lam = model['rho'] * (model['vp'] ** 2) - 2 * self.mu

        self.rho = model['rho']

        self.vp = model['vp']

        self.vs = model['vs']

        self._set_buffer("rho_b", 1 / model['rho'])

        self._set_buffer("mu_b", self.mu)
        self._set_buffer("lam_b", self.lam)


class WavePropagator(WavePreparation):
    """
    wave_propagator is a class to handle the forward modeling and gradient calculation.

    [extended_summary]

    Parameters
    ----------
    inpa : dict
        A dictionary including most of the required inputs
    src : class
        Source object
    rec_loc : ndarray
        Location of the receivers
    model_shape : tuple
        Shape of the model
    n_well_rec: int
        Number of receivers in the well
    chpr : percentage
        Checkpoint ratio in percentage
    component:
        Seismic output
    """
    def __init__(self, inpa, src, rec_loc, 
                 model_shape, components=0, 
                 n_well_rec=0, chpr=0):
        WavePreparation.__init__(self, inpa=inpa, src=src, 
                                 rec_loc=rec_loc, model_shape=model_shape, 
                                 components=components, 
                                 n_well_rec=n_well_rec, chpr=chpr)

    def forward_propagator(self, model):
        """ This function is in charge of forward modelling for acoustic case

        Parameters
        ---------
            model: dictionary
                A dictionary containing p-wave velocity and density

        """
        coeff = np.int32(+1)

        if self.forward_show:
            self.initial_wavefield_plot(model)

        # The arguments that do not change during the time loop are set once (step 1.3).
        self._set_update_args(coeff)
        for s in range(self.ns):
            self._backend.call(self.knl.MakeAllZero, self.global_size,
                               self.vx_b, self.vz_b,
                               self.taux_b, self.tauz_b, self.tauxz_b)
            self._set_injection_args(s)

            self.__kernel(s, coeff)

        # The checkpoint copies did not block: wait for them before W can be read.
        self._backend.finish()

        if self.acq_type == 2:
            for par in self.seismogram:
                self.seismogram[par][:, :self.n_well_rec] = np.flip(self.seismogram[par][:, :self.n_well_rec], axis=1)
        return self.seismogram
    
    def _field_buffers(self):
        """The five device buffers of the forward field, in the order of ``COMPONENTS``."""
        return (self.vx_b, self.vz_b, self.taux_b, self.tauz_b, self.tauxz_b)

    def _launch(self, kernel):
        """Launch ``kernel`` on the whole grid with the arguments set before (``set_args``/``set_arg``).

        ``pyopencl.enqueue_nd_range_kernel`` only enqueues, whereas calling the kernel object
        (``kernel(queue, size, None, *args)``) also sets every argument again. Used for the
        launches of the time loops (step 1.3 of the fork).
        """
        self._backend.launch(kernel, self.global_size)

    def _set_update_args(self, coeff):
        """Set the arguments of the four update kernels of the elastic fields.

        Parameters
        ----------
        coeff : numpy.int32
            +1 to propagate forward in time (forward modelling), -1 to go backward (the forward
            field rebuilt during the gradient).

        Called at the start of each forward modelling and of each gradient, after the model and
        PML buffers of that call exist (``elastic_buffers`` and ``pml_preparation`` create new
        ones at every forward modelling).
        """
        self._backend.set_args(self.knl.update_velx, coeff, self.vx_b, self.taux_b, self.tauxz_b, self.rho_b,
                               self.vdx_pml_b, self.vdz_pml_b)
        self._backend.set_args(self.knl.update_velz, coeff, self.vz_b, self.tauz_b, self.tauxz_b, self.rho_b,
                               self.vdx_pml_b, self.vdz_pml_b)
        self._backend.set_args(self.knl.update_tauz, coeff, self.vx_b, self.vz_b, self.taux_b, self.tauz_b,
                               self.lam_b, self.mu_b, self.vdx_pml_b, self.vdz_pml_b)
        self._backend.set_args(self.knl.update_tauxz, coeff, self.vx_b, self.vz_b, self.tauxz_b, self.mu_b,
                               self.vdx_pml_b, self.vdz_pml_b)

    def _set_injection_args(self, s):
        """Set the arguments of injSrc that stay the same for every time step of source ``s``.

        The last five arguments (the four source samples and the time index) change at each
        injection and are set by ``__injSrc``; zeros are placeholders until then.
        """
        zero = np.float32(0)
        self._backend.set_args(self.knl.injSrc, self.vx_b, self.vz_b,
                               self.taux_b, self.tauz_b, self.tauxz_b,
                               self.seismogramid_vx_b, self.seismogramid_vz_b,
                               self.seismogramid_taux_b, self.seismogramid_tauz_b, self.seismogramid_tauxz_b,
                               self.dxr,
                               self.srcx[s], self.srcz[s],
                               zero, zero, zero, zero, np.int32(0))

    def __injSrc(self, t, s):
        src_kv_x, src_kv_z, src_kt_x, src_kt_z, src_kt_xz = np.float32(self.src(t))

        # Arguments 13 to 17: the source samples and the time index t (the others are set once
        # per source by _set_injection_args). src_kt_xz is not used by the kernel, as in 0.1.10.
        for index, value in enumerate((src_kv_x, src_kv_z, src_kt_x, src_kt_z, t), start=13):
            self._backend.set_arg(self.knl.injSrc, index, value)
        self._launch(self.knl.injSrc)

    def __update_fwd(self, coeff):
        # Arguments set by _set_update_args at the start of the forward modelling.
        self._launch(self.knl.update_velx)
        self._launch(self.knl.update_velz)
        self._launch(self.knl.update_tauz)
        self._launch(self.knl.update_tauxz)
            
    def __kernel(self, s, coeff=+1):
        showpurose = np.zeros((self.tnz, self.tnx), dtype=np.float32)
        chpc = 0
        t_src = 0
        for t in np.arange(self.nt):
            if t % self.dt_scale == 0:
                self.__injSrc(np.int32(t_src), s)
                t_src += 1

            self.__update_fwd(coeff=coeff)

            if t in self.chp:
                # Straight into the contiguous checkpoint storage, without blocking (step 1.5).
                # The queue runs in order, so the next updates start after the copies; the host
                # waits once, at the end of forward_propagator.
                for par, buffer in zip(COMPONENTS, self._field_buffers()):
                    self._backend.to_host(self._checkpoints[par][s, chpc], buffer, blocking=False)

                chpc += 1


            if self.forward_show and np.remainder(t, 20) == 0:
                self._backend.to_host(showpurose, self.vx_b)
                self.plot_propagation(showpurose, t)
        self.make_seismogram(s, t)
        
    def __adjoint_modelling_per_source(self, res):
        self._backend.call(self.knl.MakeGradZero, self.global_size,
                           self.Gmu_b, self.Glam_b, self.Grho_b,
                           self.g_mu_precond_b, self.g_lam_precond_b, self.g_rho_precond_b)

        # The arguments of the time loop do not change during the gradient: set them once
        # (step 1.3). The update kernels rebuild the forward field backwards (coeff = -1).
        self._set_update_args(np.int32(-1))
        self._set_adjoint_args()

        for s in range(self.ns):
            self._backend.call(self.knl.MakeAllZero, self.global_size,
                               self.vx_b, self.vz_b,
                               self.taux_b, self.tauz_b, self.tauxz_b)

            self._backend.call(self.knl.MakeAllZero, self.global_size,
                               self.avx_b, self.avz_b,
                               self.ataux_b, self.atauz_b, self.atauxz_b)
            self.residual_to_device(res, s)

            self.__kernel_gradient(res, s)

    def _set_adjoint_args(self):
        """Set the arguments of the adjoint kernels (Adj_injSrc, Adj_update_tau, Adj_update_v, Grad).

        Called at the start of each gradient, after ``adjoint_buffer_preparing`` has created the
        adjoint and gradient buffers. The residual buffers keep their identity; only their content
        changes (``make_residual``).
        """
        self._backend.set_args(self.knl.Adj_injSrc, self.avx_b, self.avz_b,
                               self.ataux_b, self.atauz_b, self.atauxz_b,
                               self.res_vx_b, self.res_vz_b,
                               self.res_taux_b, self.res_tauz_b, self.res_tauxz_b,
                               self.dxr)
        self._backend.set_args(self.knl.Adj_update_tau, self.avx_b, self.avz_b,
                               self.ataux_b, self.atauz_b, self.atauxz_b,
                               self.rho_b,
                               self.vdx_pml_b, self.vdz_pml_b)
        self._backend.set_args(self.knl.Adj_update_v, self.avx_b, self.avz_b,
                               self.ataux_b, self.atauz_b, self.atauxz_b,
                               self.lam_b, self.mu_b,
                               self.vdx_pml_b, self.vdz_pml_b)
        self._backend.set_args(self.knl.Grad, self.vx_b, self.vz_b,
                               self.taux_b, self.tauz_b, self.tauxz_b,
                               self.avx_b, self.avz_b,
                               self.ataux_b, self.atauz_b, self.atauxz_b,
                               self.lam_b, self.mu_b, self.rho_b,
                               self.Gmu_b, self.g_mu_precond_b,
                               self.Glam_b, self.g_lam_precond_b,
                               self.Grho_b, self.g_rho_precond_b)

    def __kernel_gradient(self, res, s, coeff=-1):
        chpc = self.nchp - 1

        coeff = np.int32(coeff)

        vx_show = np.zeros((self.tnz, self.tnx), dtype=np.float32)
        adj_vx_show = np.zeros((self.tnz, self.tnx), dtype=np.float32)

        t_src = np.int32(self.nt_ext)
        # time loop
        for t in range(self.nt - 1, 0, -1):  # range(self.nt-1,self.nt-2,-1):#
            if t == self.chp[chpc]:
                # Straight from the contiguous checkpoint storage, without blocking (step 1.5):
                # the checkpoints do not change during the gradient.
                for par, buffer in zip(COMPONENTS, self._field_buffers()):
                    self._backend.to_device(buffer, self._checkpoints[par][s, chpc], blocking=False)

                chpc -= 1

            else:
                """ Backward propagation  """
                # Arguments (coeff = -1) set by _set_update_args at the start of the gradient.
                self._launch(self.knl.update_tauxz)
                self._launch(self.knl.update_tauz)
                self._launch(self.knl.update_velz)
                self._launch(self.knl.update_velx)

            """ Adjoint modeling """
            # Arguments set by _set_adjoint_args at the start of the gradient.
            if t % self.dt_scale == 0:
                self.residual_sample(t_src)  # 0.1.10: self.make_residual(res, s, t_src)

                self._launch(self.knl.Adj_injSrc)
                t_src -= 1

            self._launch(self.knl.Adj_update_tau)
            self._launch(self.knl.Adj_update_v)
            self._launch(self.knl.Grad)

            # Plotting wave propagation
            if self.backward_show and (np.remainder(t, 20) == 0 or t == self.nt_ext - 2):
                self._backend.to_host(vx_show, self.taux_b)

                self._backend.to_host(adj_vx_show, self.ataux_b)

                self.plot_propagation(vx_show, t_src, adj_vx_show)


    def forward_modeling(self, model0, show=False):
        """
        forward_modeling performs the forward modeling.


        Parameters
        ----------
        model0 : dict
            The earth model
        show : bool, optional
            True if you desire to see the propagation of the wave, by default False

        Returns
        -------
        dict
            Seismic section
        """
        self.forward_show = show
        # The summary of PyFWI.info() goes to the "PyFWI" logger (INFO) at the first run (step 1.8).
        _info.log_first_run(self.ctx.devices)
        model = model0.copy()

        for params in model:
            # model[params] = model[params]  # To avoid sticking BC. to the original model
            model[params] = expand_model(model[params], self.tnz, self.tnx, self.npml)

        self.pml_preparation(model['vp'].max())
        self.elastic_buffers(model)
        seismo = self.forward_propagator(model)
        data = acq.seismic_section(seismo, self.components, shape=self.seismo_shape)
        return data

    def gradient(self, res, show=False, parameterization='dv'):
        """
        gradient estimates the gradient using adjoint-state method.

        Parameters
        ----------
        res : dict
            The adjoint of the derivative of ost function with respect to wavefield
        show : bool, optional
            True if you desire to see the backward wave propagation, by default False
        parameterization : str, optional
            Specify the parameterization for output, by default 'dv'

        Returns
        -------
        dict
            Gradient
        """
        self.backward_show = show
        self.adjoint_buffer_preparing()

        # To reorder the receivers in the left well
        if self.acq_type == 2:
            for par in self.seismogram:
                self.seismogram[par][:, :self.n_well_rec] = np.flip(self.seismogram[par][:, :self.n_well_rec], axis=1)

        res = prepare_residual(res, 1)
        if show:
            self.initial_wavefield_plot({'vp':self.vp}, plot_type="Backward")

        self.__adjoint_modelling_per_source(res)

        glam, gmu, grho0 = self.gradient_reading()

        if parameterization == 'dv':
            gvp, gvs, grho = grad_lmd_to_vd(glam, gmu, grho0,
                                              self.lam[self.npml: self.tnz-self.npml, self.npml: self.tnx-self.npml],
                                              self.mu[self.npml: self.tnz-self.npml, self.npml: self.tnx-self.npml],
                                              self.rho[self.npml: self.tnz-self.npml, self.npml: self.tnx-self.npml])
            final_grad = {'vp': gaussian_filter(gvp, self.g_smooth),
                          'vs': gaussian_filter(gvs, self.g_smooth),
                          'rho': gaussian_filter(grho, self.g_smooth)
                          }
        else:
            final_grad = {'lam': gaussian_filter(glam, self.g_smooth),
                          'mu': gaussian_filter(gmu, self.g_smooth),
                          'rho': gaussian_filter(grho0, self.g_smooth)
                          }

        return final_grad


if __name__ == "__main__":
    import PyFWI.model_dataset as md
    import PyFWI.acquisition as acq
    import PyFWI.seiplot as splt
    import PyFWI.fwi as fwi
    
    GRADIENT = 1
    INVERSION = 0

    model_gen = md.ModelGenerator('louboutin') #

    model = model_gen()
    model['vs'] *=0
    # model_gen.show(['vs'])
    model_shape = model[[*model][0]].shape

    inpa = {}
    # Number of pml layers
    inpa['npml'] = 20
    inpa['pmlR'] = 1e-5
    inpa['pml_dir'] = 2
    # inpa['device'] = 0
    inpa['energy_balancing'] = False
    inpa['seisout'] = 0

    chpr = 100
    sdo = 4
    fdom = 25
    inpa['fn'] = 125
    vp = model['vp']
    D = tools.Fdm(order=sdo)
    dh = vp.min()/(D.dh_n * inpa['fn'])
    dh = 2.
    inpa['dh'] = dh

    dt = D.dt_computation(vp.max(), inpa['dh'])
    inpa['dt'] = dt

    # print(f'{dh = } ........... {dt = }')
    inpa['t'] = 0.36

    offsetx = inpa['dh'] * model_shape[1]
    depth = inpa['dh'] * model_shape[0]

    inpa['rec_dis'] = 2.  # inpa['dh']
    ns = 1
    inpa['acq_type'] = 0

    src_loc, rec_loc, n_surface_rec, n_well_rec = acq.acq_parameters(ns, inpa['rec_dis'], offsetx, depth, inpa['dh'], sdo, inpa['acq_type'])

    src = acq.Source(src_loc, inpa['dh'], inpa['dt'])
    src.Ricker(fdom)

    W = WavePropagator(inpa, src, rec_loc, model_shape, n_well_rec, chpr=0, components=inpa['seisout'])
    d_obs = W.forward_modeling(model, False)
    d_obs = prepare_residual(d_obs, 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    splt.seismic_section(ax, d_obs['taux'], vmin=d_obs['taux'].min() / 5, vmax=d_obs['taux'].max() / 5)

    m0 = model_gen(vintage=1, smoothing=True)

    if GRADIENT:
        Lam = WavePropagator(inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr, components=inpa['seisout'])
        d_est = Lam.forward_modeling(m0, show=False)
        d_est = prepare_residual(d_est, 1)

        CF = tools.CostFunction('l2')
        rms, res = CF(d_est, d_obs)

        print(rms)

        grad = Lam.gradient(res, show=False)

        splt.earth_model(grad, cmap='jet')
        plt.show()
        a=1

    elif INVERSION:

        FWI = fwi.FWI(d_obs, inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr, components=4, param_functions=None)
        inverted_model, rms = FWI(m0, method=2, iter=[3], freqs=[25], n_params=1, k_0=1, k_end=2)


        splt.earth_model(inverted_model, cmap='jet')
        plt.show(block=False)

        plt.figure()
        plt.plot(rms/max(rms))
        plt.show()
