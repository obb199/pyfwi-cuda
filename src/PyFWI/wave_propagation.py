import os
import numpy as np
# from numpy.core.shape_base import block
import pyopencl as cl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import PyFWI.fwi_tools as tools
from PyFWI.backends.opencl import OpenCLBackend
from PyFWI.backends.selection import select_backend, select_cuda_kernels, select_numerics
from PyFWI.engine.propagation import PropagationEngine, checkpoint_steps
import PyFWI._info as _info
from PyFWI.fwi_tools import expand_model
import PyFWI.acquisition as acq
from PyFWI.processing import prepare_residual
from PyFWI.grad_switcher import grad_lmd_to_vd


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

        # Time steps whose forward field is stored for the gradient (PyFWI.engine.checkpoint_steps).
        self.chp = checkpoint_steps(chpr, self.nt)

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

        self.D = tools.Fdm(order=self.sdo * 2)

        self.components = components

        # To call openCl
        # Every OpenCL call goes through the backend (step 2.1 of the fork, PyFWI.backends.opencl).
        # The device: inpa['platform'] and inpa['device'] (0 when missing), otherwise the
        # PYOPENCL_CTX environment variable, otherwise the first device of the first platform; one
        # context per device for the whole process, one queue per propagator.
        # Which backend runs this propagator (step 2.2 of the fork, PyFWI.backends.selection):
        # inpa['backend'], else PYFWI_BACKEND, else OpenCL. The CUDA backend (step 2.3) runs the
        # same kernel files; with it W.ctx and W.queue, which are OpenCL objects, are None.
        self.backend_selection = select_backend(inpa)
        # The numeric mode (step 2.4 of the fork): "legacy" computes what PyFWI 0.1.10 computes,
        # "fixed" corrects it (see kernel_caller); inpa['numerics'], else PYFWI_NUMERICS, else the
        # default of the backend (OpenCL: legacy, CUDA: fixed).
        self.numerics = select_numerics(inpa, self.backend_selection.backend)
        if self.backend_selection.backend == "cuda":
            from PyFWI.backends.cuda import CUDABackend
            self._backend = CUDABackend(self.backend_selection.cuda_ordinal)
        else:
            self._backend = OpenCLBackend(inpa.get("platform"), inpa.get("device"))
        self.ctx = self._backend.context
        self.queue = self._backend.queue
        self.mf = cl.mem_flags
        # The kernels of the CUDA backend (Phase 3 of the fork): the native CUDA kernels of
        # PyFWI/kernels/elastic.cu in the fixed mode, unless PYFWI_CUDA_KERNELS=parity; the OpenCL
        # files otherwise. None on OpenCL.
        self.cuda_kernels = select_cuda_kernels(self.numerics) if self.backend_selection.backend == "cuda" else None

        # The time loops, the device buffers (with margins), the kernels and the checkpoints live in
        # the engine (step 2.1 of the fork, PyFWI.engine.propagation); its attributes (W.vx_b,
        # W.knl, W.prg...) are shown as attributes of the propagator (__getattr__ below).
        kernel, kernel_crosswell, kernel_surface = self.kernel_caller()
        sources = {"elastic": kernel, "crosswell": kernel_crosswell, "surface": kernel_surface}
        if self.cuda_kernels == "native":
            sources["native"] = self.native_kernel_source()
        injection = {0: "crosswell", 1: "surface"}.get(self.acq_type, "elastic")
        self._engine = PropagationEngine(
            self._backend, sources,
            injection, tnz=self.tnz, tnx=self.tnx, ns=self.ns, nr=self.nr, nt=self.nt, nt_ext=self.nt_ext,
            dt_scale=self.dt_scale, chp=self.chp, dxr=self.dxr, srcx=self.srcx, srcz=self.srcz, source=self.src,
            rec_top_right_var=self.rec_top_right_var, numerics=self.numerics)

        # Make a list for seismograms (filled by the engine)
        self.seismogram = self._engine.seismogram

    def __getattr__(self, name):
        """The attributes of the engine as attributes of the propagator, as in 0.1.10 (step 2.1).

        Python calls this only for names the propagator itself does not have: the device buffers
        (``W.vx_b``...), the programs and kernels (``W.prg``, ``W.knl``), the margins
        (``W.margin_bytes``, ``W.margins_changed()``) and the other attributes of
        ``PyFWI.engine.propagation.PropagationEngine``.
        """
        engine = self.__dict__.get("_engine")
        if engine is None:
            raise AttributeError(name)
        return getattr(engine, name)

    @property
    def W(self):
        """Checkpoints of the forward field, ``{component: array (tnz, tnx, ns, nchp)}``, as in 0.1.10.

        ``W['vx'][:, :, s, c]`` is the field ``vx`` of shot ``s`` at checkpoint ``c`` (time step
        ``chp[c]``), absorbing boundary included. Since step 1.5 the arrays are views of the
        checkpoint storage (``W._engine.checkpoints``, layout ``(ns, nchp, tnz, tnx)``): same shape, dtype,
        axis order and values as in 0.1.10, and writing into them changes the checkpoints, as
        before. The dictionary itself is new at each access, so replacing one of its entries has
        no effect; assign ``W`` instead (setter below).
        """
        return {par: store.transpose(2, 3, 0, 1) for par, store in self._engine.checkpoints.items()}

    @W.setter
    def W(self, value):
        """Replace the checkpoints with ``{component: array (tnz, tnx, ns, nchp)}``, the 0.1.10 layout.

        Used by ``PyFWI.optimization`` (``FO_waves.W = copy.deepcopy(W)``). The arrays are copied
        into the storage layout, as float32.
        """
        self._engine.checkpoints = {par: np.ascontiguousarray(np.asarray(array, dtype=np.float32).transpose(2, 3, 0, 1))
                                    for par, array in value.items()}

    def adjoint_buffer_preparing(self):
        """Create the gradient and adjoint-field buffers for a new gradient (``PropagationEngine.prepare_gradient``)."""
        self._engine.prepare_gradient()

    def gradient_reading(self):
        # The gradient and (with energy balancing) the preconditioners, read from the device.
        g_mu, g_lam, g_rho, g_mu_precond, g_lam_precond, g_rho_precond = \
            self._engine.read_gradient(preconditioners=self.energy_balancing)

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
        """Read the seismograms of shot ``s`` into ``self.seismogram`` (``PropagationEngine.make_seismogram``)."""
        self._engine.make_seismogram(s)

    def make_residual(self, res, s, t):
        """Copy sample ``t - 1`` of the residual of shot ``s`` from the host, as 0.1.10 did (kept, unused)."""
        self._engine.make_residual(res, s, t)

    def residual_to_device(self, res, s):
        """Copy the residual of shot ``s`` to the device (``PropagationEngine.residual_to_device``)."""
        self._engine.residual_to_device(res, s)

    def residual_sample(self, t):
        """Copy residual sample ``t - 1`` on the device (``PropagationEngine.residual_sample``)."""
        self._engine.residual_sample(t)

    def pml_preparation(self, v_max):

        vdx_pml = self.dx_pml * v_max
        vdz_pml = self.dz_pml * v_max
        self._engine.set_pml(vdx_pml, vdz_pml)

    def native_kernel_source(self):
        """The native CUDA kernels of Phase 3 (``PyFWI/kernels/elastic.cu``) after the ``#define`` block.

        Used by ``__init__`` when the CUDA backend runs the fixed mode (``self.cuda_kernels``).
        """
        with open(os.path.join(os.path.dirname(__file__), "kernels", "elastic.cu")) as f:
            return WavePreparation.kernel_defines(self) + f.read()

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
        # The #define block of the kernels (kernel_defines, the same for every source of the fork).
        macro = WavePreparation.kernel_defines(self)

        # Decide on openCl file based on medium type
        cl_file_name = "elastic.cl"

        # Call the openCl file (closed after reading: 0.1.10 left the three files open)
        path = os.path.dirname(__file__) + "/"
        with open(path + cl_file_name, 'r') as f:
            fstr = f.read()
        with open(path + 'elastic_crosswell.cl', 'r') as f_cw:
            fstr_cw = f_cw.read()
        with open(path + 'elastic_surface.cl', 'r') as f_surf:
            fstr_surf = f_surf.read()

        kernel_source = macro + fstr
        kernel_src_crosswell = macro + fstr_cw
        kernel_surface = macro + fstr_surf

        return kernel_source , kernel_src_crosswell, kernel_surface

    def kernel_defines(self):
        """The ``#define`` block of the kernels: grid, time step, spacing, coefficients, acquisition.

        It was the first part of ``kernel_caller`` in 0.1.10 (the text of the legacy mode is exactly
        that of 0.1.10); the native CUDA kernels of Phase 3 are compiled after the same block.

        Used by ``kernel_caller`` and ``native_kernel_source``.
        """
        # The numeric mode of the fork (step 2.4): "fixed" corrects the numerical bugs of 0.1.10.
        fixed = getattr(self, "numerics", "legacy") == "fixed"
        if self.sdo == 2:  # For O4
            c1 = 9 / 8
            c2 = -1 / 24
            c3 = 0
            c4 = 0
        elif self.sdo == 4 and fixed:  # For O8, fixed mode
            # The Taylor coefficients of the order-8 staggered first derivative. 0.1.10 (below) uses
            # them multiplied by 1434 and rounded to integers: then c1 + 3 c2 + 5 c3 + 7 c4, which
            # must be 1 for the derivative of a linear function to be exact, is 1436 / 1434, and
            # every wave of order 8 travels 0.14 % too fast.
            c1 = 1225 / 1024
            c2 = -245 / 3072
            c3 = 49 / 5120
            c4 = -5 / 7168
        elif self.sdo == 4:  # For O8
            c1 = 1715 / 1434  #1.2257  #
            c2 = -114 / 1434  # - 0.099537  #
            c3 = 14 / 1434  # 0.018063  #
            c4 = -1 / 1434  # - 0.0026274  #

        # The fixed numeric mode writes the floating-point constants as float literals (suffix f)
        # with 9 significant digits, so that the kernels compute in FP32 only and dt is not
        # truncated: 0.1.10 wrote them with "%f", 6 decimals and no suffix, i.e. as doubles (FP64
        # arithmetic, 1/64 of the FP32 rate on GeForce) and dt = 0.000571 instead of 0.000571428...
        # The legacy text below is exactly that of 0.1.10.
        real = "%.9ef" if fixed else "%f"
        macro = ("""
            #define Nz	   %d
            #define Nx     %d
            #define Ns     %d
            #define Nr     %d
            #define dt     REAL
            #define dx     REAL
            #define dz     REAL
            #define npml   %d
            #define n_main_rec %d
            #define n_extera_rec %d

            #define sdo     %d
            #define c1     REAL
            #define c2     REAL
            #define c3     REAL
            #define c4     REAL

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
        """).replace("REAL", real) % (self.tnz, self.tnx, self.ns, self.nr, self.dt, self.dh, self.dh, self.npml,
               self.n_surface_rec, self.n_well_rec, self.sdo,
               c1, c2, c3, c4, self.src_cts,
               self.rec_top_left_const, self.rec_top_left_var,
               self.rec_top_right_const, self.rec_top_right_var,
               self.rec_surface_const, self.rec_surface_var)
        if fixed:
            macro += "\n            #define PYFWI_FIXED 1\n"
        return macro

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

        self._engine.set_model(1 / model['rho'], self.mu, self.lam)


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
        observe = None
        if self.forward_show:
            self.initial_wavefield_plot(model)
            showpurose = np.zeros((self.tnz, self.tnx), dtype=np.float32)

            def observe(s, t):  # called by the engine after every time step
                if np.remainder(t, 20) == 0:
                    self.plot_propagation(self._engine.read_field("vx_b", showpurose), t)

        # The time loop of every shot (PyFWI.engine.propagation.PropagationEngine.forward).
        self._engine.forward(observe)

        if self.acq_type == 2:
            for par in self.seismogram:
                self.seismogram[par][:, :self.n_well_rec] = np.flip(self.seismogram[par][:, :self.n_well_rec], axis=1)
        return self.seismogram
    
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
        _info.log_first_run(self._backend.description)
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

        observe = None
        if show:
            vx_show = np.zeros((self.tnz, self.tnx), dtype=np.float32)
            adj_vx_show = np.zeros((self.tnz, self.tnx), dtype=np.float32)

            def observe(s, t, t_src):  # called by the engine after every time step
                if np.remainder(t, 20) == 0 or t == self.nt_ext - 2:
                    self._engine.read_field("taux_b", vx_show)
                    self._engine.read_field("ataux_b", adj_vx_show)
                    self.plot_propagation(vx_show, t_src, adj_vx_show)

        # The backward time loop of every shot (PyFWI.engine.propagation.PropagationEngine.gradient).
        self._engine.gradient(res, observe)

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
