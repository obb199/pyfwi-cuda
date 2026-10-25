import os
from types import SimpleNamespace
import numpy as np
# from numpy.core.shape_base import block
import pyopencl as cl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import PyFWI.fwi_tools as tools
from PyFWI._opencl_device import choose_devices
from PyFWI.fwi_tools import expand_model
import PyFWI.acquisition as acq
from PyFWI.processing import prepare_residual
from PyFWI.grad_switcher import grad_lmd_to_vd


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

        self.W = {
            'vx': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'vz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'taux': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'tauz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'tauxz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
        }

        self.D = tools.Fdm(order=self.sdo * 2)

        self.components = components

        # To call openCl
        # The OpenCL device: inpa['platform'] and inpa['device'] (0 when missing), otherwise the
        # PYOPENCL_CTX environment variable, otherwise the first device of the first platform.
        # PyFWI._opencl_device.choose_devices replaces pyopencl.create_some_context(), which
        # failed on single-device platforms with PyOpenCL 2022 and could ask the user.
        platform = inpa.get("platform")
        device = inpa.get("device")
        devices = choose_devices(platform, device)
        if device is not None:
            # As in 0.1.10, the choice is also written to the environment, so that a later
            # WavePropagator created without inpa['device'] runs on the same device.
            os.environ['PYOPENCL_CTX'] = str(0 if platform is None else platform) + ':' + str(device)
            os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx)
        
        kernel, kernel_crosswell, kernel_surface = self.kernel_caller()

        self.prg_cw = cl.Program(self.ctx, kernel_crosswell).build()
        self.prg = cl.Program(self.ctx, kernel).build()
        self.prg_surf = cl.Program(self.ctx, kernel_surface).build()

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
        self.knl = SimpleNamespace(**{name: getattr(self.prg, name) for name in KERNELS})
        # Global size of every launch: one work item per grid point, absorbing boundary included.
        self.global_size = (int(self.tnz), int(self.tnx))

        self.mf = cl.mem_flags

        # Buffer for residuals
        res = np.zeros((np.int32(self.nr))).astype(np.float32, order='C')

        self.res_vx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                  self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_vz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                  self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_taux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_tauz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_tauxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                     self.mf.COPY_HOST_PTR, hostbuf=res)


        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        # Buffer for forward modelling
        self.vx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.vz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.taux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.tauz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.tauxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                 self.mf.COPY_HOST_PTR, hostbuf=v)

        # Buffer for seismograms
        seismogram_id = np.zeros((self.nt_ext, self.nr)).astype(np.float32, order='C')

        self.seismogramid_vx_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_vz_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_taux_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_tauz_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_tauxz_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)

        # Make a list for seismograms
        self.seismogram = {
            'vx': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'vz': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'taux': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'tauz': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
            'tauxz': np.zeros((self.nt_ext, self.nr, self.ns)).astype(np.float32, order='C'),
        }

    def adjoint_buffer_preparing(self):
        # Buffer for gradient
        g_mu = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.Gmu_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu)

        g_lam = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.Glam_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_lam)

        g_rho = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.Grho_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_rho)

        g_mu_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.g_mu_precond_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu_precond)

        self.g_lam_precond_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu_precond)

        self.g_rho_precond_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu_precond)

        # Buffer for backward modelling
        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.avx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.avz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.ataux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.atauz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.atauxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                 self.mf.COPY_HOST_PTR, hostbuf=v)

    def gradient_reading(self):
        g_mu = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        g_lam = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        g_rho = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        cl.enqueue_copy(self.queue, g_mu, self.Gmu_b)
        cl.enqueue_copy(self.queue, g_lam, self.Glam_b)
        cl.enqueue_copy(self.queue, g_rho, self.Grho_b)

        if self.energy_balancing:
            g_mu_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_lam_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_rho_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

            cl.enqueue_copy(self.queue, g_mu_precond, self.g_mu_precond_b)
            cl.enqueue_copy(self.queue, g_lam_precond, self.g_lam_precond_b)
            cl.enqueue_copy(self.queue, g_rho_precond, self.g_rho_precond_b)
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
        def get_from_opencl(buffer):
            seismogram_id = np.zeros((self.nt_ext, self.nr)).astype(np.float32, order='C')
            cl.enqueue_copy(self.queue, seismogram_id, buffer)
            return np.copy(seismogram_id)

        self.seismogram['vx'][:, :, s] = \
            get_from_opencl(self.seismogramid_vx_b)

        self.seismogram['vz'][:, :, s] = \
            get_from_opencl(self.seismogramid_vz_b)

        self.seismogram['taux'][:, :, s] = \
            get_from_opencl(self.seismogramid_taux_b)

        self.seismogram['tauz'][:, :, s] = \
            get_from_opencl(self.seismogramid_tauz_b)

        self.seismogram['tauxz'][:, :, s] = \
            get_from_opencl(self.seismogramid_tauxz_b)

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
        cl.enqueue_copy(self.queue, self.res_vx_b, res_src_vx)

        res_src_vz = (res['vz'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_vz_b, res_src_vz)

        res_src_taux = (res['taux'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_taux_b, res_src_taux)

        res_src_tauz = (res['tauz'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_tauz_b, res_src_tauz)

        res_src_tauxz = (res['tauxz'][np.int32(t - 1), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_tauxz_b, res_src_tauxz)

    def pml_preparation(self, v_max):

        vdx_pml = self.dx_pml * v_max
        vdz_pml = self.dz_pml * v_max
        self.vdx_pml_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=vdx_pml)
        self.vdz_pml_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=vdz_pml)

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

        self.rho_b = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR,
                               hostbuf=1 / model['rho'])

        self.mu_b = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR,
                              hostbuf=self.mu)
        self.lam_b = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR,
                               hostbuf=self.lam)


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
            self.knl.MakeAllZero(self.queue, self.global_size, None,
                                 self.vx_b, self.vz_b,
                                 self.taux_b, self.tauz_b, self.tauxz_b
                                 )
            self._set_injection_args(s)

            self.__kernel(s, coeff)

        if self.acq_type == 2:
            for par in self.seismogram:
                self.seismogram[par][:, :self.n_well_rec] = np.flip(self.seismogram[par][:, :self.n_well_rec], axis=1)
        return self.seismogram
    
    def _launch(self, kernel):
        """Launch ``kernel`` on the whole grid with the arguments set before (``set_args``/``set_arg``).

        ``pyopencl.enqueue_nd_range_kernel`` only enqueues, whereas calling the kernel object
        (``kernel(queue, size, None, *args)``) also sets every argument again. Used for the
        launches of the time loops (step 1.3 of the fork).
        """
        cl.enqueue_nd_range_kernel(self.queue, kernel, self.global_size, None)

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
        self.knl.update_velx.set_args(coeff, self.vx_b, self.taux_b, self.tauxz_b, self.rho_b,
                                      self.vdx_pml_b, self.vdz_pml_b)
        self.knl.update_velz.set_args(coeff, self.vz_b, self.tauz_b, self.tauxz_b, self.rho_b,
                                      self.vdx_pml_b, self.vdz_pml_b)
        self.knl.update_tauz.set_args(coeff, self.vx_b, self.vz_b, self.taux_b, self.tauz_b,
                                      self.lam_b, self.mu_b, self.vdx_pml_b, self.vdz_pml_b)
        self.knl.update_tauxz.set_args(coeff, self.vx_b, self.vz_b, self.tauxz_b, self.mu_b,
                                       self.vdx_pml_b, self.vdz_pml_b)

    def _set_injection_args(self, s):
        """Set the arguments of injSrc that stay the same for every time step of source ``s``.

        The last five arguments (the four source samples and the time index) change at each
        injection and are set by ``__injSrc``; zeros are placeholders until then.
        """
        zero = np.float32(0)
        self.knl.injSrc.set_args(self.vx_b, self.vz_b,
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
            self.knl.injSrc.set_arg(index, value)
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
                copy_purpose = np.zeros((self.tnz, self.tnx), dtype=np.float32)
                cl.enqueue_copy(self.queue, copy_purpose, self.vx_b)
                self.W['vx'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.vz_b)
                self.W['vz'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.taux_b)
                self.W['taux'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.tauz_b)
                self.W['tauz'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.tauxz_b)
                self.W['tauxz'][:, :, s, chpc] = np.copy(copy_purpose)

                chpc += 1


            if self.forward_show and np.remainder(t, 20) == 0:
                cl.enqueue_copy(self.queue, showpurose, self.vx_b)
                self.plot_propagation(showpurose, t)
        self.make_seismogram(s, t)
        
    def __adjoint_modelling_per_source(self, res):
        self.knl.MakeGradZero(self.queue, self.global_size, None,
                              self.Gmu_b, self.Glam_b, self.Grho_b,
                              self.g_mu_precond_b, self.g_lam_precond_b, self.g_rho_precond_b)

        # The arguments of the time loop do not change during the gradient: set them once
        # (step 1.3). The update kernels rebuild the forward field backwards (coeff = -1).
        self._set_update_args(np.int32(-1))
        self._set_adjoint_args()

        for s in range(self.ns):
            self.knl.MakeAllZero(self.queue, self.global_size, None,
                                 self.vx_b, self.vz_b,
                                 self.taux_b, self.tauz_b, self.tauxz_b)

            self.knl.MakeAllZero(self.queue, self.global_size, None,
                                 self.avx_b, self.avz_b,
                                 self.ataux_b, self.atauz_b, self.atauxz_b)

            self.__kernel_gradient(res, s)

    def _set_adjoint_args(self):
        """Set the arguments of the adjoint kernels (Adj_injSrc, Adj_update_tau, Adj_update_v, Grad).

        Called at the start of each gradient, after ``adjoint_buffer_preparing`` has created the
        adjoint and gradient buffers. The residual buffers keep their identity; only their content
        changes (``make_residual``).
        """
        self.knl.Adj_injSrc.set_args(self.avx_b, self.avz_b,
                                     self.ataux_b, self.atauz_b, self.atauxz_b,
                                     self.res_vx_b, self.res_vz_b,
                                     self.res_taux_b, self.res_tauz_b, self.res_tauxz_b,
                                     self.dxr)
        self.knl.Adj_update_tau.set_args(self.avx_b, self.avz_b,
                                         self.ataux_b, self.atauz_b, self.atauxz_b,
                                         self.rho_b,
                                         self.vdx_pml_b, self.vdz_pml_b)
        self.knl.Adj_update_v.set_args(self.avx_b, self.avz_b,
                                       self.ataux_b, self.atauz_b, self.atauxz_b,
                                       self.lam_b, self.mu_b,
                                       self.vdx_pml_b, self.vdz_pml_b)
        self.knl.Grad.set_args(self.vx_b, self.vz_b,
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
                vx = np.copy(self.W['vx'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.vx_b, vx)

                vz = np.copy(self.W['vz'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.vz_b, vz)

                taux = np.copy(self.W['taux'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.taux_b, taux)

                tauz = np.copy(self.W['tauz'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.tauz_b, tauz)

                tauxz = np.copy(self.W['tauxz'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.tauxz_b, tauxz)

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
                self.make_residual(res, s, t_src)

                self._launch(self.knl.Adj_injSrc)
                t_src -= 1

            self._launch(self.knl.Adj_update_tau)
            self._launch(self.knl.Adj_update_v)
            self._launch(self.knl.Grad)

            # Plotting wave propagation
            if self.backward_show and (np.remainder(t, 20) == 0 or t == self.nt_ext - 2):
                cl.enqueue_copy(self.queue, vx_show, self.taux_b)

                cl.enqueue_copy(self.queue, adj_vx_show, self.ataux_b)

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
