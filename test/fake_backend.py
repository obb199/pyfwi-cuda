"""A backend of NumPy arrays and toy kernels, to test the engine without a device (roadmap step 2.1).

``FakeBackend`` implements the interface of ``PyFWI.backends`` with NumPy: a buffer is a float32
view into its allocation (so the margins are real), copies are array assignments, and every
operation is appended to ``log``. Its kernels are not the elastic stencils but *toy kernels* with
the same names and arguments, whose effect is simple enough to predict by hand:

=================  ==============================================================================
kernel             effect (``c`` = coefficient +1 or -1, ``cell`` = the source cell)
=================  ==============================================================================
MakeAllZero        zero the five buffers it gets
MakeGradZero       zero the six gradient and preconditioner buffers
injSrc             add the four source samples to vx, vz, taux, tauz at ``cell``; then, unless the
                   row is -1 (fixed mode), record for every component and receiver ``r``:
                   ``value at cell + r + 100 * srcx``
update_velx/velz   ``vx += c * rho[0]`` and ``vz += c * rho[0]`` (``rho`` holds 1 / density)
update_tauz        ``taux += c * lam[0]`` and ``tauz += c * lam[0]``
update_tauxz       ``tauxz += c * mu[0]``
Adj_injSrc         ``a<component>[0] += res_<component>[0]`` (the first receiver)
Adj_update_*       nothing
Grad               ``Gmu += vx``, ``Glam += avx``, ``Grho += 1``, ``g_mu_precond += 1``
=================  ==============================================================================

Used by ``test/test_engine.py``.
"""
import numpy as np


class FakeEvent:
    """What a fake copy returns: already complete."""

    def wait(self):
        pass


class FakeKernel:
    """A toy kernel: a name, its arguments, and the NumPy function that plays it."""

    def __init__(self, name, function):
        self.name, self.function, self.args = name, function, []


class FakeProgram:
    """A built program: gives a new toy kernel for each name, as ``pyopencl.Program`` does."""

    def __init__(self, backend, source):
        self.backend, self.source = backend, source

    def __getattr__(self, name):
        if name not in TOY_KERNELS:
            raise AttributeError(name)
        self.backend.log.append(("kernel", name))
        return FakeKernel(name, TOY_KERNELS[name])


class FakeBackend:
    """The interface of ``PyFWI.backends`` on NumPy arrays (see the module docstring).

    Parameters
    ----------
    nr : int
        Number of receivers, which the toy injSrc needs to record a row of seismogram.
    tnx : int
        Grid width, to find the source cell from its coordinates.
    """

    name = "fake"
    alignment = 16

    def __init__(self, nr, tnx):
        self.nr, self.tnx = nr, tnx
        self.log = []

    # memory
    def allocate(self, host, margin_bytes, fill):
        host = np.ascontiguousarray(host, dtype=np.float32)
        whole = np.full(2 * margin_bytes // 4 + host.size, fill, np.float32)
        start = margin_bytes // 4
        whole[start:start + host.size] = host.reshape(-1)
        self.log.append(("allocate", host.nbytes))
        return whole[start:start + host.size], whole

    def allocate_empty(self, nbytes):
        self.log.append(("allocate", nbytes))
        return np.empty(nbytes // 4, np.float32)

    @staticmethod
    def nbytes(buffer):
        return buffer.nbytes

    # copies
    def to_device(self, buffer, host, blocking=True):
        buffer[:] = np.asarray(host, np.float32).reshape(-1)
        self.log.append(("to_device", blocking))
        return FakeEvent()

    def to_host(self, host, buffer, blocking=True):
        host.reshape(-1)[:] = buffer
        self.log.append(("to_host", blocking))
        return FakeEvent()

    def copy(self, target, source, nbytes, offset=0):
        target[:nbytes // 4] = source[offset // 4:(offset + nbytes) // 4]
        self.log.append(("copy", offset))

    def wait(self, events):
        self.log.append(("wait", len(list(events))))

    def finish(self):
        self.log.append(("finish",))

    # kernels
    def build(self, source):
        self.log.append(("build",))
        return FakeProgram(self, source)

    def build_native(self, source):
        self.log.append(("build_native",))
        return FakeProgram(self, source)

    @staticmethod
    def kernel(program, name):
        return getattr(program, name)

    @staticmethod
    def set_args(kernel, *args):
        kernel.args = list(args)

    @staticmethod
    def set_arg(kernel, index, value):
        kernel.args[index] = value

    def launch(self, kernel, global_size):
        self.log.append(("launch", kernel.name))
        kernel.function(self, *kernel.args)

    def call(self, kernel, global_size, *args):
        self.set_args(kernel, *args)
        self.launch(kernel, global_size)


# ---------------------------------------------------------------------------- toy kernels

def _write(buffer, start, values):
    """Write ``values`` from ``buffer[start]`` on, through the allocation around the buffer.

    As on a device, a write past the end of the buffer lands in its margin (a fake buffer is a view
    of its allocation, see ``FakeBackend.allocate``), where ``margins_changed`` sees it.
    """
    whole = buffer.base if buffer.base is not None else buffer
    offset = (buffer.__array_interface__["data"][0] - whole.__array_interface__["data"][0]) // buffer.itemsize
    whole[offset + start:offset + start + len(values)] = values


def _zero(backend, *buffers):
    for buffer in buffers:
        buffer[:] = 0


def _inject(backend, vx, vz, taux, tauz, tauxz, s_vx, s_vz, s_taux, s_tauz, s_tauxz, dxr, srcx, srcz,
            sample_vx, sample_vz, sample_taux, sample_tauz, t):
    cell = int(srcz) * backend.tnx + int(srcx)
    for field, sample in ((vx, sample_vx), (vz, sample_vz), (taux, sample_taux), (tauz, sample_tauz)):
        field[cell] += sample
    if int(t) < 0:  # the guard of the fixed mode: inject without recording
        return
    nr = backend.nr
    for field, record in ((vx, s_vx), (vz, s_vz), (taux, s_taux), (tauz, s_tauz), (tauxz, s_tauxz)):
        _write(record, int(t) * nr, field[cell] + np.arange(nr) + 100 * int(srcx))


def _update_vel(backend, coeff, own, stress1, stress2, rho, vdx, vdz):
    own += coeff * rho[0]


def _update_tauz(backend, coeff, vx, vz, taux, tauz, lam, mu, vdx, vdz):
    taux += coeff * lam[0]
    tauz += coeff * lam[0]


def _update_tauxz(backend, coeff, vx, vz, tauxz, mu, vdx, vdz):
    tauxz += coeff * mu[0]


def _adjoint_inject(backend, avx, avz, ataux, atauz, atauxz, res_vx, res_vz, res_taux, res_tauz, res_tauxz, dxr):
    for field, res in ((avx, res_vx), (avz, res_vz), (ataux, res_taux), (atauz, res_tauz), (atauxz, res_tauxz)):
        field[0] += res[0]


def _nothing(backend, *args):
    pass


def _grad(backend, vx, vz, taux, tauz, tauxz, avx, avz, ataux, atauz, atauxz, lam, mu, rho,
          g_mu, mu_precond, g_lam, lam_precond, g_rho, rho_precond):
    g_mu += vx
    g_lam += avx
    g_rho += 1
    mu_precond += 1


#: The toy kernels by name (see the module docstring).
TOY_KERNELS = {"MakeAllZero": _zero, "MakeGradZero": _zero, "injSrc": _inject,
               "update_velx": _update_vel, "update_velz": _update_vel,
               "update_tauz": _update_tauz, "update_tauxz": _update_tauxz, "Adj_injSrc": _adjoint_inject,
               "Adj_update_tau": _nothing, "Adj_update_v": _nothing, "Grad": _grad}
