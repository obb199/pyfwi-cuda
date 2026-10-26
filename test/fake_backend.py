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
inject_and_record  (native, steps 3.4 and 3.6) add the source of the step on the device (``step``) at
                   the source cell; at a recorded step, record every component at the cell of each
                   receiver (``cells``)
inject_residual    (native) at a recorded step, add the residual of each receiver at its cell,
                   receivers sharing a cell added by the first one (``first``, ``following``)
advance            (native) ``step += delta``
update_v           (native, fused, step 3.5) update_velx and update_velz
update_tau         (native, fused) update_tauz and update_tauxz
grad_adjoint_v     (native, fused) Grad, then Adj_update_v (nothing)
sum_slots          (native, step 3.9) add the gradients of slots 1 to ``slots - 1`` to slot 0
copy_checkpoint    (native, step 3.9) copy the field of every slot into its checkpoint, or back
save_band          (native, step 4.2) store the 5 fields at the cells of the band, at the step on the device
restore_band       (native) write back the stored stresses or velocities of the band, step + delta
remove_source      (native) subtract the source of step + delta at the source cell
=================  ==============================================================================

A launch over ``(rows, columns, slots)`` runs a batch of shots (step 3.9): the kernels act on the
field of every slot, slot b starting ``b * (tnz + 1) * tnx`` values into the buffers of the fields
and gradients (``slot_values`` of elastic.cu) and ``b * rows * nr`` into the seismograms and
residuals; the source cell of slot b is ``source_cell[b]``.

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


class FakeGraph:
    """What ``FakeBackend.capture`` returns: the recorded launches, run again by ``launch``."""

    def __init__(self, backend, launches):
        self.backend, self.launches = backend, launches

    def launch(self):
        self.backend.log.append(("graph", len(self.launches)))
        for name, function, args, slots in self.launches:
            self.backend.log.append(("launch", name))
            self.backend.slots = slots
            function(self.backend, *args)


class FakeBackend:
    """The interface of ``PyFWI.backends`` on NumPy arrays (see the module docstring).

    Parameters
    ----------
    nr : int
        Number of receivers, which the toy injSrc needs to record a row of seismogram.
    tnz, tnx : int
        Grid size: ``tnx`` finds the source cell from its coordinates, both the field of each slot of a
        batch (step 3.9).
    budget : int, optional
        Bytes of checkpoints the engine may keep on the "device" (``checkpoint_budget``, step 3.8).
    """

    name = "fake"
    alignment = 16

    def __init__(self, nr, tnz, tnx, budget=0):
        self.nr, self.tnz, self.tnx = nr, tnz, tnx
        #: Slots of the launch running (its third global size, step 3.9).
        self.slots = 1
        self.log = []
        #: ``(kernel name, global size)`` of every launch outside a graph.
        self.sizes = []
        self.budget = budget
        #: While ``capture`` records: the launches, which then do not run.
        self._recording = None

    # memory
    def allocate(self, host, margin_bytes, fill):
        host = np.ascontiguousarray(host)
        host = host if host.dtype == np.float16 else host.astype(np.float32)  # float16: the FP16 fields
        whole = np.full(2 * margin_bytes // host.itemsize + host.size, fill, host.dtype)
        start = margin_bytes // host.itemsize
        whole[start:start + host.size] = host.reshape(-1)
        self.log.append(("allocate", host.nbytes))
        return whole[start:start + host.size], whole

    def allocate_empty(self, nbytes, dtype=np.float32):
        self.log.append(("allocate", nbytes))
        return np.empty(nbytes // np.dtype(dtype).itemsize, dtype)

    @staticmethod
    def nbytes(buffer):
        return buffer.nbytes

    # copies
    def to_device(self, buffer, host, blocking=True):
        host = np.asarray(host)
        if host.dtype.kind in "iu":  # an integer table: its bytes, as a device copy does
            buffer.view(host.dtype)[:host.size] = host.reshape(-1)
        else:
            buffer[:] = host.astype(buffer.dtype).reshape(-1)
        self.log.append(("to_device", blocking))
        return FakeEvent()

    def to_host(self, host, buffer, blocking=True):
        host.reshape(-1)[:] = buffer
        self.log.append(("to_host", blocking))
        return FakeEvent()

    def checkpoint_budget(self):
        return self.budget

    @staticmethod
    def section(buffer, offset, nbytes):
        return buffer[offset // buffer.itemsize:(offset + nbytes) // buffer.itemsize]

    def copy(self, target, source, nbytes, offset=0):
        target[:nbytes // target.itemsize] = source[offset // source.itemsize:(offset + nbytes) // source.itemsize]
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
        self.sizes.append((kernel.name, tuple(global_size)))
        slots = global_size[2] if len(global_size) > 2 else 1  # a batch of shots (step 3.9)
        if self._recording is not None:  # a graph is being recorded: nothing runs
            self._recording.append((kernel.name, kernel.function, list(kernel.args), slots))
            return
        self.log.append(("launch", kernel.name))
        self.slots = slots
        kernel.function(self, *kernel.args)

    def capture(self, record):
        """Record the launches of ``record()`` (with the arguments they have now) into a ``FakeGraph``."""
        self._recording = []
        try:
            record()
        finally:
            launches, self._recording = self._recording, None
        self.log.append(("capture", len(launches)))
        return FakeGraph(self, launches)

    def call(self, kernel, global_size, *args):
        self.set_args(kernel, *args)
        self.launch(kernel, global_size)


class FakeBackendFactory:
    """Builds a ``FakeBackend`` in a worker process of ``DistributedEngine`` (step 4.4): picklable."""

    def __init__(self, nr, tnz, tnx, budget=0, fail_on=None):
        self.nr, self.tnz, self.tnx, self.budget, self.fail_on = nr, tnz, tnx, budget, fail_on

    def __call__(self, ordinal):
        if ordinal == self.fail_on:
            raise RuntimeError(f"no device {ordinal}")
        return FakeBackend(self.nr, self.tnz, self.tnx, self.budget)


# ---------------------------------------------------------------------------- toy kernels

def _write(buffer, start, values):
    """Write ``values`` from ``buffer[start]`` on, through the allocation around the buffer.

    As on a device, a write past the end of the buffer lands in its margin (a fake buffer is a view
    of its allocation, see ``FakeBackend.allocate``), where ``margins_changed`` sees it.
    """
    whole = buffer.base if buffer.base is not None else buffer
    offset = (buffer.__array_interface__["data"][0] - whole.__array_interface__["data"][0]) // buffer.itemsize
    whole[offset + start:offset + start + len(values)] = values


def _slots(backend, buffer):
    """The field of every slot launched in ``buffer`` (step 3.9): slot b starts ``b * (tnz + 1) * tnx``
    values in; without batches, the buffer is the field of slot 0."""
    plane, stride = backend.tnz * backend.tnx, (backend.tnz + 1) * backend.tnx
    return [buffer[b * stride:b * stride + plane] for b in range(backend.slots)]


def _zero(backend, *buffers):
    for buffer in buffers:
        for field in _slots(backend, buffer):
            field[:] = 0


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


def _recorded_sample(t, dt_scale, rows):
    """The recorded sample of step ``t``, or -1 (``recorded_sample`` of elastic.cu)."""
    return t // dt_scale if t % dt_scale == 0 and t // dt_scale < rows else -1


def _inject_and_record(backend, vx, vz, taux, tauz, tauxz, s_vx, s_vz, s_taux, s_tauz, s_tauxz, cells,
                       source_cell, source_steps, step, dt_scale, rows):
    t = int(step.view(np.int32)[0])
    row = _recorded_sample(t, int(dt_scale), int(rows))
    table = cells.view(np.int32)
    for slot, fields in enumerate(zip(*(_slots(backend, f) for f in (vx, vz, taux, tauz, tauxz)))):
        cell = int(source_cell.view(np.int32)[slot])
        for c, field in enumerate(fields[:4]):
            field[cell] += source_steps[4 * t + c]
        if row < 0:
            continue
        for field, record in zip(fields, (s_vx, s_vz, s_taux, s_tauz, s_tauxz)):
            _write(record, (slot * int(rows) + row) * backend.nr, field[table])


def _inject_residual(backend, avx, avz, ataux, atauz, atauxz, res_vx, res_vz, res_taux, res_tauz, res_tauxz,
                     cells, first, following, step, dt_scale, rows):
    sample = _recorded_sample(int(step.view(np.int32)[0]), int(dt_scale), int(rows))
    if sample < 0:
        return
    table, owner, nxt = cells.view(np.int32), first.view(np.int32), following.view(np.int32)
    residuals = (res_vx, res_vz, res_taux, res_tauz, res_tauxz)
    for slot, fields in enumerate(zip(*(_slots(backend, f) for f in (avx, avz, ataux, atauz, atauxz)))):
        base = (slot * int(rows) + sample) * backend.nr
        for field, res in zip(fields, residuals):
            for r in range(backend.nr):
                if table[r] < 0 or not owner[r]:
                    continue
                total, q = res[base + r], nxt[r]
                while q >= 0:
                    total, q = total + res[base + q], nxt[q]
                field[table[r]] += total


def _advance(backend, step, delta):
    step.view(np.int32)[0] += int(delta)


def _copy_checkpoint(backend, vx, vz, taux, tauz, tauxz, checkpoints, offset, shot_pitch, component_pitch, restore):
    plane = backend.tnz * backend.tnx
    for slot, fields in enumerate(zip(*(_slots(backend, f) for f in (vx, vz, taux, tauz, tauxz)))):
        for c, field in enumerate(fields):
            start = int(c * component_pitch + offset + slot * shot_pitch)
            if restore:
                field[:] = checkpoints[start:start + plane]
            else:
                checkpoints[start:start + plane] = field


def _band_store(store, band_shot, slot, step, n_band, shot_values):
    """The 5 x n_band values stored for ``step`` of the shot of ``slot`` (boundary saving)."""
    start = int(band_shot.view(np.int32)[slot]) * int(shot_values) + step * 5 * int(n_band)
    return store[start:start + 5 * int(n_band)].reshape(5, int(n_band))


def _save_band(backend, vx, vz, taux, tauz, tauxz, store, band, band_shot, step, n_band, shot_values):
    t, cells = int(step.view(np.int32)[0]), band.view(np.int32)[:int(n_band)]
    for slot, fields in enumerate(zip(*(_slots(backend, f) for f in (vx, vz, taux, tauz, tauxz)))):
        stored = _band_store(store, band_shot, slot, t, n_band, shot_values)
        for c, field in enumerate(fields):
            stored[c] = field[cells]


def _restore_band(backend, vx, vz, taux, tauz, tauxz, store, band, band_shot, step, delta, stresses, n_band,
                  shot_values):
    t, cells = int(step.view(np.int32)[0]) + int(delta), band.view(np.int32)[:int(n_band)]
    components = (2, 3, 4) if int(stresses) else (0, 1)
    for slot, fields in enumerate(zip(*(_slots(backend, f) for f in (vx, vz, taux, tauz, tauxz)))):
        stored = _band_store(store, band_shot, slot, t, n_band, shot_values)
        for c in components:
            fields[c][cells] = stored[c]


def _remove_source(backend, vx, vz, taux, tauz, source_cell, source_steps, step, delta):
    t = int(step.view(np.int32)[0]) + int(delta)
    for slot, fields in enumerate(zip(*(_slots(backend, f) for f in (vx, vz, taux, tauz)))):
        cell = int(source_cell.view(np.int32)[slot])
        for c, field in enumerate(fields):
            field[cell] -= source_steps[4 * t + c]


def _sum_slots(backend, *buffers_and_slots):
    *buffers, slots = buffers_and_slots
    backend.slots = int(slots)
    for buffer in buffers:
        first, *others = _slots(backend, buffer)
        for field in others:
            first += field


def _update_v(backend, coeff, vx, vz, taux, tauz, tauxz, rho, vdx, vdz, *undamped):
    _update_vel(backend, coeff, vx, taux, tauxz, rho, vdx, vdz)
    _update_vel(backend, coeff, vz, tauz, tauxz, rho, vdx, vdz)


def _update_tau(backend, coeff, vx, vz, taux, tauz, tauxz, lam, mu, vdx, vdz, *undamped):
    _update_tauz(backend, coeff, vx, vz, taux, tauz, lam, mu, vdx, vdz)
    _update_tauxz(backend, coeff, vx, vz, tauxz, mu, vdx, vdz)


def _grad_adjoint_v(backend, vx, vz, taux, tauz, tauxz, avx, avz, ataux, atauz, atauxz, lam, mu, rho, vdx, vdz,
                    g_mu, mu_precond, g_lam, lam_precond, g_rho, rho_precond, *undamped):
    _grad(backend, vx, vz, taux, tauz, tauxz, avx, avz, ataux, atauz, atauxz, lam, mu, rho,
          g_mu, mu_precond, g_lam, lam_precond, g_rho, rho_precond)


def _update_vel(backend, coeff, own, stress1, stress2, rho, vdx, vdz):
    for field in _slots(backend, own):
        field += coeff * rho[0]


def _update_tauz(backend, coeff, vx, vz, taux, tauz, lam, mu, vdx, vdz):
    for field in _slots(backend, taux) + _slots(backend, tauz):
        field += coeff * lam[0]


def _update_tauxz(backend, coeff, vx, vz, tauxz, mu, vdx, vdz):
    for field in _slots(backend, tauxz):
        field += coeff * mu[0]


def _adjoint_inject(backend, avx, avz, ataux, atauz, atauxz, res_vx, res_vz, res_taux, res_tauz, res_tauxz, dxr):
    for field, res in ((avx, res_vx), (avz, res_vz), (ataux, res_taux), (atauz, res_tauz), (atauxz, res_tauxz)):
        field[0] += res[0]


def _nothing(backend, *args):
    pass


def _grad(backend, vx, vz, taux, tauz, tauxz, avx, avz, ataux, atauz, atauxz, lam, mu, rho,
          g_mu, mu_precond, g_lam, lam_precond, g_rho, rho_precond):
    for v, a, gm, gl, gr, mp in zip(*(_slots(backend, b) for b in (vx, avx, g_mu, g_lam, g_rho, mu_precond))):
        gm += v
        gl += a
        gr += 1
        mp += 1


#: The toy kernels by name (see the module docstring).
TOY_KERNELS = {"MakeAllZero": _zero, "MakeGradZero": _zero, "injSrc": _inject,
               "update_velx": _update_vel, "update_velz": _update_vel,
               "update_tauz": _update_tauz, "update_tauxz": _update_tauxz, "Adj_injSrc": _adjoint_inject,
               "Adj_update_tau": _nothing, "Adj_update_v": _nothing, "Grad": _grad,
               "inject_and_record": _inject_and_record, "inject_residual": _inject_residual,
               "update_v": _update_v, "update_tau": _update_tau, "grad_adjoint_v": _grad_adjoint_v,
               "advance": _advance, "sum_slots": _sum_slots, "copy_checkpoint": _copy_checkpoint,
               "save_band": _save_band, "restore_band": _restore_band, "remove_source": _remove_source}
