"""The shots of a propagator shared between several processes, one per GPU (roadmap step 4.4).

``PYFWI_CUDA_DEVICES`` lists CUDA devices, one worker process per entry: ``0,1`` runs on two GPUs,
``0,0`` runs two processes on one GPU (the test of a machine with a single GPU). ``DistributedEngine``
has the interface of ``PyFWI.engine.propagation.PropagationEngine`` that ``WavePropagator`` uses, and
gives every worker a contiguous block of the shots:

* every worker builds its own CUDA backend and engine for its shots, with the arguments of the engine
  of a single process (``spawn`` processes: CUDA cannot be forked);
* the model and the absorbing layer go to every worker; the seismograms come back shot by shot;
* the gradient of every shot comes back on its own (``PropagationEngine.gradient(per_shot=True)``) and
  the gradients are added on the host in the order of the shots, whatever worker computed them. A
  shot's gradient does not depend on the worker, nor on the other shots of its batch (in the FP16 mode
  too: its scales are powers of two), so one process and several give the same bits.

An error in a worker is raised in the main process with the worker's traceback; a worker that stops
(killed, out of memory) is reported by name. ``show=True``, the checkpoints (``W.W``) and the device
buffers (``W.vx_b``...) need a single process.

Used by
-------
``PyFWI.wave_propagation.WavePreparation`` when ``PYFWI_CUDA_DEVICES`` is set. Tested by
``test/test_distributed.py``.
"""
import multiprocessing
import os
import traceback

import numpy as np

from PyFWI.engine.propagation import COMPONENTS, PropagationEngine

#: The environment variable of the devices, one process per entry.
ENVIRONMENT_VARIABLE = "PYFWI_CUDA_DEVICES"


def requested_devices(environ=None):
    """The CUDA devices of ``PYFWI_CUDA_DEVICES`` (``"0,1"`` -> ``[0, 1]``), or None when unset or empty.

    Raises
    ------
    ValueError
        A value that is not a comma-separated list of device numbers.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get(ENVIRONMENT_VARIABLE, "")).strip()
    if not value:
        return None
    entries = [entry.strip() for entry in value.split(",")]
    if not all(entry.isdigit() for entry in entries):
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must list CUDA device numbers separated by commas "
                         f"(for example 0,1), not {value!r}")
    return [int(entry) for entry in entries]


def shot_blocks(ns, workers):
    """Contiguous blocks of the ``ns`` shots, as equal as possible, one per worker: ``[range, ...]``.

    The first ``ns % workers`` blocks take one shot more; a worker without shots gets an empty range.
    """
    size, extra = divmod(int(ns), int(workers))
    blocks, start = [], 0
    for w in range(int(workers)):
        count = size + (1 if w < extra else 0)
        blocks.append(range(start, start + count))
        start += count
    return blocks


def cuda_backend(ordinal):
    """The CUDA backend of device ``ordinal``: the backend of the workers (a picklable function)."""
    from PyFWI.backends.cuda import CUDABackend
    return CUDABackend(ordinal)


def _serve(connection, backend, ordinal, kwargs):
    """The loop of a worker: build the engine of its shots on device ``ordinal`` (``backend(ordinal)``), then
    run commands.

    Every command is ``(name, arguments)``; the answer is ``("ok", result)`` or ``("error", traceback)``.
    ``"stop"`` ends the loop.
    """
    try:
        engine = PropagationEngine(backend(ordinal), **kwargs)
        connection.send(("ok", None))
    except BaseException:  # noqa: BLE001 (reported to the main process)
        connection.send(("error", traceback.format_exc()))
        return
    commands = {
        "set_pml": engine.set_pml,
        "set_model": engine.set_model,
        "prepare_gradient": engine.prepare_gradient,
        "forward": lambda: {par: values.copy() for par, values in engine.forward().items()},
        "gradient": lambda res: engine.gradient(res, per_shot=True),
    }
    while True:
        name, arguments = connection.recv()
        if name == "stop":
            break
        try:
            connection.send(("ok", commands[name](*arguments)))
        except BaseException:  # noqa: BLE001 (reported to the main process)
            connection.send(("error", traceback.format_exc()))


class DistributedEngine:
    """The engine of a propagator whose shots are shared between processes (see the module docstring).

    Parameters
    ----------
    devices : list of int
        CUDA device of every worker process.
    kwargs : dict
        The arguments of ``PropagationEngine`` for all the shots, but the backend (``sources``,
        ``injection``, ``ns``, ``srcx``, ``srcz``...); every worker gets those of its block of shots.
    description : str
        What runs, for the log (``PyFWI.info``).
    backend : callable, optional
        ``backend(ordinal)`` builds the backend of a worker, in the worker: ``cuda_backend`` by default
        (the tests pass a fake one). It must be picklable.
    """

    def __init__(self, devices, kwargs, description="", backend=cuda_backend):
        self.devices, self.ns, self.nr = list(devices), int(kwargs["ns"]), int(kwargs["nr"])
        self.nt_ext, self.description = int(kwargs["nt_ext"]), description
        # With boundary saving the engines keep the last field only (PropagationEngine).
        self.chp = np.array([int(kwargs["nt"]) - 1], np.int32) if kwargs.get("boundary_width") else kwargs["chp"]
        self.nchp = len(self.chp)
        self.blocks = shot_blocks(self.ns, len(self.devices))
        #: The seismograms of the last forward modelling, ``{component: (nt_ext, nr, ns)}``.
        self.seismogram = {par: np.zeros((self.nt_ext, self.nr, self.ns), np.float32) for par in COMPONENTS}
        self._gradient = None
        context = multiprocessing.get_context("spawn")
        self._workers = []
        for w, (ordinal, block) in enumerate(zip(self.devices, self.blocks)):
            if not len(block):
                continue
            mine = dict(kwargs, ns=len(block), srcx=np.asarray(kwargs["srcx"])[block.start:block.stop],
                        srcz=np.asarray(kwargs["srcz"])[block.start:block.stop])
            parent, child = context.Pipe()
            process = context.Process(target=_serve, args=(child, backend, ordinal, mine), daemon=True,
                                      name=f"pyfwi-worker-{w}")
            process.start()
            child.close()
            self._workers.append((w, ordinal, block, process, parent))
        for worker in self._workers:  # "ok" once its engine is built
            self._receive(worker)

    # ------------------------------------------------------------------ messages

    def _receive(self, worker):
        """The answer of ``worker``; raises with its traceback, or when it stopped."""
        w, ordinal, _, process, connection = worker
        try:
            status, value = connection.recv()
        except (EOFError, OSError):
            raise RuntimeError(f"PyFWI worker {w} (CUDA device {ordinal}) stopped "
                               f"(exit code {process.exitcode}): see its output") from None
        if status == "error":
            raise RuntimeError(f"PyFWI worker {w} (CUDA device {ordinal}) failed:\n{value}")
        return value

    @staticmethod
    def _send(worker, name, arguments):
        """Send a command to ``worker``; raises when it stopped."""
        try:
            worker[4].send((name, arguments))
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"PyFWI worker {worker[0]} (CUDA device {worker[1]}) stopped "
                               f"(exit code {worker[3].exitcode}): see its output") from None

    def _broadcast(self, name, *arguments):
        """Send a command to every worker, then wait for every answer: the workers run together."""
        for worker in self._workers:
            self._send(worker, name, arguments)
        return [self._receive(worker) for worker in self._workers]

    # ------------------------------------------------------------------ the interface of PropagationEngine

    def set_pml(self, vdx_pml, vdz_pml):
        """The absorbing layer of every worker."""
        self._broadcast("set_pml", vdx_pml, vdz_pml)

    def set_model(self, rho_inverse, mu, lam):
        """The model of every worker."""
        self._broadcast("set_model", rho_inverse, mu, lam)

    def prepare_gradient(self):
        self._broadcast("prepare_gradient")

    def forward(self, observe=None):
        """Forward modelling of every shot, each worker its block; returns ``self.seismogram``."""
        if observe is not None:
            raise ValueError("show=True needs a single process: unset PYFWI_CUDA_DEVICES")
        for (_, _, block, _, _), seismograms in zip(self._workers, self._broadcast("forward")):
            for par in COMPONENTS:
                self.seismogram[par][:, :, block.start:block.stop] = seismograms[par]
        return self.seismogram

    def gradient(self, res, observe=None):
        """The gradient of every shot, each worker its block, added on the host in the order of the shots."""
        if observe is not None:
            raise ValueError("show=True needs a single process: unset PYFWI_CUDA_DEVICES")
        columns = {par: np.asarray(res[par]) for par in COMPONENTS}
        for worker in self._workers:
            block = worker[2]
            mine = {par: values[:, block.start * self.nr:block.stop * self.nr] for par, values in columns.items()}
            self._send(worker, "gradient", (mine,))
        total = None
        for worker in self._workers:  # the blocks follow each other in the order of the shots
            for shot in self._receive(worker):
                total = [np.array(g) for g in shot] if total is None else [t + g for t, g in zip(total, shot)]
        self._gradient = total

    def read_gradient(self, preconditioners=True):
        """``(g_mu, g_lam, g_rho, mu_precond, lam_precond, rho_precond)`` of the last ``gradient``."""
        g_mu, g_lam, g_rho, *precond = self._gradient
        if not preconditioners:
            return g_mu, g_lam, g_rho, *(np.ones_like(g_mu) for _ in range(3))
        return (g_mu, g_lam, g_rho, *precond)

    def read_field(self, name, host):
        raise ValueError(f"reading a field ({name}) needs a single process: unset PYFWI_CUDA_DEVICES")

    @property
    def checkpoints(self):
        raise ValueError("the checkpoints (W.W) need a single process: unset PYFWI_CUDA_DEVICES")

    def close(self):
        """Stop the workers (also when the engine is garbage-collected)."""
        for worker in getattr(self, "_workers", []):
            try:
                worker[4].send(("stop", ()))
            except (BrokenPipeError, OSError):
                pass
        for worker in getattr(self, "_workers", []):
            worker[3].join(timeout=10)
        self._workers = []

    def __del__(self):
        self.close()
