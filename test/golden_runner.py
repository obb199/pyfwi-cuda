"""Run one golden-master case and collect its outputs (roadmap step 0.4).

The same code runs in the reference environment (PyFWI 0.1.10 from PyPI, to store
the golden masters) and in the development environment (the fork, to compare with
them), so it only uses the public API of PyFWI plus two read-only hooks that work
in both: the ``seismogram`` attribute and the ``gradient_reading`` method of the
propagator.

Outputs are layered, so that a difference can be traced to where it starts:

=====================  ==================================================================
Key                    Content
=====================  ==================================================================
``raw_seismogram/<c>`` ``W.seismogram[c]`` after the forward run of the true model, for the
                       five components ``vx, vz, taux, tauz, tauxz``: what the kernels
                       recorded, shape ``(nt, nr, ns)``, before any reformatting.
``observed/<c>``       ``forward_modeling`` of the true model: the components selected by
                       ``components``, shape ``(nt, nr * ns)``.
``predicted/<c>``      ``forward_modeling`` of the starting model.
``misfit``             L2 misfit between predicted and observed data (a 0-d array).
``raw_gradient/<p>``   What ``gradient_reading`` returns (lam, mu, rho): the gradient as
                       read from the GPU, cut to the model and preconditioned, before the
                       change of parameters and the smoothing.
``gradient/<p>``       What ``W.gradient`` returns: vp, vs, rho (``"dv"``) or lam, mu, rho.
=====================  ==================================================================

Every run uses a race-free geometry and, by default, zero guard bands: the two rules that
make PyFWI 0.1.10 reproducible (see ``test_reproducibility.py``). Since step 1.7 the fork
pads its buffers itself, so the characterisation tests run it without guard bands
(``guard=False``) and still get the bits of the golden masters.

Used by
-------
``tools/make_golden.py`` and the characterisation tests.
"""
import contextlib

import numpy as np

from golden_cases import PropagationCase
from helpers import race_free_geometry
from oob_guard import zero_guard_bands
from tools.check_env import inclusion_model

#: The five field components PyFWI records, in its own order.
COMPONENTS = ("vx", "vz", "taux", "tauz", "tauxz")


def case_inputs(case: PropagationCase):
    """Build the PyFWI inputs of a case: input parameters, models, source, receivers.

    Parameters
    ----------
    case : PropagationCase
        The case to build.

    Returns
    -------
    dict
        ``inpa`` (input parameters), ``true`` and ``start`` (models, ``{"vp", "vs", "rho"}``
        of ``float32`` arrays), ``src`` (``PyFWI.acquisition.Source``), ``rec_loc``
        (receiver positions, ``(nr, 2)`` metres) and ``n_well_rec``.
    """
    import PyFWI.acquisition as acq

    inpa = {"ns": case.ns, "sdo": case.order, "fdom": 15, "dh": case.dh, "dt": 0.004, "acq_type": case.acq_type,
            "t": case.t, "npml": case.npml, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": case.dh,
            "energy_balancing": case.energy_balancing}
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(case.ns, case.dh, case.dh * case.nx, case.dh * case.nz,
                                                         case.dh, case.order, acq_type=case.acq_type)
    src_loc, rec_loc = race_free_geometry(src_loc, rec_loc, n_well_rec, case.dh, case.dh, case.nz)
    src = acq.Source(src_loc, case.dh, inpa["dt"])
    src.Ricker(inpa["fdom"])
    return {"inpa": inpa, "true": inclusion_model(case.nz, case.nx), "start": inclusion_model(case.nz, case.nx, smooth=1),
            "src": src, "rec_loc": rec_loc, "n_well_rec": n_well_rec}


def run_propagation_case(case: PropagationCase, guard=True):
    """Run the forward modelling and the gradient of one case and return the layered outputs.

    The OpenCL device is the one selected by ``PYOPENCL_CTX``.

    Parameters
    ----------
    case : PropagationCase
        The case to run.
    guard : bool, optional
        True (default): run inside zero guard bands, the reproducible definition used
        by the golden masters. False: a plain run, used for the fork since step 1.7 (its
        margins replace the guard bands) and by ``tools/make_golden.py`` to record whether
        0.1.10 itself gives the same bits (in a fresh process, because plain runs of 0.1.10
        can crash the NVIDIA driver after a few propagations).

    Returns
    -------
    dict[str, numpy.ndarray]
        The outputs described in the module docstring, as independent copies.
    """
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    import PyFWI.wave_propagation as wave

    x = case_inputs(case)
    out = {}
    with zero_guard_bands() if guard else contextlib.nullcontext():
        W = wave.WavePropagator(x["inpa"], x["src"], x["rec_loc"], (case.nz, case.nx), components=case.components,
                                n_well_rec=x["n_well_rec"], chpr=case.chpr)
        observed = W.forward_modeling(x["true"], show=False)
        out.update({f"raw_seismogram/{c}": np.array(W.seismogram[c]) for c in COMPONENTS})
        out.update({f"observed/{c}": np.array(v) for c, v in observed.items()})
        predicted = W.forward_modeling(x["start"], show=False)
        out.update({f"predicted/{c}": np.array(v) for c, v in predicted.items()})
        misfit, adj_src = tools.cost_seismic(process.prepare_residual(predicted, 1),
                                             process.prepare_residual(observed, 1), fun=tools.CostFunction("l2"))
        out["misfit"] = np.asarray(misfit)

        # Hook: keep what gradient_reading returns (raw gradient) without changing it.
        reading = W.gradient_reading
        raw = {}

        def capture():
            raw["lam"], raw["mu"], raw["rho"] = (np.array(g) for g in reading())
            return raw["lam"], raw["mu"], raw["rho"]
        W.gradient_reading = capture
        gradient = W.gradient(adj_src, show=False, parameterization=case.parameterization)
        out.update({f"raw_gradient/{p}": v for p, v in raw.items()})
        out.update({f"gradient/{p}": np.array(v) for p, v in gradient.items()})
    return out


#: Leaf names that hold run times; they change at every run and are never stored.
TIMING_KEYS = ("pt", "seconds")


def flatten(obj, prefix=""):
    """Turn nested results into ``{"a/b/c": numpy.ndarray}`` with only the numeric leaves.

    Dicts and lists/tuples are walked (list items get their index as key); arrays,
    NumPy scalars and Python numbers become arrays; strings, None and other objects are
    skipped, and so are run times (leaves named ``pt``, ``pt_<x>`` or ``seconds``).

    Parameters
    ----------
    obj : object
        What an example or an inversion returned.
    prefix : str, optional
        Key prefix of ``obj`` (used by the recursion).

    Returns
    -------
    dict[str, numpy.ndarray]
    """
    out = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    else:
        items = None
    if items is not None:
        for key, value in items:
            name = str(key)
            if name in TIMING_KEYS or name.startswith("pt_"):
                continue
            out.update(flatten(value, f"{prefix}{name}/"))
        return out
    if isinstance(obj, (np.ndarray, np.generic, bool, int, float)) and np.asarray(obj).dtype.kind in "biuf":
        out[prefix.rstrip("/")] = np.array(obj)
    return out


def run_inversion_case(case, guard=True):
    """Run a short FWI or time-lapse FWI (``InversionCase``) and return its flattened outputs.

    Returns
    -------
    dict[str, numpy.ndarray]
        For ``"fwi"``: ``model/<p>`` (the estimated model) and ``misfit/...`` (as returned
        by ``FWI``). For ``"tlfwi"``: every array the strategy returns (``b1/vp``, ``dm/vp``,
        ...) and ``misfit/...``.
    """
    import io
    from contextlib import redirect_stdout

    import PyFWI.processing as process
    from helpers import tiny_case

    guard_bands = zero_guard_bands() if guard else contextlib.nullcontext()  # see run_propagation_case
    with guard_bands, redirect_stdout(io.StringIO()):  # PyFWI prints every misfit evaluation
        c = tiny_case(ns=case.ns, nz=case.nz, nx=case.nx, inpa_extra={"energy_balancing": True})
        c.src.w = c.src.w * np.float32(1000)  # misfit of order 1 (see InversionCase)
        observed = process.prepare_residual(c.W.forward_modeling(c.model, show=False), 1)
        if case.kind == "fwi":
            from PyFWI.fwi import FWI
            inversion = FWI(observed, c.inpa, c.src, c.rec_loc, (case.nz, case.nx), components=0, chpr=20,
                            n_well_rec=c.n_well_rec)
            model, misfit = inversion(c.m0, method=case.method, freqs=[25], iter=[case.iterations], n_params=1,
                                      k_0=1, k_end=2)
            return {**flatten(model, "model/"), **flatten(misfit, "misfit/")}
        from PyFWI.tl_fwi import TimeLapse
        monitor = {k: v.copy() for k, v in c.model.items()}
        monitor["vp"][8:12, 15:25] *= np.float32(0.9)
        observed_monitor = process.prepare_residual(c.W.forward_modeling(monitor, show=False), 1)
        tl = TimeLapse(observed, observed_monitor, c.inpa, c.src, c.rec_loc, (case.nz, case.nx), components=0,
                       chpr=20, n_well_rec=c.n_well_rec)
        models, misfit = tl(c.m0, tl_method=case.method, freqs=[25], iter=[case.iterations], n_params=1,
                            k_0=1, k_end=2)
        return {**flatten(models), **flatten(misfit, "misfit/")}


def run_example_case(case, guard=True):
    """Run ``main(size="small")`` of an example script (``ExampleCase``) and return its flattened outputs.

    ``guard`` as in ``run_propagation_case``.
    """
    import importlib.util
    import io
    import os
    from contextlib import redirect_stdout

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example", "scripts", f"{case.script}.py")
    spec = importlib.util.spec_from_file_location(f"golden_example_{case.script}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with zero_guard_bands() if guard else contextlib.nullcontext(), redirect_stdout(io.StringIO()):
        return flatten(module.main(size="small", show=False))


def run_case(case, guard=True):
    """Run any golden-master case and return its outputs (dispatch on the case type).

    ``guard`` as in ``run_propagation_case``.
    """
    from golden_cases import ExampleCase, InversionCase
    if isinstance(case, InversionCase):
        return run_inversion_case(case, guard=guard)
    if isinstance(case, ExampleCase):
        return run_example_case(case, guard=guard)
    return run_propagation_case(case, guard=guard)
