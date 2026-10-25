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

Every run happens inside zero guard bands with a race-free geometry, the two rules
that make PyFWI 0.1.10 reproducible (see ``test_reproducibility.py``).

Used by
-------
``tools/make_golden.py`` and the characterisation tests.
"""
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


def run_propagation_case(case: PropagationCase):
    """Run the forward modelling and the gradient of one case and return the layered outputs.

    The OpenCL device is the one selected by ``PYOPENCL_CTX``.

    Parameters
    ----------
    case : PropagationCase
        The case to run.

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
    with zero_guard_bands():
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
