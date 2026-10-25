"""Gradient of the misfit with the adjoint-state method, checkpoints and parameterisations.

What this example shows
-----------------------
* The ingredients of an FWI gradient: observed data (true model), predicted data
  (starting model), the L2 misfit and its adjoint source (``fwi_tools.cost_seismic``),
  and ``WavePropagator.gradient``.
* What ``chpr`` does. During the forward run PyFWI stores ``chpr`` percent of the time
  steps (the *checkpoints*) of the five fields, in host memory:
  ``5 x ns x nchp x (nz + 2 npml) x (nx + 2 npml) x 4 bytes``. The gradient run then
  rebuilds the forward field backwards in time between checkpoints. A few percent is
  enough for the exact gradient; ``chpr=1`` gives an approximation; ``chpr=0`` (the
  default of ``WavePropagator``!) keeps only two snapshots, and rebuilding the field
  backwards through the absorbing boundary over the whole recording is not exact: the
  gradient is wrong. Always pass ``chpr`` when you compute a gradient.
* The two parameterisations: ``parameterization="dv"`` (vp, vs, rho, the default) and
  any other value, which returns the gradient with respect to the Lamé parameters and
  density (lam, mu, rho). ``grad_switcher.grad_lmd_to_vd`` converts one into the other.
* Why the gradient points to the right place: the true model has a faster inclusion
  than the smooth starting model, and minus the vp gradient (the steepest-descent
  update) is positive inside it.

How to run it
-------------
::

    python example/scripts/gradient.py                       # full size, with figures
    python example/scripts/gradient.py --size small --no-show

The test suite (``test/test_examples.py``) calls ``main(size="small")`` and checks the
sign of the update, that checkpointed gradients agree, and the parameterisations.

Note for PyFWI 0.1.10: this script runs about a dozen propagations in one process, and
0.1.10 can then stop with ``clEnqueueReadBuffer failed: OUT_OF_RESOURCES`` on NVIDIA
GPUs (its kernels read outside their buffers; fixed in step 1.7 of the fork's roadmap).
If that happens, run the study for one ``chpr`` at a time.
"""
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

import PyFWI.acquisition as acq
import PyFWI.fwi_tools as tools
import PyFWI.processing as process
import PyFWI.wave_propagation as wave
from PyFWI.grad_switcher import grad_lmd_to_vd

#: Problem sizes. nz, nx: model size in grid points; dh: grid spacing (m);
#: ns: number of sources; t: recording length (s).
SIZES = {
    "small": {"nz": 40, "nx": 60, "dh": 10.0, "ns": 2, "t": 0.4},
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 5, "t": 0.8},
}

#: Checkpoint percentages compared by the example.
CHPR_STUDY = (0, 1, 5, 20)


def make_models(nz, nx):
    """True model (circular inclusion) and smooth starting model.

    Returns
    -------
    true, start : dict[str, numpy.ndarray]
        ``{"vp", "vs", "rho"}`` (m/s, m/s, kg/m³), ``float32`` arrays ``(nz, nx)``.
        The starting model is the true vp smoothed by a Gaussian filter of standard
        deviation ``nz / 4`` grid points, with vs and rho derived from it the same way.
    """
    z, x = np.mgrid[:nz, :nx]
    inside = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp = np.where(inside, 3000.0, 2500.0)

    def elastic(vp):
        model = {"vp": vp, "vs": vp / np.sqrt(3.0), "rho": 310.0 * vp ** 0.25}
        return {name: values.astype(np.float32) for name, values in model.items()}
    return elastic(vp), elastic(gaussian_filter(vp, nz / 4))


def cosine(a, b):
    """Cosine similarity of two arrays: 1 means the same direction (same gradient up to a scale)."""
    return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(size="small", show=False):
    """Compute the gradient at the starting model in several ways.

    Parameters
    ----------
    size : {"small", "full"}
        Problem size, see ``SIZES``.
    show : bool
        If True, plot the model perturbation, minus the gradient and the effect of ``chpr``.

    Returns
    -------
    dict
        * ``"misfit"``: value of the L2 misfit at the starting model;
        * ``"gradient"``: ``{chpr: {"vp", "vs", "rho"}}`` for every ``chpr`` of ``CHPR_STUDY``;
        * ``"gradient_lmd"``: gradient with respect to ``{"lam", "mu", "rho"}`` (``chpr=20``);
        * ``"gradient_lmd_as_dv"``: the latter converted to vp, vs, rho with ``grad_lmd_to_vd``;
        * ``"true"``, ``"start"``: the models; ``"inpa"``.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_gradient``.
    """
    p = SIZES[size]
    nz, nx, dh = p["nz"], p["nx"], p["dh"]
    true, start = make_models(nz, nx)
    inpa = {"ns": p["ns"], "sdo": 4, "fdom": 15, "dh": dh, "dt": 0.004, "acq_type": 1, "t": p["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh}  # see forward_surface.py for each key
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(inpa["ns"], dh, dh * nx, dh * nz, dh, inpa["sdo"],
                                                         acq_type=1)
    src_loc[:, 1] -= dh  # one row above the receivers (see forward_surface.py)
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])

    gradients, misfit = {}, None
    for chpr in CHPR_STUDY:
        W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=0, n_well_rec=n_well_rec, chpr=chpr)
        # prepare_residual turns the output of forward_modeling into the five-component
        # dictionary the cost function expects (missing components become zeros).
        d_obs = process.prepare_residual(W.forward_modeling(true, show=False), 1)
        d_est = process.prepare_residual(W.forward_modeling(start, show=False), 1)
        # misfit = 0.5 * ||d_est - d_obs||^2; adj_src = its derivative w.r.t. d_est (the adjoint source).
        misfit, adj_src = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))
        # gradient() back-propagates the adjoint source and correlates it with the forward
        # field rebuilt from the checkpoints of the *last* forward run (here: the starting model).
        gradients[chpr] = W.gradient(adj_src, show=False, parameterization="dv")
        if chpr == 20:
            gradient_lmd = W.gradient(adj_src, show=False, parameterization="lmd")

    # Lamé parameters of the starting model: mu = rho vs^2, lam = rho vp^2 - 2 mu.
    mu = start["rho"] * start["vs"] ** 2
    lam = start["rho"] * start["vp"] ** 2 - 2 * mu
    gvp, gvs, grho = grad_lmd_to_vd(gradient_lmd["lam"], gradient_lmd["mu"], gradient_lmd["rho"], lam, mu, start["rho"])

    if show:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        extent = [0, dh * nx, dh * nz, 0]
        perturbation = true["vp"] - start["vp"]
        clip = np.abs(perturbation).max()  # symmetric colour scale: white means no difference
        ax[0].imshow(perturbation, cmap="seismic", vmin=-clip, vmax=clip, extent=extent)
        ax[0].set(title="true vp - starting vp (m/s)", xlabel="x (m)", ylabel="z (m)")
        update = -gradients[20]["vp"]
        clip = np.abs(update).max()
        ax[1].imshow(update, cmap="seismic", vmin=-clip, vmax=clip, extent=extent)
        ax[1].set(title="minus the vp gradient (chpr = 20)", xlabel="x (m)")
        ax[2].bar([str(c) for c in CHPR_STUDY], [cosine(gradients[c]["vp"], gradients[20]["vp"]) for c in CHPR_STUDY])
        ax[2].set(title="agreement with chpr = 20", xlabel="chpr (%)", ylabel="cosine similarity", ylim=(0, 1.05))
        plt.tight_layout()
        plt.show()

    return {"misfit": misfit, "gradient": gradients, "gradient_lmd": gradient_lmd,
            "gradient_lmd_as_dv": {"vp": gvp, "vs": gvs, "rho": grho}, "true": true, "start": start, "inpa": inpa}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient of the misfit with the adjoint-state method.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="problem size")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
