"""Multi-scale full-waveform inversion (FWI) with L-BFGS.

What this example shows
-----------------------
* How to set up ``PyFWI.fwi.FWI``: it receives the observed data and the same
  acquisition description as ``WavePropagator`` (it *is* a ``WavePropagator``).
* The arguments of an inversion call:

  - ``method``: ``"lbfgs"`` (SciPy's L-BFGS-B) or ``"cg"`` (SciPy's non-linear conjugate gradient);
  - ``freqs``: the multi-scale strategy. The data are low-pass filtered below each
    frequency in turn (Hz), from low to high, and each stage starts from the result
    of the previous one;
  - ``iter``: the maximum number of iterations of each stage;
  - ``k_0``, ``k_end``, ``n_params``: which parameters are inverted. Parameters are
    numbered 1 = vp, 2 = vs, 3 = rho; PyFWI inverts parameters ``k_0`` to
    ``k_end - 1``, ``n_params`` at a time. ``k_0=1, k_end=2, n_params=1`` inverts vp only.

* ``inpa["energy_balancing"] = True``: the gradient is divided by the square root of
  the energy of the forward field, which compensates for the decay of amplitudes with
  depth (a simple preconditioner).
* Why the source amplitude matters in PyFWI 0.1.10. The optimiser stops when the
  largest gradient value is below an *absolute* tolerance (``pgtol = 1e-8`` for
  L-BFGS). With a unit-amplitude Ricker wavelet the gradients of a realistic model are
  around 1e-11 to 1e-8, so L-BFGS declares convergence before the first iteration and
  returns the starting model. Scaling the wavelet by ``SOURCE_AMPLITUDE`` scales the
  observed and the predicted data alike: the misfit and its gradient grow by the square
  of the factor, the problem itself does not change, and the optimiser can work.
* How to judge the result: the vp model gets closer to the true one, and vs and rho,
  which are not inverted, do not change.

How to run it
-------------
::

    python example/scripts/fwi.py                       # full size, with a figure
    python example/scripts/fwi.py --size small --no-show

The test suite (``test/test_examples.py``) calls ``main(size="small")`` and checks that
the model error decreases and that vp increases inside the faster inclusion.

Requirements: ``PyFWI.fwi`` imports ``MemoizeJac`` from ``scipy.optimize.optimize``,
which only SciPy versions before 1.8 provide in PyFWI 0.1.10 (fixed in step 1.1 of
the fork's roadmap). The inversion prints the progress of SciPy's optimiser.
"""
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

import PyFWI.acquisition as acq
import PyFWI.wave_propagation as wave
from PyFWI.fwi import FWI

#: Factor applied to the source wavelet so that the misfit is of order 1 (see the module
#: docstring). Any value above ~1000 gives the same inverted model.
SOURCE_AMPLITUDE = 1000.0

#: Problem sizes. nz, nx: model size in grid points; dh: grid spacing (m); ns: number of
#: sources; t: recording length (s); freqs: multi-scale stages (Hz); iter: iterations per stage.
SIZES = {
    "small": {"nz": 40, "nx": 60, "dh": 10.0, "ns": 3, "t": 0.4, "freqs": [25, 45], "iter": [2, 2]},
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 4, "t": 0.6, "freqs": [25, 45], "iter": [2, 2]},
}


def make_models(nz, nx):
    """True model (circular inclusion) and smooth starting model.

    Returns
    -------
    true, start : dict[str, numpy.ndarray]
        ``{"vp", "vs", "rho"}`` (m/s, m/s, kg/m³), ``float32`` arrays ``(nz, nx)``.
        The starting vp is the true one smoothed by a Gaussian filter of standard
        deviation ``nz / 4`` grid points; vs and rho are the same in both models, so
        that only vp has to be recovered.
    """
    z, x = np.mgrid[:nz, :nx]
    inside = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp = np.where(inside, 3000.0, 2500.0).astype(np.float32)
    background = {"vs": np.full((nz, nx), 2500.0 / np.sqrt(3.0), np.float32),
                  "rho": np.full((nz, nx), 310.0 * 2500.0 ** 0.25, np.float32)}
    true = {"vp": vp, **background}
    start = {"vp": gaussian_filter(vp, nz / 4).astype(np.float32), **{k: v.copy() for k, v in background.items()}}
    return true, start


def main(size="small", show=False):
    """Invert vp from surface data, starting from a smooth model.

    Parameters
    ----------
    size : {"small", "full"}
        Problem size, see ``SIZES``.
    show : bool
        If True, plot the true, starting and estimated vp.

    Returns
    -------
    dict
        * ``"estimate"``: the estimated model ``{"vp", "vs", "rho"}``, ``(nz, nx)`` arrays;
        * ``"misfit"``: the misfit at the end of each multi-scale stage, as returned by ``FWI``;
        * ``"error_start"``, ``"error_end"``: L2 norm of (vp - true vp) before and after;
        * ``"true"``, ``"start"``: the models; ``"inpa"``.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_fwi``.
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
    src.w = src.w * np.float32(SOURCE_AMPLITUDE)  # src.w: the wavelet samples, one per dt

    # "Observed" data: here they are modelled in the true model.
    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=0, n_well_rec=n_well_rec, chpr=0)
    d_obs = W.forward_modeling(true, show=False)

    inpa["energy_balancing"] = True
    # chpr matters here: FWI computes gradients (see gradient.py; chpr=0 gives a wrong one).
    fwi = FWI(d_obs, inpa, src, rec_loc, (nz, nx), components=0, chpr=20, n_well_rec=n_well_rec)
    estimate, misfit = fwi(start, method="lbfgs", freqs=p["freqs"], iter=p["iter"], n_params=1, k_0=1, k_end=2)

    result = {"estimate": estimate, "misfit": np.asarray(misfit),
              "error_start": float(np.linalg.norm(start["vp"] - true["vp"])),
              "error_end": float(np.linalg.norm(estimate["vp"] - true["vp"])),
              "true": true, "start": start, "inpa": inpa}
    if show:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        extent = [0, dh * nx, dh * nz, 0]
        for a, (title, model) in zip(ax, (("true vp", true), ("starting vp", start), ("estimated vp", estimate))):
            im = a.imshow(model["vp"], cmap="jet", vmin=2400, vmax=3000, extent=extent)
            a.set(title=title, xlabel="x (m)")
        ax[0].set_ylabel("z (m)")
        plt.colorbar(im, ax=ax, label="m/s")
        plt.show()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scale FWI with L-BFGS.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="problem size")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
