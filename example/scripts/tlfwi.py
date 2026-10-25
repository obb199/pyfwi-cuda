"""Time-lapse FWI (TL-FWI): recover a change between two surveys.

What this example shows
-----------------------
* A time-lapse problem: a *baseline* survey and a *monitor* survey recorded over the
  same area, where the monitor model differs from the baseline in a small zone (here
  vp drops by 10 %, as when gas replaces brine in a reservoir).
* How to set up ``PyFWI.tl_fwi.TimeLapse``: it receives both observed data sets. They
  must already be in the five-component format of ``processing.prepare_residual``,
  because ``TimeLapse`` hands them to its inner ``FWI`` as they are.
* The ``tl_method`` argument chooses the strategy. ``"cc"`` (cascaded) inverts the
  baseline data from the starting model, then the monitor data starting from the
  baseline result; the time-lapse estimate is ``dm = monitor result - baseline result``.
  The other strategies are ``"pa"`` (parallel), ``"dd"`` (double difference), ``"sim"``
  (simultaneous), ``"wa"`` (weighted average), ``"cj"`` (cascaded joint), ``"cd"``
  (central difference) and ``"cu"`` (cross-updating).
* The source wavelet is scaled for the absolute tolerances of the optimiser (see
  ``fwi.py``).

How to run it
-------------
::

    python example/scripts/tlfwi.py                       # full size, with a figure
    python example/scripts/tlfwi.py --size small --no-show

The test suite (``test/test_examples.py``) calls ``main(size="small")`` and checks that
the time-lapse estimate is negative and concentrated where the monitor model changed.

Requirements: like ``fwi.py``, this needs ``PyFWI.fwi`` to import (SciPy before 1.8 for
PyFWI 0.1.10; fixed in step 1.1 of the fork's roadmap).
"""
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter

import PyFWI.acquisition as acq
import PyFWI.processing as process
import PyFWI.wave_propagation as wave
from PyFWI.tl_fwi import TimeLapse

#: Factor applied to the source wavelet so that the misfit is of order 1 (see fwi.py).
SOURCE_AMPLITUDE = 1000.0

#: Problem sizes. nz, nx: model size in grid points; dh: grid spacing (m); ns: number of
#: sources; t: recording length (s); freqs: multi-scale stages (Hz); iter: iterations per stage.
SIZES = {
    "small": {"nz": 40, "nx": 60, "dh": 10.0, "ns": 3, "t": 0.4, "freqs": [45], "iter": [3]},
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 4, "t": 0.6, "freqs": [25, 45], "iter": [2, 2]},
}


def make_models(nz, nx):
    """Baseline, monitor and starting models.

    Returns
    -------
    baseline, monitor, start : dict[str, numpy.ndarray]
        ``{"vp", "vs", "rho"}`` (m/s, m/s, kg/m³), ``float32`` arrays ``(nz, nx)``.
        The baseline has a faster circular inclusion; the monitor is the baseline with
        vp lowered by 10 % in a box above the inclusion (rows ``nz/4`` to ``3nz/8``,
        columns ``3nx/8`` to ``5nx/8``); the starting model is the smoothed baseline vp.
        vs and rho are the same everywhere (only vp is inverted).
    changed : numpy.ndarray of bool
        True where the monitor differs from the baseline.
    """
    z, x = np.mgrid[:nz, :nx]
    inclusion = (z - nz // 2) ** 2 + (x - nx // 2) ** 2 <= (nz // 10) ** 2
    vp_base = np.where(inclusion, 3000.0, 2500.0).astype(np.float32)
    changed = (z >= nz // 4) & (z < 3 * nz // 8) & (x >= 3 * nx // 8) & (x < 5 * nx // 8)
    vp_monitor = np.where(changed, 0.9 * vp_base, vp_base).astype(np.float32)
    vs = np.full((nz, nx), 2500.0 / np.sqrt(3.0), np.float32)
    rho = np.full((nz, nx), 310.0 * 2500.0 ** 0.25, np.float32)

    def model(vp):
        return {"vp": vp, "vs": vs.copy(), "rho": rho.copy()}
    return model(vp_base), model(vp_monitor), model(gaussian_filter(vp_base, nz / 4).astype(np.float32)), changed


def main(size="small", show=False):
    """Run a cascaded time-lapse inversion.

    Parameters
    ----------
    size : {"small", "full"}
        Problem size, see ``SIZES``.
    show : bool
        If True, plot the true and the estimated time-lapse vp change.

    Returns
    -------
    dict
        * ``"dm"``: estimated time-lapse change ``{"vp", "vs", "rho"}`` (monitor - baseline result);
        * ``"baseline_estimate"``, ``"monitor_estimate"``: the two inverted models;
        * ``"true_dm"``: the true vp change; ``"changed"``: boolean mask of the changed zone;
        * ``"baseline"``, ``"monitor"``, ``"start"``: the models; ``"inpa"``.

    Used by
    -------
    The command line (below) and ``test/test_examples.py::test_tlfwi``.
    """
    p = SIZES[size]
    nz, nx, dh = p["nz"], p["nx"], p["dh"]
    baseline, monitor, start, changed = make_models(nz, nx)
    inpa = {"ns": p["ns"], "sdo": 4, "fdom": 15, "dh": dh, "dt": 0.004, "acq_type": 1, "t": p["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh}  # see forward_surface.py for each key
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(inpa["ns"], dh, dh * nx, dh * nz, dh, inpa["sdo"],
                                                         acq_type=1)
    src_loc[:, 1] -= dh  # one row above the receivers (see forward_surface.py)
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])
    src.w = src.w * np.float32(SOURCE_AMPLITUDE)

    # The two observed data sets, in the five-component format TimeLapse expects.
    W = wave.WavePropagator(inpa, src, rec_loc, (nz, nx), components=0, n_well_rec=n_well_rec, chpr=0)
    d_baseline = process.prepare_residual(W.forward_modeling(baseline, show=False), 1)
    d_monitor = process.prepare_residual(W.forward_modeling(monitor, show=False), 1)

    inpa["energy_balancing"] = True
    tl = TimeLapse(d_baseline, d_monitor, inpa, src, rec_loc, (nz, nx), components=0, chpr=20, n_well_rec=n_well_rec)
    inverted, _ = tl(start, tl_method="cc", freqs=p["freqs"], iter=p["iter"], n_params=1, k_0=1, k_end=2)

    result = {"dm": inverted["dm"], "baseline_estimate": inverted["b1"], "monitor_estimate": inverted["m1"],
              "true_dm": monitor["vp"] - baseline["vp"], "changed": changed,
              "baseline": baseline, "monitor": monitor, "start": start, "inpa": inpa}
    if show:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        extent = [0, dh * nx, dh * nz, 0]
        for a, (title, image) in zip(ax, (("true vp change (m/s)", result["true_dm"]),
                                          ("estimated vp change (m/s)", result["dm"]["vp"]))):
            clip = np.abs(image).max()
            im = a.imshow(image, cmap="seismic", vmin=-clip, vmax=clip, extent=extent)
            a.set(title=title, xlabel="x (m)", ylabel="z (m)")
            plt.colorbar(im, ax=a)
        plt.tight_layout()
        plt.show()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-lapse FWI with the cascaded strategy.")
    parser.add_argument("--size", choices=sorted(SIZES), default="full", help="problem size")
    parser.add_argument("--no-show", action="store_true", help="do not plot")
    args = parser.parse_args()
    main(size=args.size, show=not args.no_show)
