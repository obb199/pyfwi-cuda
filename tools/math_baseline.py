"""Mathematical baseline of PyFWI (roadmap step 0.6): how correct is the physics and the gradient?

The golden masters check that results do not *change*; this script checks how *right*
they are, with four metrics computed on small homogeneous or inclusion models:

``gradient_fd``
    The adjoint-state gradient against centred finite differences of the misfit
    J(m) = 1/2 ||d(m) - d_obs||^2. For six smooth random perturbations dm of vp,
    limited to the interior of the model, the ratio
    [J(m + eps dm) - J(m - eps dm)] / (2 eps) / <g, dm> should be 1. Metric: the median
    of |ratio - 1| (and the six ratios).
``analytic``
    A homogeneous acoustic medium (vs = 0, rho = 1) against the 2D Green's function
    convolved with the time derivative of the Ricker wavelet, at 100 to 400 m from the
    source, orders 4 and 8. Traces are normalised (the source scaling of PyFWI is not
    modelled). Metrics: the smallest correlation and the largest peak-time difference.
``pml_reflection``
    The same source and receiver in a small model (the receiver 100 m from the
    absorbing boundary) and in a large one (boundary too far to matter within the
    recording). Metric: max |small - large| / max |large| at the receiver.
``late_energy``
    Energy of the wavefield at the end of a 1.2 s run divided by its peak, in a 600 m
    homogeneous model (the wave should have left through the absorbing boundary).

The script prints the metrics as JSON. With ``--write`` (only with the published
0.1.10, like ``make_golden.py``) it stores them as the baseline in
``test/golden/math_baseline.json``; ``test/test_math.py`` then requires every later
version not to be worse.

Not measured on 0.1.10, and why (see the roadmap): the convergence order (at grid
spacings of 4 to 6 m the source injection of 0.1.10 excites a spurious grid-scale
oscillation, a known bug) and the adjoint dot-product test (the public API exposes no
single time step and its adjoint).

Usage::

    ../envs/ref-0.1.10/bin/python tools/math_baseline.py --write
    ../envs/dev/bin/python tools/math_baseline.py

Used by
-------
``test/test_math.py``.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "test"), REPO]

from oob_guard import zero_guard_bands

#: Where the baseline of the published 0.1.10 is stored.
BASELINE_PATH = os.path.join(REPO, "test", "golden", "math_baseline.json")

#: Velocity (m/s), recording sampling (s) and Ricker frequency (Hz) of the homogeneous tests.
VELOCITY, DT, FDOM = 2500.0, 0.004, 15


def homogeneous_run(size, dh, order, src_xz, rec_xz, t, npml=20, chpr=0):
    """Forward modelling in a homogeneous acoustic square model; returns ``(pressure traces, propagator)``.

    Parameters
    ----------
    size : float
        Side of the square model in metres.
    dh : float
        Grid spacing in metres.
    order : int
        Spatial order (4 or 8).
    src_xz : tuple of float
        Source position ``(x, z)`` in metres.
    rec_xz : list of tuple
        Receiver positions in metres, on one horizontal line, equally spaced.
    t : float
        Recording length in seconds.
    npml : int, optional
        Thickness of the absorbing boundary in grid points.
    chpr : int, optional
        Checkpoint percentage (``late_energy`` reads the stored snapshots).
    """
    import PyFWI.acquisition as acq
    import PyFWI.wave_propagation as wave

    n = round(size / dh)
    model = {"vp": np.full((n, n), VELOCITY, np.float32), "vs": np.zeros((n, n), np.float32),
             "rho": np.ones((n, n), np.float32)}
    rec_loc = np.asarray(rec_xz, np.float32)
    rec_dis = float(rec_loc[1, 0] - rec_loc[0, 0]) if len(rec_loc) > 1 else dh
    inpa = {"ns": 1, "sdo": order, "fdom": FDOM, "dh": dh, "dt": DT, "acq_type": 1, "t": t, "npml": npml,
            "pmlR": 1e-5, "pml_dir": 2, "rec_dis": rec_dis}
    src = acq.Source(np.asarray([src_xz], np.float32), dh, DT)
    src.Ricker(FDOM)
    with zero_guard_bands():
        W = wave.WavePropagator(inpa, src, rec_loc, (n, n), components=0, chpr=chpr)
        traces = W.forward_modeling(model, show=False)["taux"]
    return np.asarray(traces, np.float64), W


def analytic_trace(r, nt, t):
    """Normalised 2D acoustic pressure at distance ``r`` (m) for PyFWI's source, sampled at ``DT``.

    The pressure of a stress (explosive) source with time function s(t) is the 2D Green's
    function H(t - r/c) / sqrt(t^2 - r^2/c^2) convolved with s'(t). The Green's function
    is evaluated at the midpoints of a fine grid so that its integrable singularity at
    t = r/c never falls on a grid point.
    """
    fine = 1e-5
    tt = (np.arange(int((t + DT) / fine)) + 0.5) * fine
    tw = np.arange(-1.0 / FDOM, 1.0 / FDOM + fine / 3, fine)
    wavelet = (1 - 2 * (np.pi * FDOM * tw) ** 2) * np.exp(-(np.pi * FDOM * tw) ** 2)
    green = np.where(tt > r / VELOCITY, 1.0 / np.sqrt(np.maximum(tt ** 2 - (r / VELOCITY) ** 2, 1e-30)), 0.0)
    pressure = np.convolve(green, np.gradient(wavelet, fine))[: tt.size] * fine
    sampled = np.interp(np.arange(nt) * DT, tt, pressure)
    return sampled / np.abs(sampled).max()


def metric_analytic():
    """Smallest correlation and largest peak-time difference (s) with the analytic solution, orders 4 and 8."""
    correlations, shifts = [], []
    for order in (4, 8):
        rec = [(500.0 + k * 100.0, 500.0) for k in range(1, 5)]
        traces, _ = homogeneous_run(1000.0, 10.0, order, (500.0, 500.0), rec, t=0.4)
        for k, (x, _) in enumerate(rec):
            a = analytic_trace(x - 500.0, traces.shape[0], 0.4)
            b = traces[:, k] / np.abs(traces[:, k]).max()
            correlations.append(float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))
            shifts.append(abs(int(np.argmax(np.abs(a))) - int(np.argmax(np.abs(b)))) * DT)
    return {"min_correlation": min(correlations), "max_peak_shift": max(shifts), "correlations": correlations}


def metric_pml_reflection():
    """Relative boundary reflection at a receiver 100 m from the absorbing boundary."""
    small, _ = homogeneous_run(600.0, 10.0, 4, (300.0, 300.0), [(400.0, 300.0), (500.0, 300.0)], t=0.6)
    # The same source-receiver geometry in a 1600 m model: the boundary is 500 m farther away.
    large, _ = homogeneous_run(1600.0, 10.0, 4, (800.0, 800.0), [(900.0, 800.0), (1000.0, 800.0)], t=0.6)
    return {"relative_error": float(np.abs(small[:, 1] - large[:, 1]).max() / np.abs(large[:, 1]).max())}


def metric_late_energy():
    """Energy of the wavefield at the end of a 1.2 s run divided by its peak (600 m model, 10 m grid)."""
    _, W = homogeneous_run(600.0, 10.0, 4, (300.0, 300.0), [(400.0, 300.0), (500.0, 300.0)], t=1.2, chpr=10)
    energy = [float(np.sum(W.W["taux"][:, :, 0, k].astype(np.float64) ** 2)) for k in range(W.nchp)]
    return {"end_over_peak": energy[-1] / max(energy)}


def metric_gradient_fd():
    """Ratios of finite-difference directional derivatives to <gradient, direction> (should be 1)."""
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    from helpers import tiny_case

    cost = tools.CostFunction("l2")
    with zero_guard_bands():
        case = tiny_case(ns=2, nz=40, nx=40, chpr=100, t=0.3)
        observed = process.prepare_residual(case.W.forward_modeling(case.model, False), 1)

        def misfit(model):
            predicted = process.prepare_residual(case.W.forward_modeling(model, False), 1)
            return tools.cost_seismic(predicted, observed, fun=cost)

        _, adjoint_source = misfit(case.m0)
        gradient = case.W.gradient(adjoint_source, False)["vp"].astype(np.float64)
        rng = np.random.default_rng(0)
        interior = np.zeros((40, 40))
        interior[12:34, 6:34] = 1.0  # away from the edges and the surface, where PyFWI masks the gradient
        ratios, eps = [], 5.0  # m/s
        for _ in range(6):
            direction = gaussian_filter(rng.standard_normal((40, 40)), 3) * interior
            direction = (direction / np.abs(direction).max()).astype(np.float32)
            plus = {**case.m0, "vp": (case.m0["vp"] + eps * direction).astype(np.float32)}
            minus = {**case.m0, "vp": (case.m0["vp"] - eps * direction).astype(np.float32)}
            fd = (misfit(plus)[0] - misfit(minus)[0]) / (2 * eps)
            ratios.append(float(fd / np.sum(gradient * direction)))
    return {"median_abs_error": float(np.median(np.abs(np.array(ratios) - 1))), "ratios": ratios}


def measure():
    """All metrics, as a JSON-serialisable dict."""
    return {"gradient_fd": metric_gradient_fd(), "analytic": metric_analytic(),
            "pml_reflection": metric_pml_reflection(), "late_energy": metric_late_energy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="store as the baseline (published 0.1.10 only)")
    args = parser.parse_args()
    import PyFWI
    origin = os.path.dirname(os.path.abspath(PyFWI.__file__))
    if args.write and (PyFWI.__version__ != "0.1.10" or origin.startswith(REPO + os.sep)):
        parser.error("--write needs the published PyFWI 0.1.10 (run with ../envs/ref-0.1.10/bin/python)")
    if "PYOPENCL_CTX" not in os.environ:
        from tools.make_golden import default_opencl_context
        default_opencl_context()
    metrics = measure()
    print(json.dumps(metrics, indent=1))
    if args.write:
        with open(BASELINE_PATH, "w") as f:
            json.dump({"pyfwi": PyFWI.__version__, "metrics": metrics}, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
