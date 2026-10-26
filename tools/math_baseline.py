"""Mathematical baseline of PyFWI (roadmap step 0.6): how correct is the physics and the gradient?

The golden masters check that results do not *change*; this script checks how *right*
they are, with four metrics computed on small homogeneous or inclusion models:

``gradient_fd`` (vp), ``gradient_fd_vs`` (vs) and ``gradient_fd_rho`` (density)
    The adjoint-state gradient against centred finite differences of the misfit
    J(m) = 1/2 ||d(m) - d_obs||^2. For six smooth random perturbations dm of vp (vs, rho),
    limited to the interior of the model, the ratio
    [J(m + eps dm) - J(m - eps dm)] / (2 eps) / <g, dm> should be 1. Metric: the median
    of |ratio - 1| (and the six ratios). PyFWI sizes its absorbing layer with the
    largest vp of the model, which the perturbations of vp would change; a corner cell of
    the starting model, outside the perturbations, holds the largest vp, so that the
    finite differences see the model and not a moving boundary (without it, 0.1.10
    measured 31 % instead of 7 %).
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
``convergence``
    The observed order of accuracy in space, orders 4 and 8, by self-convergence: the
    pressure 100 m from a source in a homogeneous 800 m model at grid spacings of 20, 10 and
    5 m, against the same run at 2.5 m, with the same time step (0.2 ms, so ``dt_scale`` is
    1 and the time error is the same in every run), before any boundary reflection arrives.
    A source added to one cell is a moment proportional to the cell area, so traces are
    divided by dh^2. The wavelet is a Ricker over +-1.5 / fdom: the default one (+-1 / fdom)
    starts and ends with a jump of 1e-3 of its peak, whose broadband content no grid
    resolves, a floor of 1e-3 for the differences. Metric: the relative L2 differences and
    their ratios as orders, log2(e(2h) / e(h)). FP32 rounding sets a floor of about 1e-6.
``adjoint_dot``
    The dot-product test of one time step: |<(A - I) u, v> - <u, (A^T - I) v>| divided by
    |<(A - I) u, v>|, A being the forward step of the propagation engine and A^T the adjoint
    updates of its gradient loop, for random fields away from the edges, without absorbing
    layer. It needs the engine of the fork (``None`` for the published 0.1.10, whose public
    API exposes no single time step); the ``legacy`` mode runs the kernels of 0.1.10 in their
    order, so it measures them.

The script prints the metrics as JSON. With ``--write`` (only with the published
0.1.10, like ``make_golden.py``) it stores them as the baseline in
``test/golden/math_baseline.json``; ``test/test_math.py`` then requires every later
version not to be worse.

The numeric mode is that of ``PYFWI_NUMERICS`` (``test/test_math.py`` measures both).

Usage::

    ../envs/ref-0.1.10/bin/python tools/math_baseline.py --write
    ../envs/dev/bin/python tools/math_baseline.py

Used by
-------
``test/test_math.py``.
"""
import argparse
import itertools
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

#: Grid spacings (m) of the convergence study, the last one being the reference, and its time step (s).
CONVERGENCE_SPACINGS, CONVERGENCE_DT = (20.0, 10.0, 5.0, 2.5), 0.0002

#: The five wavefield components, in the order of the engine's field buffers.
COMPONENTS = ("vx", "vz", "taux", "tauz", "tauxz")


def homogeneous_run(size, dh, order, src_xz, rec_xz, t, npml=20, chpr=0, dt=DT, half_width=1.0):
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
    dt : float, optional
        Time step of the recording and of the source wavelet in seconds.
    half_width : float, optional
        The Ricker wavelet spans +-half_width / fdom (``Source.Ricker``: 1).
    """
    import PyFWI.acquisition as acq
    import PyFWI.wave_propagation as wave

    n = round(size / dh)
    model = {"vp": np.full((n, n), VELOCITY, np.float32), "vs": np.zeros((n, n), np.float32),
             "rho": np.ones((n, n), np.float32)}
    rec_loc = np.asarray(rec_xz, np.float32)
    rec_dis = float(rec_loc[1, 0] - rec_loc[0, 0]) if len(rec_loc) > 1 else dh
    inpa = {"ns": 1, "sdo": order, "fdom": FDOM, "dh": dh, "dt": dt, "acq_type": 1, "t": t, "npml": npml,
            "pmlR": 1e-5, "pml_dir": 2, "rec_dis": rec_dis}
    src = acq.Source(np.asarray([src_xz], np.float32), dh, dt)
    src.Ricker(FDOM)
    if half_width != 1.0:  # the same Ricker over a longer support
        tw = np.arange(-half_width / FDOM, half_width / FDOM + dt / 3, dt)
        src.w = np.float32((1.0 - 2.0 * (np.pi * FDOM * tw) ** 2) * np.exp(-(np.pi * FDOM * tw) ** 2))
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


def metric_gradient_fd(parameter="vp"):
    """Ratios of finite-difference directional derivatives to <gradient, direction> (should be 1).

    Parameters
    ----------
    parameter : str, optional
        The model parameter perturbed and the gradient compared: ``"vp"`` or ``"vs"``.
    """
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    from helpers import tiny_case

    cost = tools.CostFunction("l2")
    with zero_guard_bands():
        case = tiny_case(ns=2, nz=40, nx=40, chpr=100, t=0.3)
        observed = process.prepare_residual(case.W.forward_modeling(case.model, False), 1)
        # The absorbing layer depends on max(vp) (WavePropagator.pml_preparation): pin it with the
        # bottom-right cell, outside the perturbations, so that it is the same in every run.
        m0 = {**case.m0, "vp": case.m0["vp"].copy()}
        m0["vp"][-1, -1] = m0["vp"].max() + 100.0

        def misfit(model):
            predicted = process.prepare_residual(case.W.forward_modeling(model, False), 1)
            return tools.cost_seismic(predicted, observed, fun=cost)

        _, adjoint_source = misfit(m0)
        gradient = case.W.gradient(adjoint_source, False)[parameter].astype(np.float64)
        rng = np.random.default_rng(0)
        interior = np.zeros((40, 40))
        interior[12:34, 6:34] = 1.0  # away from the edges and the surface, where PyFWI masks the gradient
        ratios, eps = [], 5.0  # m/s
        for _ in range(6):
            direction = gaussian_filter(rng.standard_normal((40, 40)), 3) * interior
            direction = (direction / np.abs(direction).max()).astype(np.float32)
            plus = {**m0, parameter: (m0[parameter] + eps * direction).astype(np.float32)}
            minus = {**m0, parameter: (m0[parameter] - eps * direction).astype(np.float32)}
            fd = (misfit(plus)[0] - misfit(minus)[0]) / (2 * eps)
            ratios.append(float(fd / np.sum(gradient * direction)))
    return {"median_abs_error": float(np.median(np.abs(np.array(ratios) - 1))), "ratios": ratios}


def metric_convergence():
    """Observed orders of accuracy in space, orders 4 and 8 (self-convergence, see the module docstring)."""
    result = {}
    for order in (4, 8):
        traces = {}
        for dh in CONVERGENCE_SPACINGS:
            pressure, _ = homogeneous_run(800.0, dh, order, (400.0, 400.0), [(500.0, 400.0), (600.0, 400.0)],
                                          t=0.26, dt=CONVERGENCE_DT, half_width=1.5)
            traces[dh] = pressure[:, 0] / dh ** 2
        reference = traces[CONVERGENCE_SPACINGS[-1]]
        errors = [float(np.linalg.norm(traces[dh] - reference) / np.linalg.norm(reference))
                  for dh in CONVERGENCE_SPACINGS[:-1]]
        result[f"order_{order}"] = {"errors": errors,
                                    "orders": [float(np.log2(a / b)) for a, b in itertools.pairwise(errors)]}
    return result


def metric_adjoint_dot():
    """Relative mismatch of the dot-product test of one time step (``None`` without the fork's engine)."""
    from helpers import tiny_case

    case = tiny_case()
    if not hasattr(case.W, "_engine"):  # the published 0.1.10
        return {"relative_mismatch": None}
    case.W.forward_modeling(case.model, False)  # the model and the arguments of the kernels
    engine = case.W._engine
    engine.prepare_gradient()
    zero = np.zeros((engine.tnz, engine.tnx), np.float32)
    engine.set_pml(zero, zero)
    rng = np.random.default_rng(3)
    # Stresses scaled by an impedance (2,200 kg/m3 x 2,500 m/s), adjoint stresses by its inverse,
    # so that no block of the step dominates the dot products.
    scale = {"vx": 1.0, "vz": 1.0, "taux": 5.5e6, "tauz": 5.5e6, "tauxz": 5.5e6}

    def random_fields(power):
        fields = {}
        for par in COMPONENTS:
            field = zero.copy()
            field[12:-12, 12:-12] = rng.standard_normal((engine.tnz - 24, engine.tnx - 24)) * scale[par] ** power
            fields[par] = field
        return fields

    def apply(buffers, fields, step):
        for buffer, par in zip(buffers, COMPONENTS):
            engine.backend.to_device(buffer, fields[par])
        step()
        out = {par: np.empty_like(zero) for par in COMPONENTS}
        for buffer, par in zip(buffers, COMPONENTS):
            engine.backend.to_host(out[par], buffer)
        return out

    u, v = random_fields(1), random_fields(-1)
    engine._set_update_args(np.int32(1))
    au = apply(engine._field_buffers(), u, engine._forward_step)
    engine._set_adjoint_args()
    adjoint = (engine.avx_b, engine.avz_b, engine.ataux_b, engine.atauz_b, engine.atauxz_b)
    atv = apply(adjoint, v, engine._adjoint_updates)
    left = sum(np.sum((au[p].astype(np.float64) - u[p]) * v[p]) for p in COMPONENTS)
    right = sum(np.sum(u[p].astype(np.float64) * (atv[p].astype(np.float64) - v[p])) for p in COMPONENTS)
    return {"relative_mismatch": float(abs(left - right) / abs(left))}


def measure():
    """All metrics, as a JSON-serialisable dict."""
    return {"gradient_fd": metric_gradient_fd("vp"), "gradient_fd_vs": metric_gradient_fd("vs"),
            "gradient_fd_rho": metric_gradient_fd("rho"), "analytic": metric_analytic(),
            "pml_reflection": metric_pml_reflection(), "late_energy": metric_late_energy(),
            "convergence": metric_convergence(), "adjoint_dot": metric_adjoint_dot()}


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
