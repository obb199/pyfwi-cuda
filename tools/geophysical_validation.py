"""Geophysical validation of the CUDA backend (roadmap step 4.6): reference inversions on both backends.

The other checks of the fork compare numbers: seismograms, gradients, a few iterations of an inversion
(``tools/parity_report.py``). This one runs what a geophysicist runs with PyFWI, complete multi-scale
inversions, on the GPU of the project, and writes a report with figures and misfit curves for the
review of a geophysicist (``docs/cuda-migration/reports/geophysical-validation-<version>.md``).

The cases
---------
The model of Louboutin et al. (2018), as ``PyFWI.model_dataset.ModelGenerator("louboutin")`` builds
it: a homogeneous background (vp 2500 m/s) with a faster circular inclusion (vp 3000 m/s), vs from a
Poisson's ratio of 0.25 and the density from Gardner's relation. Two cases, after the examples of the
original repository:

* ``fwi``: multi-scale FWI of vp with L-BFGS, surface acquisition (``example/fwi_example_surface.py``);
* the eight time-lapse strategies of ``PyFWI.tl_fwi.TimeLapse`` (``cc``, ``pa``, ``dd``, ``sim``, ``wa``,
  ``cj``, ``cd``, ``cu``: see ``STRATEGIES``), crosswell acquisition, with the monitor model of
  ``example/tlfwi_example_crosswell.ipynb``: the baseline with vp 20 % higher in a box above the inclusion.

Every case runs in three configurations on the same GPU (``CONFIGURATIONS``):

* OpenCL in the ``legacy`` numeric mode, its default (the arithmetic of PyFWI 0.1.10);
* OpenCL in the ``fixed`` mode (the corrections of step 2.4);
* CUDA in the ``fixed`` mode, its default (the native kernels of Phase 3).

OpenCL ``fixed`` against CUDA ``fixed`` isolates the backend: the same mathematics, other kernels and
compilers. OpenCL ``legacy`` against the ``fixed`` modes shows what the corrections of the ``fixed``
mode change in a complete inversion, the question of gate G3 about the default mode of OpenCL.

Departures from the original examples, each for a reason:

* The starting model is the background of the model (``ModelGenerator("louboutin")(smoothing=1)``, as in
  the examples) for vp only; vs and rho are the true ones. The examples set vs to 0 and rho to 1 in the
  starting model while the data come from the true vs and rho: as only vp is inverted, the inversion
  would then try to explain the vs and rho of the data with vp. Here vp is the only unknown.
* The source wavelet is multiplied by ``SOURCE_AMPLITUDE``: the optimisers of the ``legacy`` mode stop
  on absolute tolerances (see ``example/scripts/fwi.py``); the problem itself does not change.
* Surface acquisition: the sources are one grid row above the receivers (see
  ``example/scripts/forward_surface.py``).
* Longer inversions (``SIZES``): the examples run 2 iterations at 25 Hz and 2 at 45 Hz, which leaves the
  model almost unchanged; here 15 iterations in each of three frequency bands.

What is measured
----------------
For every case and configuration: the vp estimates (the inverted model for ``fwi``, every time-lapse
estimate ``dm...`` that the strategy returns), the misfit of every evaluation (one forward modelling and
one gradient, or two for the joint inversions of ``sim`` and ``cj``), the number of evaluations and the
run time. From them (``fwi_metrics``, ``time_lapse_metrics``, ``agreement``):

* FWI: the model error after the inversion over the error of the starting model, and the share of the
  velocity contrast of the inclusion that the estimate recovers;
* time-lapse: the share of the change recovered inside the box, the correlation of the estimate with the
  true change, the level of the artefacts outside the box;
* the agreement of two configurations: the norm of the difference of their estimates over the norm of
  what the inversion has to recover (the model update for FWI, the true change for time-lapse).

Usage::

    ../envs/dev/bin/python tools/geophysical_validation.py          # all cases, about one hour
    ../envs/dev/bin/python tools/geophysical_validation.py --size small --cases fwi cc --out /tmp/geo
    ../envs/dev/bin/python tools/geophysical_validation.py --redraw  # figures and report from saved results

Outputs: ``bench/geophysical-<version>/results.json`` and ``models.npz`` (the vp estimates), the figures
``docs/cuda-migration/reports/figures/validation_*.png`` and the report. With ``--out DIR`` everything goes to
DIR.

Used by
-------
Step 4.6 of the roadmap and gate G4. Tested by ``test/test_geophysical_validation.py``.
"""
import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # after choosing the backend
from matplotlib.colors import LinearSegmentedColormap

from tools.plot_overview import AQUA, AXIS, BLUE, GRID, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE

#: Where the results, the figures and the report of a release go (``{}``: the version of PyFWI).
RESULTS = os.path.join(REPO, "bench", "geophysical-{}")
FIGURES = os.path.join(REPO, "docs", "cuda-migration", "reports", "figures")  # figures/ is the overview's
REPORT = os.path.join(REPO, "docs", "cuda-migration", "reports", "geophysical-validation-{}.md")

#: The configurations: ``(name, label, backend, numerics)``; the subject of the validation, CUDA, first.
CONFIGURATIONS = (
    ("cuda-fixed", "CUDA, fixed (its default)", "cuda:strict", "fixed"),
    ("opencl-fixed", "OpenCL, fixed", "opencl", "fixed"),
    ("opencl-legacy", "OpenCL, legacy (its default)", "opencl", "legacy"),
)

#: Short names of the configurations, for the titles of the figures.
SHORT = {"cuda-fixed": "CUDA", "opencl-fixed": "OpenCL fixed", "opencl-legacy": "OpenCL legacy"}

#: The pairs of configurations compared: ``(name, first, second, what the difference measures)``.
PAIRS = (
    ("backend", "cuda-fixed", "opencl-fixed", "the backend: CUDA against OpenCL, both in the fixed mode"),
    ("numerics", "opencl-legacy", "opencl-fixed", "the numeric mode: legacy against fixed, both on OpenCL"),
    ("defaults", "cuda-fixed", "opencl-legacy", "the defaults: CUDA (fixed) against OpenCL (legacy)"),
)

#: The time-lapse strategies of ``TimeLapse``: ``{tl_method: (name, what it does, main estimate)}``. The
#: main estimate is the one that uses every inversion of the strategy (the figures show it; the tables
#: give every estimate).
STRATEGIES = {
    "cc": ("Cascaded",
           "the baseline from the starting model, then the monitor from the baseline result; dm = monitor - baseline",
           "dm"),
    "pa": ("Parallel",
           "the baseline and the monitor, each from the starting model; dm = monitor - baseline",
           "dm"),
    "dd": ("Double difference",
           ("the baseline, then the composite data (monitor data - baseline data + the data of the baseline result) "
            "from the starting model; dm = composite - baseline"),
           "dm"),
    "sim": ("Simultaneous",
            ("the baseline, then the baseline and the monitor together (one misfit for both data sets) from the "
             "baseline result; dm = monitor - baseline"),
            "dm"),
    "wa": ("Weighted average",
           ("cascaded, then the baseline again from the monitor result: dm11 = m1 - b1, dm12 = m1 - b2 (the weights "
            "of the average are not in PyFWI)"),
           "dm12"),
    "cj": ("Cascaded joint",
           ("a simultaneous inversion from the starting model, then another from the average of its two models: "
            "dm0 = m1 - b1, dm7 = m2 - b2, dm3 = m2 - b1, dm6 = m1 - b2"),
           "dm7"),
    "cd": ("Central difference",
           ("baseline then monitor, and monitor then baseline, both from the starting model: dm3 and dm4 are the "
            "two cascaded differences, dm34 their average, dm0 and dm1 the differences of the parallel and of the "
            "second inversions"),
           "dm34"),
    "cu": ("Cross-updating",
           ("baseline, monitor, baseline, monitor, each from the previous result (Maharramov and Biondi, 2014): "
            "dmij = mi - bj"),
           "dm22"),
}

#: The cases, in the order they run.
CASES = ("fwi",) + tuple(STRATEGIES)

#: Factor applied to the source wavelet (see the module docstring and ``example/scripts/fwi.py``).
SOURCE_AMPLITUDE = 1000.0

#: Problem sizes. ``full``: the reference cases (the grid and the acquisition of the examples). ``small``:
#: the same geometry on a quarter of the grid, for the tests. nz, nx: grid points; dh: spacing (m); ns:
#: sources; t: recording (s); freqs: the frequency bands of the multi-scale inversion (Hz); iter: the
#: iterations of each band.
SIZES = {
    "full": {"nz": 100, "nx": 100, "dh": 7.0, "ns": 4, "t": 0.6, "freqs": [15, 25, 45], "iter": [15, 15, 15]},
    "small": {"nz": 50, "nx": 50, "dh": 7.0, "ns": 3, "t": 0.4, "freqs": [15, 45], "iter": [3, 3]},
}

#: The opt-in features of Phase 4 and the settings of the backends are not validated here: the variables
#: that start with this prefix are removed before running (the defaults are what is validated).
VARIABLE_PREFIX = "PYFWI_"


# ---------------------------------------------------------------------------- models and acquisitions

def make_models(nz, nx):
    """The baseline, monitor and starting models, and the inclusion and the box of the change.

    At 100 x 100 the baseline is ``ModelGenerator("louboutin")()`` and the monitor the one of
    ``example/tlfwi_example_crosswell.ipynb``; other sizes scale the geometry (inclusion of radius
    ``nz // 10`` at the centre, box over rows ``nz/5`` to ``3nz/10`` and columns ``7nx/20`` to ``11nx/20``).

    Returns
    -------
    baseline, monitor, start : dict[str, numpy.ndarray]
        ``{"vp", "vs", "rho"}``, ``float32`` arrays ``(nz, nx)`` (m/s, m/s, g/cm³). The monitor differs
        from the baseline by vp only; the starting model has the vp of the background and the vs and rho
        of the baseline (see the module docstring).
    inclusion, box : numpy.ndarray of bool
        Where the inclusion is; where the monitor differs from the baseline.
    """
    import PyFWI.rock_physics as rp
    from PyFWI.model_dataset import add_circle, background

    def properties(vp):
        return {"vp": vp, "vs": rp.ShearVelocity().poisson_ratio_vs(vp), "rho": rp.Density().gardner(vp)}

    # The steps of ModelGenerator("louboutin"), with the geometry scaled to the grid.
    baseline = background((nz, nx), properties(2500.0))
    for par, value in properties(3000.0).items():
        baseline[par] = add_circle(baseline[par], value, r=nz // 10, cx=nx // 2, cz=nz // 2)
    start = {"vp": background((nz, nx), {"vp": 2500.0})["vp"], "vs": baseline["vs"].copy(),
             "rho": baseline["rho"].copy()}
    monitor = {par: values.copy() for par, values in baseline.items()}
    rows, columns = slice(nz // 5, 3 * nz // 10), slice(7 * nx // 20, 11 * nx // 20)
    monitor["vp"][rows, columns] += 0.2 * monitor["vp"][rows, columns]  # as the notebook does
    box = np.zeros((nz, nx), bool)
    box[rows, columns] = True
    return baseline, monitor, start, baseline["vp"] > 2500.0, box


def acquisition(p, acq_type, backend, numerics):
    """``(inpa, src, rec_loc, n_well_rec)`` of a case: surface (``acq_type=1``) or crosswell (``0``).

    The parameters of the examples (see ``example/scripts/forward_surface.py`` for each key), with the
    backend and the numeric mode of the configuration in ``inpa``.
    """
    import PyFWI.acquisition as acq

    nz, nx, dh = p["nz"], p["nx"], p["dh"]
    inpa = {"ns": p["ns"], "sdo": 4, "fdom": 15, "dh": dh, "dt": 0.004, "acq_type": acq_type, "t": p["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": dh, "backend": backend, "numerics": numerics}
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(p["ns"], dh, dh * nx, dh * nz, dh, inpa["sdo"],
                                                         acq_type=acq_type)
    if acq_type == 1:
        src_loc[:, 1] -= dh  # one row above the receivers (see forward_surface.py)
    src = acq.Source(src_loc, dh, inpa["dt"])
    src.Ricker(inpa["fdom"])
    src.w = src.w * np.float32(SOURCE_AMPLITUDE)
    return inpa, src, rec_loc, n_well_rec


# ---------------------------------------------------------------------------- recording the inversions

class Recorder:
    """The misfit of every evaluation of the inversions of one case, stage by stage.

    ``watch`` wraps, on an ``FWI`` or ``TimeLapse`` object, the method that runs one stage (one frequency
    band: ``lbfgs``, ``tl_lbfgs``) and the one that evaluates the misfit and its gradient (``fprime``,
    ``tl_fprime``). ``stages`` is then ``[{"inversion": i, "kind": "fwi" | "joint", "freq": Hz,
    "misfit": [...]}, ...]``: an inversion is ``n_freqs`` consecutive stages (one parameter, vp, is
    inverted), ``joint`` the inversions of the baseline and the monitor together (``sim``, ``cj``).
    """

    def __init__(self, n_freqs):
        self.n_freqs, self.stages = int(n_freqs), []

    def watch(self, obj, stage, evaluation, kind):
        """Wrap ``obj.<stage>`` and ``obj.<evaluation>`` (instance attributes shadow the methods)."""
        run_stage, evaluate = getattr(obj, stage), getattr(obj, evaluation)

        def wrapped_stage(*args, **kwargs):  # (model, iterations, freq, ...)
            self.stages.append({"inversion": len(self.stages) // self.n_freqs + 1, "kind": kind,
                                "freq": float(args[2]), "misfit": []})
            return run_stage(*args, **kwargs)

        def wrapped_evaluation(*args, **kwargs):
            misfit, gradient = evaluate(*args, **kwargs)
            self.stages[-1]["misfit"].append(float(misfit))
            return misfit, gradient

        setattr(obj, stage, wrapped_stage)
        setattr(obj, evaluation, wrapped_evaluation)


def run_fwi(p, backend, numerics):
    """Multi-scale FWI of vp, surface acquisition: ``({"vp": estimate}, stages)``."""
    import PyFWI.wave_propagation as wave
    from PyFWI.fwi import FWI

    true, _, start, _, _ = make_models(p["nz"], p["nx"])
    inpa, src, rec_loc, n_well_rec = acquisition(p, 1, backend, numerics)
    shape = (p["nz"], p["nx"])
    W = wave.WavePropagator(inpa, src, rec_loc, shape, components=0, n_well_rec=n_well_rec, chpr=0)
    d_obs = W.forward_modeling(true, show=False)
    inpa["energy_balancing"] = True
    fwi = FWI(d_obs, inpa, src, rec_loc, shape, components=0, chpr=20, n_well_rec=n_well_rec)
    recorder = Recorder(len(p["freqs"]))
    recorder.watch(fwi, "lbfgs", "fprime", "fwi")
    estimate, _ = fwi(start, method="lbfgs", freqs=p["freqs"], iter=p["iter"], n_params=1, k_0=1, k_end=2)
    return {"vp": np.asarray(estimate["vp"], np.float32)}, recorder.stages


def run_time_lapse(p, strategy, backend, numerics):
    """One time-lapse strategy, crosswell acquisition: ``({estimate key: dm of vp}, stages)``."""
    import PyFWI.processing as process
    import PyFWI.wave_propagation as wave
    from PyFWI.tl_fwi import TimeLapse

    baseline, monitor, start, _, _ = make_models(p["nz"], p["nx"])
    inpa, src, rec_loc, n_well_rec = acquisition(p, 0, backend, numerics)
    shape = (p["nz"], p["nx"])
    W = wave.WavePropagator(inpa, src, rec_loc, shape, components=0, n_well_rec=n_well_rec, chpr=0)
    d_baseline = process.prepare_residual(W.forward_modeling(baseline, show=False), 1)
    d_monitor = process.prepare_residual(W.forward_modeling(monitor, show=False), 1)
    inpa["energy_balancing"] = True
    tl = TimeLapse(d_baseline, d_monitor, inpa, src, rec_loc, shape, components=0, chpr=20, n_well_rec=n_well_rec)
    recorder = Recorder(len(p["freqs"]))
    recorder.watch(tl.inv_obj, "lbfgs", "fprime", "fwi")
    recorder.watch(tl, "tl_lbfgs", "tl_fprime", "joint")
    inverted, _ = tl(start, tl_method=strategy, freqs=p["freqs"], iter=p["iter"], n_params=1, k_0=1, k_end=2)
    estimates = {key: np.asarray(value["vp"], np.float32) for key, value in inverted.items() if key.startswith("dm")}
    return estimates, recorder.stages


# ---------------------------------------------------------------------------- metrics

def _norm(values):
    return float(np.linalg.norm(np.asarray(values, np.float64)))


def fwi_metrics(vp, true, start, inclusion):
    """``{"error_ratio", "contrast"}`` of an FWI estimate of vp.

    ``error_ratio``: ``||vp - true|| / ||start - true||`` (1: no progress, 0: the true model);
    ``contrast``: the mean velocity increase inside the inclusion over the true one (1: all recovered).
    """
    vp, true, start = (np.asarray(a, np.float64) for a in (vp, true, start))
    return {"error_ratio": _norm(vp - true) / _norm(start - true),
            "contrast": float((vp[inclusion] - start[inclusion]).mean() / (true[inclusion] - start[inclusion]).mean())}


def time_lapse_metrics(dm, true_dm, box):
    """``{"recovered", "correlation", "artefacts", "error"}`` of a time-lapse estimate of vp.

    ``recovered``: the mean of the estimate inside the box over the true change (1: all recovered);
    ``correlation``: Pearson's correlation of the estimate and the true change over the model (1: the same
    pattern; 0 for an estimate that is zero everywhere); ``artefacts``: the root mean square of the estimate
    outside the box over the true change; ``error``: ``||dm - true|| / ||true||``.
    """
    dm, true_dm = np.asarray(dm, np.float64), np.asarray(true_dm, np.float64)
    change = true_dm[box].mean()
    correlation = float(np.corrcoef(dm.ravel(), true_dm.ravel())[0, 1]) if dm.std() > 0 else 0.0
    return {"recovered": float(dm[box].mean() / change), "correlation": correlation,
            "artefacts": float(np.sqrt(np.mean(dm[~box] ** 2)) / abs(change)),
            "error": _norm(dm - true_dm) / _norm(true_dm)}


def agreement(first, second, reference):
    """``||first - second|| / ||reference||``: the difference of two estimates relative to what is recovered."""
    return _norm(np.asarray(first, np.float64) - np.asarray(second, np.float64)) / _norm(reference)


def evaluate(results, models, p):
    """Fill ``results["cases"][case][configuration]["metrics"]`` and ``results["agreement"]`` from ``models``.

    ``models`` maps ``"<case>/<configuration>/<key>"`` to vp arrays (``key``: ``vp`` for ``fwi``, the
    estimates ``dm...`` of a strategy).
    """
    true, monitor, start, inclusion, box = make_models(p["nz"], p["nx"])
    true_dm = monitor["vp"] - true["vp"]
    results["agreement"] = {}
    for case, runs in results["cases"].items():
        reference = true["vp"] - start["vp"] if case == "fwi" else true_dm
        for configuration, run in runs.items():
            run["metrics"] = {key: (fwi_metrics(models[f"{case}/{configuration}/{key}"], true["vp"], start["vp"],
                                                inclusion) if case == "fwi" else
                                    time_lapse_metrics(models[f"{case}/{configuration}/{key}"], true_dm, box))
                              for key in run["keys"]}
        results["agreement"][case] = {}
        for pair, first, second, _ in PAIRS:
            if first in runs and second in runs:
                results["agreement"][case][pair] = {
                    key: agreement(models[f"{case}/{first}/{key}"], models[f"{case}/{second}/{key}"], reference)
                    for key in runs[first]["keys"]}
    return results


def converged(stages):
    """The stages whose misfit did not decrease: ``[(index, stage), ...]`` (empty when all did).

    A stage decreased when its smallest misfit is below its first one.
    """
    return [(i, s) for i, s in enumerate(stages) if not s["misfit"] or min(s["misfit"]) >= s["misfit"][0]]


# ---------------------------------------------------------------------------- running

def environment(configurations):
    """Machine and versions, for the report."""
    import platform

    import scipy

    import PyFWI
    lines = {"PyFWI": PyFWI.__version__, "Python": platform.python_version(), "NumPy": np.__version__,
             "SciPy": scipy.__version__}
    if any(c[2] == "opencl" for c in configurations):
        import pyopencl as cl
        answer = os.environ.get("PYOPENCL_CTX", "0").split(":")
        platform_ = cl.get_platforms()[int(answer[0])]
        device = platform_.get_devices()[int(answer[1]) if len(answer) > 1 else 0]
        lines.update({"OpenCL device": f"{device.name} ({platform_.name}, {platform_.version})",
                      "PyOpenCL": cl.VERSION_TEXT})
    if any(c[2].startswith("cuda") for c in configurations):
        import cupy

        from PyFWI.backends.cuda import CUDABackend
        lines.update({"CUDA device": CUDABackend().description, "CUDA runtime": cupy.cuda.runtime.runtimeGetVersion(),
                      "CUDA driver": cupy.cuda.runtime.driverGetVersion(), "CuPy": cupy.__version__})
    return lines


def run(size="full", cases=CASES, configurations=CONFIGURATIONS, progress=print, save=None):
    """Run the cases in every configuration: ``(results, models)``.

    The variables of PyFWI (``PYFWI_...``) are removed from ``os.environ`` first: the defaults are
    validated; ``inpa`` chooses the backend and the numeric mode of every configuration.

    Parameters
    ----------
    size : {"full", "small"}
        See ``SIZES``.
    cases : sequence of str
        ``"fwi"`` and time-lapse strategies (``STRATEGIES``).
    configurations : sequence
        Entries of ``CONFIGURATIONS``.
    progress : callable
        Receives one line per run.
    save : callable, optional
        ``save(results, models)`` after every case (a long run keeps what it has done).

    Returns
    -------
    results : dict
        ``{"size", "parameters", "configurations", "environment", "cases": {case: {configuration:
        {"time_s", "evaluations", "stages", "keys", "metrics"}}}, "agreement"}`` (JSON-serialisable).
    models : dict
        ``{"<case>/<configuration>/<key>": vp}``.
    """
    for name in [n for n in os.environ if n.startswith(VARIABLE_PREFIX)]:
        progress(f"removing {name}={os.environ.pop(name)} (the defaults are validated)")
    if any(c[2] == "opencl" for c in configurations) and "PYOPENCL_CTX" not in os.environ:
        from tools.make_golden import default_opencl_context
        default_opencl_context()
    p = SIZES[size]
    results = {"size": size, "parameters": p, "source_amplitude": SOURCE_AMPLITUDE,
               "configurations": {c[0]: {"label": c[1], "backend": c[2], "numerics": c[3]} for c in configurations},
               "environment": environment(configurations), "cases": {}}
    models = {}
    for case in cases:
        results["cases"][case] = {}
        for name, _, backend, numerics in configurations:
            tic = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):  # the optimisers print every evaluation
                if case == "fwi":
                    estimates, stages = run_fwi(p, backend, numerics)
                else:
                    estimates, stages = run_time_lapse(p, case, backend, numerics)
            elapsed = time.perf_counter() - tic
            gc.collect()  # the propagators of the case, and their device memory
            for key, values in estimates.items():
                models[f"{case}/{name}/{key}"] = values
            results["cases"][case][name] = {"time_s": elapsed, "stages": stages, "keys": sorted(estimates),
                                            "evaluations": sum(len(s["misfit"]) for s in stages)}
            progress(f"{case:4} {name:14} {elapsed:7.1f} s, {results['cases'][case][name]['evaluations']} evaluations")
        evaluate(results, models, p)
        if save is not None:
            save(results, models)
    return results, models


def save_results(results, models, folder):
    """Write ``results.json`` and ``models.npz`` (float32, compressed) into ``folder``."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(os.path.join(folder, "models.npz"), **models)


def load_results(folder):
    """``(results, models)`` written by ``save_results``."""
    with open(os.path.join(folder, "results.json")) as f:
        results = json.load(f)
    with np.load(os.path.join(folder, "models.npz")) as data:
        return results, {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------- figures

#: Sequential ramp for velocities (one hue, light to dark) and diverging ramp for changes (blue for
#: decreases, a neutral midpoint, orange for increases) of the reference palette.
VELOCITY = LinearSegmentedColormap.from_list("velocity", ["#eef3fb", "#9cc0ec", BLUE, "#123e74"])
CHANGE = LinearSegmentedColormap.from_list("change", [BLUE, "#f1f0ec", ORANGE])

#: Line colour and style of every configuration (identity is never colour alone: the style differs too).
LINES = {"cuda-fixed": (BLUE, "-"), "opencl-fixed": (ORANGE, "--"), "opencl-legacy": (AQUA, ":")}


def _image(ax, values, p, title, cmap, vmin, vmax):
    """One model panel: depth down, distances in metres, no grid."""
    extent = [0, p["dh"] * p["nx"], p["dh"] * p["nz"], 0]
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    return im


def _colorbar(fig, im, ax, label, shrink=0.85):
    bar = fig.colorbar(im, ax=ax, shrink=shrink)
    bar.set_label(label, fontsize=8, color=INK_SECONDARY)
    bar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    bar.outline.set_edgecolor(AXIS)
    return bar


def _style_lines(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _symmetric(values):
    """A symmetric colour limit for a difference: its largest magnitude, at least 1e-3 m/s."""
    return max(float(np.abs(values).max()), 1e-3)


def figure_fwi(results, models, out):
    """``validation_fwi.png``: the true model and the three estimates, their differences, a vp profile."""
    p = results["parameters"]
    true, _, start, _, _ = make_models(p["nz"], p["nx"])
    configurations = list(results["cases"]["fwi"])
    estimate = {c: models[f"fwi/{c}/vp"] for c in configurations}
    vmin, vmax = 2400.0, 3050.0
    fig = plt.figure(figsize=(13, 6.4), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(2, 4)
    top = [fig.add_subplot(grid[0, i]) for i in range(4)]
    im = _image(top[0], true["vp"], p, "True vp", VELOCITY, vmin, vmax)
    for ax, c in zip(top[1:], configurations):
        im = _image(ax, estimate[c], p, results["configurations"][c]["label"], VELOCITY, vmin, vmax)
    for ax in top[1 + len(configurations):]:
        ax.set_visible(False)
    _colorbar(fig, im, top, "vp (m/s)")
    differences = [(f"{SHORT[a]} - {SHORT[b]}", estimate[a] - estimate[b])
                   for _, a, b, _ in PAIRS[:2] if a in estimate and b in estimate]
    for i, (title, values) in enumerate(differences):
        ax = fig.add_subplot(grid[1, i])
        clip = _symmetric(values)
        _colorbar(fig, _image(ax, values, p, title, CHANGE, -clip, clip), ax, "m/s")
    profile = fig.add_subplot(grid[1, 2:])
    _style_lines(profile)
    depth = np.arange(p["nz"]) * p["dh"]
    column = p["nx"] // 2
    profile.plot(depth, true["vp"][:, column], color=INK, linewidth=2, label="true")
    profile.plot(depth, start["vp"][:, column], color=INK_MUTED, linewidth=1.5, label="start")
    for c in configurations:
        colour, style = LINES[c]
        profile.plot(depth, estimate[c][:, column], color=colour, linestyle=style, linewidth=2,
                     label=results["configurations"][c]["label"])
    profile.set_xlabel("depth (m)", fontsize=8, color=INK_SECONDARY)
    profile.set_ylabel("vp (m/s)", fontsize=8, color=INK_SECONDARY)
    profile.set_title(f"vp at x = {column * p['dh']:.0f} m (through the inclusion)", fontsize=9, color=INK)
    profile.legend(fontsize=7, frameon=False)
    return _save(fig, out, "validation_fwi.png")


def figure_time_lapse(results, models, out):
    """``validation_time_lapse.png``: the main estimate of every strategy in every configuration.

    The true change has its own colour scale; the estimates share one (the largest magnitude of the
    main estimates), so that they compare with each other; the differences of CUDA and OpenCL in the
    fixed mode (last column) have one each.
    """
    p = results["parameters"]
    baseline, monitor, *_ = make_models(p["nz"], p["nx"])
    true_dm = monitor["vp"] - baseline["vp"]
    strategies = [s for s in STRATEGIES if s in results["cases"]]
    configurations = list(results["cases"][strategies[0]])
    main = {(s, c): models[f"{s}/{c}/{STRATEGIES[s][2]}"] for s in strategies for c in configurations}
    clip = max(_symmetric(values) for values in main.values())
    columns = len(configurations) + 1
    fig, axes = plt.subplots(len(strategies) + 1, columns, figsize=(2.9 * columns, 2.4 * (len(strategies) + 1)),
                             facecolor=SURFACE, squeeze=False, layout="constrained")
    true_clip = _symmetric(true_dm)  # a box of one value: its title says it
    _image(axes[0, 0], true_dm, p, f"True change: {true_dm[true_dm != 0].mean():+.0f} m/s in the box", CHANGE,
           -true_clip, true_clip)
    for ax in axes[0, 1:]:
        ax.set_visible(False)
    for row, strategy in zip(axes[1:], strategies):
        key = STRATEGIES[strategy][2]
        for ax, c in zip(row, configurations):
            im = _image(ax, main[strategy, c], p, f"{strategy} ({key}): {SHORT[c]}", CHANGE, -clip, clip)
        if "cuda-fixed" in configurations and "opencl-fixed" in configurations:
            values = main[strategy, "cuda-fixed"] - main[strategy, "opencl-fixed"]
            small = _symmetric(values)
            diff = _image(row[-1], values, p, "CUDA - OpenCL fixed", CHANGE, -small, small)
            _colorbar(fig, diff, row[-1], "m/s")
        else:
            row[-1].set_visible(False)
    _colorbar(fig, im, axes[1:, :-1], "change of vp (m/s)", shrink=min(0.85, 2.5 / len(strategies)))
    return _save(fig, out, "validation_time_lapse.png")


def figure_misfit(results, out):
    """``validation_misfit.png``: the misfit of every evaluation of every case, over the first one of its stage.

    The stages and inversions of a case follow each other; the curve goes back to 1 where a stage starts.
    """
    cases = list(results["cases"])
    columns = 3
    rows = -(-len(cases) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.0 * rows), facecolor=SURFACE, squeeze=False,
                             layout="constrained")
    for ax, case in zip(axes.ravel(), cases):
        _style_lines(ax)
        for c, run_ in results["cases"][case].items():
            relative = [np.asarray(s["misfit"]) / s["misfit"][0] for s in run_["stages"] if s["misfit"]]
            relative = np.concatenate(relative) if relative else np.ones(1)
            colour, style = LINES.get(c, (INK_MUTED, "-"))
            ax.semilogy(np.arange(1, relative.size + 1), relative, color=colour, linestyle=style,
                        linewidth=1.6, label=results["configurations"][c]["label"])
        title = "FWI" if case == "fwi" else f"{case}: {STRATEGIES[case][0]}"
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("evaluation", fontsize=8, color=INK_SECONDARY)
    for ax in axes.ravel()[len(cases):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("misfit / first of its stage", fontsize=8, color=INK_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=8, frameon=False)
    return _save(fig, out, "validation_misfit.png")


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def figures(results, models, out):
    """Every figure of the cases that ran: ``[paths]``."""
    paths = []
    if "fwi" in results["cases"]:
        paths.append(figure_fwi(results, models, out))
    if any(s in results["cases"] for s in STRATEGIES):
        paths.append(figure_time_lapse(results, models, out))
    paths.append(figure_misfit(results, out))
    return paths


# ---------------------------------------------------------------------------- report

def _percent(value):
    return f"{100 * value:.1f} %"


def _small(value):
    """A relative difference: a percentage when it is above 0.1 %, scientific notation below."""
    return f"{100 * value:.2f} %" if value >= 1e-3 else f"{value:.1e}"


def metric_difference(results, first, second):
    """The largest difference of a metric between configurations ``first`` and ``second``, over every estimate.

    The metrics are ratios or correlations (``fwi_metrics``, ``time_lapse_metrics``), so the difference
    reads on their own scale: 0.01 is one point of a correlation or one percent of the change recovered.
    """
    differences = [abs(runs[first]["metrics"][key][name] - runs[second]["metrics"][key][name])
                   for runs in results["cases"].values() if first in runs and second in runs
                   for key in runs[first]["metrics"] for name in runs[first]["metrics"][key]]
    return max(differences, default=0.0)


def findings(results):
    """Sentences computed from the results, for the summary of the report."""
    lines = []
    stalled = [(case, c, i, s) for case, runs in results["cases"].items() for c, run_ in runs.items()
               for i, s in converged(run_["stages"])]
    total = sum(len(run_["stages"]) for runs in results["cases"].values() for run_ in runs.values())
    if stalled:
        listed = ", ".join(f"{case} {c} stage {i + 1} ({s['freq']:g} Hz)" for case, c, i, s in stalled)
        lines.append(f"The misfit decreased in {total - len(stalled)} of the {total} stages; it did not in: {listed}.")
    else:
        lines.append(f"The misfit decreased in every stage of every inversion ({total} stages).")
    for pair, first, second, meaning in PAIRS:
        values = [(value, case, key) for case, pairs in results.get("agreement", {}).items()
                  for key, value in pairs.get(pair, {}).items()]
        if values:
            value, case, key = max(values)
            lines.append(f"Largest difference for {meaning}: {_small(value)} of what the inversion recovers "
                         f"({case}, `{key}`); the metrics of the two differ by at most "
                         f"{metric_difference(results, first, second):.3f}.")
    if "fwi" in results["cases"]:
        ratios = ", ".join(f"{results['configurations'][c]['label']} {run_['metrics']['vp']['error_ratio']:.3f}"
                           for c, run_ in results["cases"]["fwi"].items())
        lines.append(f"FWI: model error after the inversion over the error of the starting model: {ratios}.")
    main = [(run_["metrics"][STRATEGIES[s][2]]["correlation"], s, c) for s in STRATEGIES if s in results["cases"]
            for c, run_ in results["cases"][s].items()]
    if main:
        low, high = min(main), max(main)
        lines.append(f"Time-lapse: the correlation of the main estimates with the true change goes from "
                     f"{low[0]:.2f} ({low[1]}, {results['configurations'][low[2]]['label']}) to {high[0]:.2f} "
                     f"({high[1]}, {results['configurations'][high[2]]['label']}).")
    return lines


def render(results, figures_folder):
    """The Markdown report; the figures are linked relative to the report's folder (``figures_folder``)."""
    p = results["parameters"]
    configurations = results["configurations"]
    version = results["environment"].get("PyFWI", "?")
    lines = [f"# Geophysical validation of the CUDA backend (PyFWI {version})", "",
             ("Complete inversions on the GPU of the project, the CUDA backend side by side with the OpenCL backend, "
             "for the review of a geophysicist (roadmap step 4.6). Written by `tools/geophysical_validation.py` from "
             f"`bench/geophysical-{version}/`; the cases, the departures from the examples and the metrics are "
             "described in the docstring of the script."), "",
             "## How to read it", "",
             ("- **FWI** (full-waveform inversion) changes a model of the subsurface, here the P-wave velocity vp, "
             "until the seismograms it predicts match the observed ones. The **misfit** measures the mismatch; each "
             "**evaluation** computes it and its gradient (one forward modelling and one adjoint modelling per "
             "source). **Multi-scale**: the data are low-pass filtered, from the lowest band to the highest, each "
             "band (a **stage**) starting from the result of the previous one."),
             ("- **Time-lapse** FWI recovers what changed between two surveys of the same area (a baseline and a "
             "monitor), here a 20 % increase of vp in a box. PyFWI has eight strategies, which differ by the order "
             "and the starting models of their inversions."),
             "- Every case runs in three **configurations**: "
             + "; ".join(f"**{v['label']}**" for v in configurations.values())
             + ". The `fixed` numeric mode corrects known errors of the arithmetic of PyFWI 0.1.10, which the "
             "`legacy` mode keeps. CUDA against OpenCL in the same mode measures the backend; legacy against fixed "
             "measures the corrections.",
             ("- Differences between configurations are given relative to what the inversion recovers: "
             "`||a - b|| / ||true model - starting model||` for FWI, `||a - b|| / ||true change||` for time-lapse."),
             ("- Two configurations never compute the same bits: their kernels and compilers round differently "
              "(about 1e-6 of a gradient for CUDA against OpenCL in the same mode, `parity` reports). An inversion "
              "amplifies these differences: a slightly different gradient changes the steps of the line search, "
              "sometimes the number of evaluations, and the paths of the two inversions part a little more at "
              "every iteration. What matters for the review is whether the estimates lead to the same "
              "interpretation: the metrics of the tables, and the figures."),
             "", "## Summary", ""]
    lines += [f"- {line}" for line in findings(results)]
    lines += ["", "## Environment", ""]
    lines += [f"- {label}: {value}" for label, value in results["environment"].items()]
    lines += ["", "## The cases", "",
              (f"- Model of Louboutin et al. (2018), {p['nz']} x {p['nx']} cells of {p['dh']:g} m: vp 2500 m/s with an "
              f"inclusion at 3000 m/s (radius {p['nz'] // 10} cells), vs from a Poisson's ratio of 0.25, density from "
              "Gardner's relation. Starting model: vp 2500 m/s everywhere, the true vs and density; vp only is inverted."),
              (f"- {p['ns']} sources, Ricker wavelet of 15 Hz (times {SOURCE_AMPLITUDE:g}), {p['t']:g} s of pressure "
              "recordings, 4th-order finite differences, absorbing layers of 20 cells; L-BFGS with energy "
              f"balancing; bands below {', '.join(f'{f:g}' for f in p['freqs'])} Hz, "
              f"{', '.join(str(i) for i in p['iter'])} iterations."),
              ("- FWI: surface acquisition. Time-lapse: crosswell acquisition (sources in a well on the left, "
              "receivers in a well on the right); the monitor has vp 20 % higher in a box above the inclusion "
              "(as `example/tlfwi_example_crosswell.ipynb`)."), ""]
    if "fwi" in results["cases"]:
        lines += ["## Multi-scale FWI", "", f"![FWI]({figures_folder}/validation_fwi.png)", "",
                  ("Top: the true vp and the estimate of every configuration. Bottom: the differences of two "
                  "estimates (note their colour scales) and a vertical profile through the inclusion."), "",
                  "| Configuration | Error ratio | Contrast recovered | Evaluations | Time (s) |",
                  "| --- | --- | --- | --- | --- |"]
        for c, run_ in results["cases"]["fwi"].items():
            m = run_["metrics"]["vp"]
            lines.append(f"| {configurations[c]['label']} | {m['error_ratio']:.3f} | {_percent(m['contrast'])} | "
                         f"{run_['evaluations']} | {run_['time_s']:.1f} |")
        lines += ["", ("Error ratio: `||estimate - true|| / ||start - true||` (lower is better). Contrast recovered: "
                  "the mean increase of vp inside the inclusion over the true one."), ""]
        lines += _agreement_table(results, "fwi")
    strategies = [s for s in STRATEGIES if s in results["cases"]]
    if strategies:
        lines += ["## Time-lapse strategies", "", f"![Time-lapse]({figures_folder}/validation_time_lapse.png)", "",
                  ("The main estimate of every strategy (the one that uses all its inversions); the estimates share "
                  "one colour scale, the true change (top) and the differences of CUDA and OpenCL in the fixed mode "
                  "(last column) have their own."), "",
                  "| Strategy | What it does |", "| --- | --- |"]
        lines += [f"| `{s}` {STRATEGIES[s][0]} | {STRATEGIES[s][1]} |" for s in strategies]
        lines += ["", ("Every estimate (`*`: the main one). Recovered: the mean of the estimate inside the box over "
                  "the true change. Correlation: with the true change over the model. Artefacts: the root mean "
                  "square outside the box over the true change."), "",
                  "| Strategy | Estimate | Configuration | Recovered | Correlation | Artefacts | Evaluations | Time (s) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in strategies:
            for c, run_ in results["cases"][s].items():
                for key in run_["keys"]:
                    m = run_["metrics"][key]
                    mark = "*" if key == STRATEGIES[s][2] else ""
                    lines.append(f"| `{s}` | `{key}`{mark} | {configurations[c]['label']} | {_percent(m['recovered'])} "
                                 f"| {m['correlation']:.2f} | {_percent(m['artefacts'])} | {run_['evaluations']} | "
                                 f"{run_['time_s']:.1f} |")
        lines.append("")
        for s in strategies:
            lines += _agreement_table(results, s)
    lines += ["## Misfit curves", "", f"![Misfit]({figures_folder}/validation_misfit.png)", "",
              ("The misfit of every evaluation over the first one of its stage, the stages and inversions of a case "
              "one after the other: the curve goes back to 1 where a stage starts. The first and last misfit of "
              "every stage:"), "",
              "| Case | Configuration | Stage | Band (Hz) | First misfit | Last misfit | Evaluations |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for case, runs in results["cases"].items():
        for c, run_ in runs.items():
            for i, s in enumerate(run_["stages"]):
                first, last = (s["misfit"][0], s["misfit"][-1]) if s["misfit"] else (float("nan"),) * 2
                lines.append(f"| {case} | {configurations[c]['label']} | {i + 1} (inversion {s['inversion']}, "
                             f"{s['kind']}) | {s['freq']:g} | {first:.4g} | {last:.4g} | {len(s['misfit'])} |")
    lines += ["", "## For the reviewer", "",
              "- Are the estimates of every configuration geophysically plausible for this model and acquisition?",
              "- Do the CUDA estimates lead to the same interpretation as the OpenCL ones?",
              ("- Does the `fixed` mode change any conclusion against `legacy`? (Gate G3 kept OpenCL on `legacy` "
              "until this review.)"),
              ("- Notes for the review: `dd` inverts the composite data from the starting model, not from the "
              "baseline result as in Asnaashari et al. (2011); `wa` returns its two estimates without the weights of "
              "the average."), "",
              "Review: *(name, date, verdict: to be filled in by the reviewer)*", ""]
    return "\n".join(lines)


def _agreement_table(results, case):
    """The differences of the pairs of configurations for one case (Markdown lines)."""
    pairs = results.get("agreement", {}).get(case, {})
    if not pairs:
        return []
    keys = sorted(next(iter(pairs.values())))
    title = "FWI" if case == "fwi" else f"`{case}`"
    lines = [f"Differences between configurations, {title}:", "",
             "| Estimate | " + " | ".join(pair for pair in pairs) + " |", "| --- |" + " --- |" * len(pairs)]
    for key in keys:
        lines.append(f"| `{key}` | " + " | ".join(_small(pairs[pair][key]) for pair in pairs) + " |")
    meanings = "; ".join(f"`{pair}`: {meaning.split(': ', 1)[1]}" for pair, _, _, meaning in PAIRS if pair in pairs)
    return lines + ["", f"({meanings}.)", ""]


# ---------------------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(SIZES), default="full")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--configurations", nargs="+", choices=[c[0] for c in CONFIGURATIONS],
                        default=[c[0] for c in CONFIGURATIONS])
    parser.add_argument("--out", help="one folder for the results, the figures and the report "
                                      "(default: bench/geophysical-<version>/, docs/cuda-migration/reports/)")
    parser.add_argument("--redraw", action="store_true", help="the figures and the report from saved results")
    args = parser.parse_args(argv)

    import PyFWI
    folder = args.out or RESULTS.format(PyFWI.__version__)
    figures_folder = args.out or FIGURES
    report = os.path.join(args.out, "report.md") if args.out else REPORT.format(PyFWI.__version__)
    if args.redraw:
        results, models = load_results(folder)
    else:
        chosen = [c for c in CONFIGURATIONS if c[0] in args.configurations]
        results, models = run(args.size, args.cases, chosen, progress=lambda line: print(line, flush=True),
                              save=lambda r, m: save_results(r, m, folder))
    for path in figures(results, models, figures_folder):
        print(f"figure: {path}")
    relative = os.path.relpath(figures_folder, os.path.dirname(os.path.abspath(report)))
    with open(report, "w") as f:
        f.write(render(results, relative))
    print(f"results: {folder}\nreport: {report}")
    for line in findings(results):
        print(line)


if __name__ == "__main__":
    main()
