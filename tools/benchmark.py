"""Reference benchmark of PyFWI (roadmap step 0.7): forward and gradient times in three scenarios.

What it measures
----------------
For each scenario: the wall-clock time of one forward modelling and of one gradient
(median of ``--repeats`` runs after one warm-up run), the time per internal time step,
the number of OpenCL kernel launches and of host<->device copies (blocking or not) per
time step, what ran (backend, numeric mode, CUDA kernel set and shots per batch:
``engine_description``), and the state of the GPU (temperature, SM clock, power) before and
after, to spot thermal throttling. GeForce cards change their clock with temperature, hence
the warm-up and the median.

Scenarios (from the analysis document)
--------------------------------------
* ``small`` -- the documentation example: 100 x 100 model + 20-point PML, 7 m grid,
  4 shots, 0.6 s (1,050 internal steps; the analysis says 1,051), ``chpr`` = 20;
* ``medium`` -- 400 x 400, 8 shots, 3,000 steps, ``chpr`` = 20. Its checkpoints would
  need about 18 GB of RAM in 0.1.10, so 2 shots are run and the times are scaled to 8
  (times are linear in the number of shots; the scaling is recorded);
* ``large`` -- 600 x 750 (450 thousand cells), 50 shots, 6,000 steps. One shot is run
  and scaled to 50, with ``chpr`` = 5 instead of 20 (20 % of 6,000 steps would need
  12 GB per shot);
* ``tiny`` -- 40 x 40, 2 shots, for the tests of this script only.

Usage
-----
::

    ../envs/ref-0.1.10/bin/python tools/benchmark.py --label ref-0.1.10 --out bench/baseline
    ../envs/dev/bin/python tools/benchmark.py --label dev --out bench/baseline
    ../envs/dev/bin/python tools/benchmark.py --scenarios tiny --repeats 1   # quick check

Each run writes ``<out>/<label>.json`` with the results and the machine description.

Runs happen inside zero guard bands (``test/oob_guard.py``): plain runs of PyFWI 0.1.10
can crash the NVIDIA kernel with ``CL_OUT_OF_RESOURCES`` after a few propagations in one
process (it happened in the large scenario), because its kernels read outside their
buffers. The guard only adds one fill copy per buffer allocation, which is negligible
against the run times. Once step 1.7 pads the buffers in PyFWI, ``--no-guard`` will be
the normal way to run.

Used by
-------
Developers, before and after performance changes and at every gate. Tested by
``test/test_benchmark.py``.
"""
import argparse
import collections
import contextlib
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "test"), REPO]

#: Scenario definitions. nz, nx: model size; dh: grid spacing (m); ns_run: shots actually run;
#: ns_target: shots of the scenario (times are scaled by ns_target / ns_run); t: recording
#: length (s); chpr: checkpoint percentage.
SCENARIOS = {
    "tiny": {"nz": 40, "nx": 40, "dh": 10.0, "ns_run": 2, "ns_target": 2, "t": 0.2, "chpr": 20},
    "small": {"nz": 100, "nx": 100, "dh": 7.0, "ns_run": 4, "ns_target": 4, "t": 0.6, "chpr": 20},
    "medium": {"nz": 400, "nx": 400, "dh": 7.0, "ns_run": 2, "ns_target": 8, "t": 1.713, "chpr": 20},
    "large": {"nz": 600, "nx": 750, "dh": 7.0, "ns_run": 1, "ns_target": 50, "t": 3.427, "chpr": 5},
}


@contextlib.contextmanager
def count_opencl_calls():
    """Count kernel launches and buffer copies made through PyOpenCL while the block runs.

    Kernel launches are counted by wrapping ``pyopencl.Kernel.__call__`` on the class,
    so every kernel object is seen, including those PyFWI retrieves once and keeps as
    attributes (``prg.injSrc = prg_surf.injSrc``), and ``pyopencl.enqueue_nd_range_kernel``,
    which the fork calls for kernels whose arguments are set beforehand (step 1.3; calling a
    kernel object does not go through that module function, so nothing is counted twice).
    Copies are counted by wrapping ``pyopencl.enqueue_copy``, which PyFWI calls through the
    module. Yields a
    ``collections.Counter`` with the keys ``launches``, ``launch:<kernel name>``,
    ``copies`` and ``blocking_copies`` (a copy to or from host memory blocks unless
    ``is_blocking=False``; a copy between two device buffers never blocks).
    """
    import pyopencl as cl

    counts = collections.Counter()
    original_call, original_copy = cl.Kernel.__call__, cl.enqueue_copy
    original_enqueue = cl.enqueue_nd_range_kernel

    def counting_call(kernel, *args, **kwargs):
        counts["launches"] += 1
        counts[f"launch:{kernel.function_name}"] += 1
        return original_call(kernel, *args, **kwargs)

    def counting_enqueue(queue, kernel, *args, **kwargs):
        counts["launches"] += 1
        counts[f"launch:{kernel.function_name}"] += 1
        return original_enqueue(queue, kernel, *args, **kwargs)

    def counting_copy(queue, dest, src, **kwargs):
        counts["copies"] += 1
        # A copy between two device buffers never makes the host wait, whatever is_blocking says.
        on_device = isinstance(dest, cl.MemoryObjectHolder) and isinstance(src, cl.MemoryObjectHolder)
        counts["blocking_copies"] += bool(kwargs.get("is_blocking", True)) and not on_device
        return original_copy(queue, dest, src, **kwargs)

    cl.Kernel.__call__, cl.enqueue_copy, cl.enqueue_nd_range_kernel = counting_call, counting_copy, counting_enqueue
    try:
        yield counts
    finally:
        cl.Kernel.__call__, cl.enqueue_copy, cl.enqueue_nd_range_kernel = original_call, original_copy, original_enqueue


def gpu_state():
    """Temperature (°C), SM clock (MHz) and power (W) of GPU 0 from nvidia-smi, or {} without it."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             check=True).stdout.split("\n")[0]
        temperature, clock, power = (float(v) for v in out.split(","))
        return {"temperature_c": temperature, "sm_clock_mhz": clock, "power_w": power}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def build(scenario):
    """PyFWI inputs of a scenario: ``(propagator, true model, starting model)``."""
    import PyFWI.acquisition as acq
    import PyFWI.wave_propagation as wave
    from tools.check_env import inclusion_model

    s = SCENARIOS[scenario]
    inpa = {"ns": s["ns_run"], "sdo": 4, "fdom": 15, "dh": s["dh"], "dt": 0.004, "acq_type": 1, "t": s["t"],
            "npml": 20, "pmlR": 1e-5, "pml_dir": 2, "rec_dis": s["dh"]}
    src_loc, rec_loc, _, n_well_rec = acq.acq_parameters(s["ns_run"], s["dh"], s["dh"] * s["nx"], s["dh"] * s["nz"],
                                                         s["dh"], 4, acq_type=1)
    src_loc[:, 1] -= s["dh"]  # one row above the receivers (no race condition)
    src = acq.Source(src_loc, s["dh"], inpa["dt"])
    src.Ricker(inpa["fdom"])
    W = wave.WavePropagator(inpa, src, rec_loc, (s["nz"], s["nx"]), components=0, n_well_rec=n_well_rec,
                            chpr=s["chpr"])
    return W, inclusion_model(s["nz"], s["nx"]), inclusion_model(s["nz"], s["nx"], smooth=1)


def run_once(scenario, guard=True):
    """One forward + gradient; returns ``(forward s, gradient s, forward counts, gradient counts, propagator)``.

    ``guard=True`` runs inside zero guard bands (see the module docstring).
    """
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    from oob_guard import zero_guard_bands

    with zero_guard_bands() if guard else contextlib.nullcontext():
        return _forward_and_gradient(scenario, tools, process)


def _forward_and_gradient(scenario, tools, process):
    """Body of ``run_once``: build the scenario, time one forward and one gradient with call counts."""
    W, true, start = build(scenario)
    observed = process.prepare_residual(W.forward_modeling(true, show=False), 1)
    with count_opencl_calls() as forward_counts:
        t0 = time.perf_counter()
        predicted = process.prepare_residual(W.forward_modeling(start, show=False), 1)
        forward_seconds = time.perf_counter() - t0
    _, adjoint_source = tools.cost_seismic(predicted, observed, fun=tools.CostFunction("l2"))
    with count_opencl_calls() as gradient_counts:
        t0 = time.perf_counter()
        W.gradient(adjoint_source, show=False)
        gradient_seconds = time.perf_counter() - t0
    return forward_seconds, gradient_seconds, forward_counts, gradient_counts, W


def engine_description(W):
    """What ran a propagator ``W``, for the result file (the default backend is ``auto`` since step 3.11).

    Returns
    -------
    dict
        ``backend`` (``Selection.describe`` of ``PyFWI.backends.selection``), ``numerics``,
        ``cuda_kernels`` (``"native"``, ``"parity"`` or None on OpenCL) and ``batch`` (shots run at
        once, step 3.9). PyFWI 0.1.10 has none of these: OpenCL, the legacy arithmetic, one shot at a
        time.
    """
    attributes = vars(W)  # not getattr: WavePropagator forwards unknown names to its engine
    selection, engine = attributes.get("backend_selection"), attributes.get("_engine")
    return {"backend": selection.describe() if selection is not None else "opencl (PyFWI 0.1.10)",
            "numerics": attributes.get("numerics", "legacy"), "cuda_kernels": attributes.get("cuda_kernels"),
            "batch": getattr(engine, "batch", 1)}


def run_scenario(scenario, repeats, guard=True):
    """Warm-up run, then ``repeats`` measured runs; returns the scenario's results."""
    s = SCENARIOS[scenario]
    run_once(scenario, guard)  # warm-up: compilation, caches, GPU clocks
    before = gpu_state()
    forwards, gradients = [], []
    for _ in range(repeats):
        f, g, fc, gc, W = run_once(scenario, guard)
        forwards.append(f)
        gradients.append(g)
    after = gpu_state()
    scale = s["ns_target"] / s["ns_run"]
    steps = int(W.nt) * s["ns_run"]  # internal time steps of one run (all shots)
    return {
        "definition": s, "internal_steps_per_shot": int(W.nt), "scaled_to_shots": s["ns_target"], "scale": scale,
        "forward_s": {"median": statistics.median(forwards) * scale, "runs": forwards},
        "gradient_s": {"median": statistics.median(gradients) * scale, "runs": gradients},
        "cv_forward": statistics.pstdev(forwards) / statistics.mean(forwards),
        "cv_gradient": statistics.pstdev(gradients) / statistics.mean(gradients),
        "host_ms_per_step": {"forward": 1e3 * statistics.median(forwards) / steps,
                             "gradient": 1e3 * statistics.median(gradients) / steps},
        "per_step": {"forward_launches": fc["launches"] / steps, "gradient_launches": gc["launches"] / steps,
                     "forward_copies": fc["copies"] / steps, "gradient_copies": gc["copies"] / steps,
                     "blocking_copies": (fc["blocking_copies"] + gc["blocking_copies"]) / steps},
        "launches_by_kernel": {k[7:]: v for k, v in (fc + gc).items() if k.startswith("launch:")},
        "engine": engine_description(W),
        "gpu_before": before, "gpu_after": after, "guard_bands": guard,
    }


def machine():
    """Description of the machine and of the software, stored with the results."""
    import pyopencl as cl

    import PyFWI
    answer = os.environ["PYOPENCL_CTX"].split(":")
    plat = cl.get_platforms()[int(answer[0])]
    dev = plat.get_devices()[int(answer[1]) if len(answer) > 1 else 0]
    return {"python": platform.python_version(), "numpy": np.__version__, "pyopencl": cl.VERSION_TEXT,
            "pyfwi": PyFWI.__version__, "pyfwi_path": os.path.dirname(os.path.abspath(PyFWI.__file__)),
            "opencl_device": dev.name, "driver": dev.driver_version, "cpu": platform.processor() or platform.machine()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="*", default=["small", "medium", "large"], choices=sorted(SCENARIOS))
    parser.add_argument("--repeats", type=int, default=5, help="measured runs per scenario (median reported)")
    parser.add_argument("--label", default="run", help="name of the result file")
    parser.add_argument("--out", help="folder of the result file (default: print only)")
    parser.add_argument("--no-guard", action="store_true", help="run without guard bands (after step 1.7)")
    args = parser.parse_args(argv)
    if "PYOPENCL_CTX" not in os.environ:
        from tools.make_golden import default_opencl_context
        default_opencl_context()
    import warnings
    warnings.filterwarnings("ignore")  # PyOpenCL's RepeatedKernelRetrieval warning on every launch of 0.1.10

    results = {"label": args.label, "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "machine": machine(), "repeats": args.repeats, "scenarios": {}}
    for scenario in args.scenarios:
        r = run_scenario(scenario, args.repeats, guard=not args.no_guard)
        results["scenarios"][scenario] = r
        print(f"{scenario:7s} forward {r['forward_s']['median']:8.2f} s  gradient {r['gradient_s']['median']:8.2f} s  "
              f"(x{r['scale']:g} shots)  host {r['host_ms_per_step']['forward']:.3f}/{r['host_ms_per_step']['gradient']:.3f} "
              f"ms/step  launches/step {r['per_step']['forward_launches']:.2f}/{r['per_step']['gradient_launches']:.2f}  "
              f"blocking copies/step {r['per_step']['blocking_copies']:.3f}  CV {r['cv_forward']:.3f}/{r['cv_gradient']:.3f}",
              flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.label}.json"), "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
            f.write("\n")
    return results


if __name__ == "__main__":
    main()
