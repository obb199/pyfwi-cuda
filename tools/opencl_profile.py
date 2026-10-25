"""Where the time goes in the OpenCL core: GPU time per kernel against wall-clock time (roadmap step 0.8).

The analysis estimated that more than 90 % of a small run is host overhead (Python,
kernel re-creation, blocking copies) rather than GPU work. This script measures it.

How
---
While ``opencl_profiling()`` is active, every ``pyopencl.CommandQueue`` is created with
``PROFILING_ENABLE`` and every kernel launch keeps its event. After the run, each event
gives the start and end time of the kernel *on the device*. Summing them gives the GPU
busy time; the rest of the wall-clock time is spent on the host or waiting for copies.
(Nsight Systems does not trace OpenCL; for the CUDA backend of Phase 2 use
``tools/profile.sh``.)

Usage::

    ../envs/ref-0.1.10/bin/python tools/opencl_profile.py --scenario small
    ../envs/dev/bin/python tools/opencl_profile.py --scenario medium --json result.json

Used by
-------
Developers at every gate. Tested by ``test/test_profiling.py``.
"""
import argparse
import collections
import contextlib
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "test"), REPO]


@contextlib.contextmanager
def opencl_profiling():
    """Enable OpenCL profiling on every queue created inside the block and record kernel events.

    Yields a list that receives ``(kernel name, pyopencl.Event)`` for every launch.
    Profiling adds a little overhead per launch; use it to see *where* the time goes,
    and ``tools/benchmark.py`` to measure *how much* time.
    """
    import pyopencl as cl

    events = []
    original_queue, original_call = cl.CommandQueue, cl.Kernel.__call__

    def profiling_queue(context, device=None, properties=None):
        flags = (properties or 0) | cl.command_queue_properties.PROFILING_ENABLE
        return original_queue(context, device, flags)

    def recording_call(kernel, *args, **kwargs):
        event = original_call(kernel, *args, **kwargs)
        events.append((kernel.function_name, event))
        return event

    cl.CommandQueue, cl.Kernel.__call__ = profiling_queue, recording_call
    try:
        yield events
    finally:
        cl.CommandQueue, cl.Kernel.__call__ = original_queue, original_call


def profile_scenario(scenario):
    """Run one forward + gradient of a ``tools/benchmark.py`` scenario with OpenCL profiling.

    Returns
    -------
    dict
        ``wall_s`` (wall-clock time of forward + gradient), ``gpu_s`` (sum of kernel times on
        the device), ``gpu_fraction`` (``gpu_s / wall_s``), ``launches``, ``mean_kernel_us`` and
        ``kernels``: per kernel name, its number of launches and total and mean device time.
    """
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    from tools.benchmark import build

    with opencl_profiling() as events:
        W, true, start = build(scenario)
        observed = process.prepare_residual(W.forward_modeling(true, show=False), 1)
        events.clear()  # measure the second forward and the gradient only
        t0 = time.perf_counter()
        predicted = process.prepare_residual(W.forward_modeling(start, show=False), 1)
        _, adjoint_source = tools.cost_seismic(predicted, observed, fun=tools.CostFunction("l2"))
        W.gradient(adjoint_source, show=False)
        W.queue.finish()
        wall = time.perf_counter() - t0
    per_kernel = collections.defaultdict(lambda: [0, 0.0])
    for name, event in events:
        per_kernel[name][0] += 1
        per_kernel[name][1] += (event.profile.end - event.profile.start) * 1e-9  # nanoseconds -> seconds
    gpu = sum(total for _, total in per_kernel.values())
    return {"scenario": scenario, "wall_s": wall, "gpu_s": gpu, "gpu_fraction": gpu / wall, "launches": len(events),
            "mean_kernel_us": 1e6 * gpu / max(len(events), 1),
            "kernels": {name: {"launches": n, "total_s": total, "mean_us": 1e6 * total / n}
                        for name, (n, total) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="small", help="a scenario of tools/benchmark.py")
    parser.add_argument("--json", help="also write the result to this file")
    args = parser.parse_args(argv)
    if "PYOPENCL_CTX" not in os.environ:
        from tools.make_golden import default_opencl_context
        default_opencl_context()
    import warnings
    warnings.filterwarnings("ignore")
    result = profile_scenario(args.scenario)
    print(f"{args.scenario}: wall {result['wall_s']:.2f} s, GPU busy {result['gpu_s']:.2f} s "
          f"({100 * result['gpu_fraction']:.1f} %), {result['launches']} launches, "
          f"mean kernel {result['mean_kernel_us']:.1f} us")
    for name, k in result["kernels"].items():
        print(f"  {name:16s} {k['launches']:7d} launches  {k['total_s']:8.3f} s  {k['mean_us']:8.1f} us/launch")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return result


if __name__ == "__main__":
    main()
