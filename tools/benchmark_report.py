"""Render benchmark result files (``tools/benchmark.py``) as a Markdown comparison table.

The first file given is the reference: every other run gets a speed-up column
relative to it (``> 1`` means faster). One table row per scenario and file.

Usage::

    ../envs/dev/bin/python tools/benchmark_report.py bench/baseline/ref-0.1.10-pyopencl-2022.json \\
        bench/baseline/dev-pyopencl-2026.json > docs/cuda-migration/reports/benchmark-baseline.md

Used by
-------
Developers at every gate (G0 baseline, then each phase). Tested by
``test/test_benchmark.py``.
"""
import argparse
import json


def load(paths):
    """Read result files; returns a list of dicts in the given order."""
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def engine(run):
    """What ran, from the first scenario of ``run`` (``engine_description`` of ``tools/benchmark.py``).

    For example ``cuda device 0 (auto, the default), fixed, native kernels, batches of 4``; files
    written before the engine was recorded (step 3.11) give ``not recorded``.
    """
    described = next(iter(run["scenarios"].values()), {}).get("engine")
    if described is None:
        return "not recorded"
    text = f"{described['backend']}, {described['numerics']}"
    if described.get("cuda_kernels"):
        text += f", {described['cuda_kernels']} kernels"
    batches = {s.get("engine", {}).get("batch", 1) for s in run["scenarios"].values()}
    if batches != {1}:
        text += ", batches of " + "/".join(str(b) for b in sorted(batches))
    return text


def render(runs):
    """Markdown text: a description of each run and one comparison table.

    Parameters
    ----------
    runs : list of dict
        Results of ``tools/benchmark.py``; ``runs[0]`` is the reference for the speed-ups.

    Returns
    -------
    str
    """
    lines = ["| Run | Device | PyFWI | Engine | PyOpenCL | NumPy | Date |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        m = run["machine"]
        lines.append(f"| {run['label']} | {m['opencl_device']} (driver {m['driver']}) | {m['pyfwi']} | "
                     f"{engine(run)} | {m['pyopencl']} | {m['numpy']} | {run['date']} |")
    note = ("Times are medians of the measured runs, scaled to the number of shots of the scenario. "
            "Speed-up is relative to the first run.")
    header = ("| Scenario | Run | Forward (s) | Gradient (s) | Speed-up fwd / grad | Host ms per step fwd / grad | "
              "Launches per step fwd / grad | Blocking copies per step | CV fwd / grad | GPU °C before / after |")
    lines += ["", note, "", header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    reference = runs[0]["scenarios"]
    order = {name: i for i, name in enumerate(("tiny", "small", "medium", "large"))}  # by size, unknown names last
    scenarios = sorted((s for s in reference if all(s in r["scenarios"] for r in runs)),
                       key=lambda s: (order.get(s, len(order)), s))
    for scenario in scenarios:
        for run in runs:
            r = run["scenarios"][scenario]
            ref = reference[scenario]
            fwd, grad = r["forward_s"]["median"], r["gradient_s"]["median"]
            speed = f"{ref['forward_s']['median'] / fwd:.2f} / {ref['gradient_s']['median'] / grad:.2f}"
            temps = f"{r['gpu_before'].get('temperature_c', '?')} / {r['gpu_after'].get('temperature_c', '?')}"
            scale = f" (scaled x{r['scale']:g} to {r['scaled_to_shots']} shots)" if r["scale"] != 1 else ""
            lines.append(
                f"| {scenario}{scale} | {run['label']} | {fwd:.2f} | {grad:.2f} | {speed} | "
                f"{r['host_ms_per_step']['forward']:.3f} / {r['host_ms_per_step']['gradient']:.3f} | "
                f"{r['per_step']['forward_launches']:.2f} / {r['per_step']['gradient_launches']:.2f} | "
                f"{r['per_step']['blocking_copies']:.2f} | {r['cv_forward']:.3f} / {r['cv_gradient']:.3f} | {temps} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="result files; the first is the reference")
    args = parser.parse_args(argv)
    print(render(load(args.files)), end="")


if __name__ == "__main__":
    main()
