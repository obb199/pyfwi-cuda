"""GPU timeline of a benchmark scenario with Nsight Systems (roadmap step 3.11, test type PRF).

Gate G3 asks for "a timeline without relevant idle gaps in the small scenario": the GPU should be
busy while PyFWI runs, not waiting for the host. This script measures it:

1. it runs ``nsys profile`` on itself (``--target``): a scenario of ``tools/benchmark.py`` is built
   and warmed up, then one forward modelling and one gradient run inside the NVTX ranges
   ``forward`` and ``gradient``; with ``--cuda-graph-trace=node`` the kernels replayed by the CUDA
   graphs of the native loops (step 3.7) are traced one by one;
2. it exports the report to SQLite (``nsys export``) and reads, inside each range, every kernel and
   every copy of the GPU;
3. it reports, per range: its length, the time the GPU was busy (the union of the kernels and
   copies), the idle gaps between them (how many, their median, the largest, those above
   ``SIGNIFICANT_GAP_NS`` with what came before and after), and the time of each kernel with an
   estimate of the memory bandwidth it reached (``ARRAYS_MOVED``).

The bandwidth that gate G3 asks for comes from Nsight Compute, which needs the GPU performance
counters (restricted to root by the driver on the development machine: an action of the project
owner). The estimate here divides the bytes a kernel must move at least, every array it reads or
writes once per cell, by its duration: a lower bound of the traffic, which ignores the absorbing
layer profiles (read at the borders only, step 3.10) and the reuse of the stencils' neighbours in the
caches.

Usage::

    ../envs/dev/bin/python tools/timeline.py                    # small scenario, default backend
    ../envs/dev/bin/python tools/timeline.py --scenario tiny --out /tmp/timeline

Only CUDA activity is traced: run it with the CUDA backend (the default ``auto`` on an NVIDIA GPU
with CuPy). Needs ``nsys`` on the PATH (the CUDA Toolkit, or Nsight Systems).

Used by
-------
Gate G3 (``docs/cuda-migration/reports/timeline-0.3.0.md``). Tested by ``test/test_timeline.py``.
"""
import argparse
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Gaps at least this long (ns) are listed one by one: longer than a kernel launch from the host
#: (a few us), so the GPU waited for the host there.
SIGNIFICANT_GAP_NS = 10_000

#: The NVTX ranges of the target run, in order.
RANGES = ("forward", "gradient")

#: Arrays of one field that each native kernel reads or writes once per cell (elastic.cu), for the
#: bandwidth estimate: update_v reads the 5 fields and 1 / density and writes vx and vz; update_tau
#: reads the 5 fields and the two Lame parameters and writes the 3 stresses; grad_adjoint_v reads the
#: 5 fields, the 5 adjoint fields, the 3 model parameters and the 6 gradients and writes the 2 adjoint
#: velocities and the 6 gradients; Adj_update_tau reads the 5 adjoint fields and 1 / density and writes
#: the 3 adjoint stresses; copy_checkpoint reads and writes the 5 fields.
ARRAYS_MOVED = {"update_v": 8, "update_tau": 10, "grad_adjoint_v": 27, "Adj_update_tau": 9, "copy_checkpoint": 10}


def busy_and_gaps(intervals):
    """The time the GPU was busy, and the idle gaps, from the ``(start, end)`` of its activities.

    Parameters
    ----------
    intervals : list of tuple
        ``(start, end)`` of every kernel and copy, in ns, sorted by start. They may overlap (copies
        run beside kernels on other engines): the busy time is the length of their union.

    Returns
    -------
    busy : int
        Length of the union of the intervals (ns).
    gaps : list of tuple
        ``(length, before, after)`` of every gap between the pieces of the union: ``before`` is the
        index of the interval that ends the piece before the gap, ``after`` of the one that starts
        the piece after it.

    Used by ``analyse``.
    """
    busy, gaps, piece_end, last = 0, [], None, None
    for index, (start, end) in enumerate(intervals):
        if piece_end is None or start > piece_end:
            if piece_end is not None:
                gaps.append((start - piece_end, last, index))
            busy += end - start
            piece_end, last = end, index
        elif end >= piece_end:
            busy += end - piece_end
            piece_end, last = end, index
    return busy, gaps


def analyse(database):
    """Read an Nsight Systems SQLite export and measure each NVTX range of ``RANGES``.

    Parameters
    ----------
    database : str
        Path of the ``.sqlite`` file (``nsys export --type sqlite``).

    Returns
    -------
    dict
        Per range name: ``span_ns``, ``busy_ns``, ``busy_fraction``, ``activities``, ``gaps``
        (count), ``median_gap_ns``, ``largest_gap_ns``, ``significant`` (the gaps of at least
        ``SIGNIFICANT_GAP_NS``: list of ``(length_ns, at_ns, before, after)``, ``at_ns`` from the
        start of the range), ``small_gaps_ns`` (their total below that length) and ``kernels``
        (name -> ``(count, total ns, threads)``: ``threads`` launched per call, on average, which is
        the number of cells rounded up to whole blocks; 0 for copies).
    """
    db = sqlite3.connect(database)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    results = {}
    for start, end, name in db.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        if name not in RANGES:
            continue
        activities = db.execute(
            "SELECT k.start, k.end, s.value, k.gridX * k.blockX * k.gridY * k.blockY * k.gridZ * k.blockZ "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
            "WHERE k.start >= ? AND k.end <= ?", (start, end)).fetchall()
        for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table in tables:
                activities += db.execute(f"SELECT start, end, '{label} ' || bytes || ' B', 0 FROM {table} "
                                         "WHERE start >= ? AND end <= ?", (start, end)).fetchall()
        activities.sort()
        busy, gaps = busy_and_gaps([(a, b) for a, b, _, _ in activities])
        lengths = sorted(length for length, _, _ in gaps)
        significant = [(length, activities[before][1] - start, activities[before][2], activities[after][2])
                       for length, before, after in sorted(gaps, reverse=True) if length >= SIGNIFICANT_GAP_NS]
        kernels = {}
        for a, b, label, threads in activities:
            count, total, all_threads = kernels.get(label, (0, 0, 0))
            kernels[label] = (count + 1, total + b - a, all_threads + threads)
        kernels = {label: (count, total, all_threads // count) for label, (count, total, all_threads) in kernels.items()}
        results[name] = {
            "span_ns": end - start, "busy_ns": busy, "busy_fraction": busy / (end - start),
            "activities": len(activities), "gaps": len(gaps),
            "median_gap_ns": statistics.median(lengths) if lengths else 0, "largest_gap_ns": max(lengths, default=0),
            "significant": significant, "small_gaps_ns": sum(g for g in lengths if g < SIGNIFICANT_GAP_NS),
            "kernels": kernels}
    return results


def render(results, title):
    """Markdown summary of ``analyse``: one table of the ranges, then the large gaps and kernels of each."""
    header = ("| Range | Length (ms) | GPU busy | Gaps | Median gap (us) | Gaps >= 10 us | Their total (ms) | "
              "Smaller gaps total (ms) |")
    lines = [f"## {title}", "", header, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['span_ns'] / 1e6:.2f} | {100 * r['busy_fraction']:.1f} % | {r['gaps']} | "
                     f"{r['median_gap_ns'] / 1e3:.2f} | {len(r['significant'])} | "
                     f"{sum(g[0] for g in r['significant']) / 1e6:.2f} | {r['small_gaps_ns'] / 1e6:.2f} |")
    for name, r in results.items():
        lines += ["", f"### {name}", "", "Gaps of 10 us or more, longest first:", "",
                  "| Gap (us) | At (ms) | Before the gap | After the gap |", "| --- | --- | --- | --- |"]
        lines += [f"| {g / 1e3:.1f} | {at / 1e6:.2f} | {before} | {after} |" for g, at, before, after in r["significant"]]
        lines += ["", "| Kernel or copy | Count | Total (ms) | Mean (us) | Threads | Estimated GB/s |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for label, (count, total, threads) in sorted(r["kernels"].items(), key=lambda item: -item[1][1]):
            lines.append(f"| {label} | {count} | {total / 1e6:.2f} | {total / count / 1e3:.2f} | {threads or ''} | "
                         f"{bandwidth(label, threads, total / count)} |")
    return "\n".join(lines) + "\n"


def bandwidth(label, threads, mean_ns):
    """Estimated bandwidth of a kernel in GB/s, as text (``ARRAYS_MOVED``; empty for other kernels).

    ``threads`` stands for the cells, 4 bytes per array and cell.
    """
    if label not in ARRAYS_MOVED or not mean_ns:
        return ""
    return f"{ARRAYS_MOVED[label] * threads * 4 / mean_ns:.0f}"  # bytes per ns = GB/s


def target(scenario):
    """What runs under Nsight Systems: the scenario, a warm-up, then the two NVTX ranges."""
    sys.path[:0] = [os.path.join(REPO, "test"), REPO]
    import cupyx.profiler

    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process
    from tools import benchmark

    W, true, start = benchmark.build(scenario)
    observed = process.prepare_residual(W.forward_modeling(true, show=False), 1)
    predicted = process.prepare_residual(W.forward_modeling(start, show=False), 1)
    _, adjoint_source = tools.cost_seismic(predicted, observed, fun=tools.CostFunction("l2"))
    W.gradient(adjoint_source, show=False)  # warm-up: compilation, first graphs, GPU clocks
    with cupyx.profiler.time_range("forward"):
        W.forward_modeling(start, show=False)
    with cupyx.profiler.time_range("gradient"):
        W.gradient(adjoint_source, show=False)
    print(f"engine: {benchmark.engine_description(W)}")


def profile(scenario, out):
    """Run ``target`` under Nsight Systems, export to SQLite; returns the path of the database."""
    os.makedirs(out, exist_ok=True)
    report = os.path.join(out, f"timeline-{scenario}")
    subprocess.run(["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--force-overwrite=true",
                    "-o", report, sys.executable, os.path.abspath(__file__), "--target", "--scenario", scenario],
                   check=True, capture_output=True, text=True)
    subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", f"{report}.sqlite",
                    f"{report}.nsys-rep"], check=True, capture_output=True, text=True)
    return f"{report}.sqlite"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="small", help="scenario of tools/benchmark.py")
    parser.add_argument("--out", help="folder of the report and database (default: a temporary folder)")
    parser.add_argument("--target", action="store_true", help=argparse.SUPPRESS)  # the run under nsys
    args = parser.parse_args(argv)
    if args.target:
        target(args.scenario)
        return None
    if shutil.which("nsys") is None:
        raise SystemExit("nsys (Nsight Systems) is not on the PATH")
    out = args.out or tempfile.mkdtemp(prefix="pyfwi-timeline-")
    results = analyse(profile(args.scenario, out))
    print(render(results, f"Timeline of the {args.scenario} scenario"), end="")
    return results


if __name__ == "__main__":
    main()
