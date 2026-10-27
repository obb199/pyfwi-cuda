"""CUDA against OpenCL on the GPU of the project (roadmap step 6): the measurements, the figures and the numbers.

The final document of the project, ``docs/cuda-migration/cuda-vs-opencl.md``, shows what the CUDA backend
gains over the OpenCL backend on the GeForce RTX 3060 of the development machine. Everything it shows comes
from result files of the repository, which this script writes and reads:

* ``--measure`` runs one session on the GPU (about 25 minutes) and writes ``bench/cuda-vs-opencl/``:

  - ``benchmark-<backend>-<mode>.json``: ``tools/benchmark.py`` (small, medium and large scenarios, 5 runs
    after a warm-up) for OpenCL and CUDA, each in the ``legacy`` and in the ``fixed`` numeric mode, so that
    the two backends compare like for like; the default of each backend is one of the four (OpenCL
    ``legacy``, CUDA ``fixed``);
  - ``busy-<backend>-<mode>-<scenario>.json``: the share of one forward modelling and one gradient during
    which the GPU works, small and medium scenarios: ``tools/opencl_profile.py`` for OpenCL (the time of its
    kernels over the wall-clock time) and ``tools/timeline.py`` for CUDA (the union of its kernels and copies
    over the time of the ranges, Nsight Systems).

* without arguments it draws the figures of the document into ``docs/cuda-migration/cuda-vs-opencl/`` from
  those files and from ``bench/release-0.4.0/`` (the GPU memory of the features of Phase 4), without a GPU.

``facts()`` computes every number the document cites, formatted as the document writes it;
``test/test_cuda_vs_opencl.py`` checks that every number with a unit in the document is one of them.

Usage::

    ../envs/dev/bin/python tools/cuda_vs_opencl.py --measure   # the session (GPU), then the figures
    ../envs/dev/bin/python tools/cuda_vs_opencl.py             # the figures from the stored results

Used by
-------
Step 6 of the roadmap. Tested by ``test/test_cuda_vs_opencl.py``.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # after choosing the backend

from tools.plot_overview import AXIS, BLUE, GRID, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE

#: The stored results of the session, and the folder of the figures.
RESULTS = os.path.join(REPO, "bench", "cuda-vs-opencl")
FIGURES = os.path.join(REPO, "docs", "cuda-migration", "cuda-vs-opencl")
#: The release whose runs give the GPU memory of the optional features (CUDA only).
RELEASE = os.path.join(REPO, "bench", "release-0.4.0")
#: The complete inversions on both backends (step 4.6, ``tools/geophysical_validation.py``).
INVERSIONS = os.path.join(REPO, "bench", "geophysical-0.4.0.dev0", "results.json")

BACKENDS = ("opencl", "cuda")
MODES = ("legacy", "fixed")
#: The default numeric mode of each backend.
DEFAULT_MODE = {"opencl": "legacy", "cuda": "fixed"}
SCENARIOS = ("small", "medium", "large")
#: The scenarios whose GPU occupancy is measured (the large one adds nothing: it is as busy as the medium one).
BUSY_SCENARIOS = ("small", "medium")
KINDS = (("forward_s", "Forward modelling"), ("gradient_s", "Gradient"))
NAMES = {"opencl": "OpenCL", "cuda": "CUDA"}


# ---------------------------------------------------------------------------- the session

def _environment(backend, mode):
    """The environment of a run: every ``PYFWI_`` variable removed, then the backend and the mode."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYFWI_")}
    env.update(PYFWI_BACKEND="opencl" if backend == "opencl" else "cuda:strict", PYFWI_NUMERICS=mode)
    return env


def measure(python=sys.executable, progress=print):
    """Run the session and write its files into ``RESULTS`` (see the module docstring)."""
    os.makedirs(RESULTS, exist_ok=True)
    for backend in BACKENDS:
        for mode in MODES:
            label = f"benchmark-{backend}-{mode}"
            progress(f"{label} ...")
            subprocess.run([python, os.path.join(REPO, "tools", "benchmark.py"), "--no-guard", "--label", label,
                            "--out", RESULTS], env=_environment(backend, mode), cwd=REPO, check=True)
    for backend in BACKENDS:
        for mode in MODES:
            for scenario in BUSY_SCENARIOS:
                path = os.path.join(RESULTS, f"busy-{backend}-{mode}-{scenario}.json")
                progress(f"{os.path.basename(path)} ...")
                env = _environment(backend, mode)
                if backend == "opencl":
                    subprocess.run([python, os.path.join(REPO, "tools", "opencl_profile.py"), "--scenario", scenario,
                                    "--json", path], env=env, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
                    with open(path) as f:
                        profile = json.load(f)
                    busy = {"method": "opencl_profile", "wall_s": profile["wall_s"], "gpu_s": profile["gpu_s"],
                            "busy_fraction": profile["gpu_fraction"]}
                else:
                    busy = _cuda_busy(python, scenario, env)
                busy.update(backend=backend, mode=mode, scenario=scenario)
                with open(path, "w") as f:
                    json.dump(busy, f, indent=1, sort_keys=True)
                    f.write("\n")


def _cuda_busy(python, scenario, env):
    """The GPU occupancy of one forward modelling and one gradient on CUDA, from ``tools/timeline.py``."""
    import tempfile

    from tools import timeline
    folder = tempfile.mkdtemp(prefix="pyfwi-cuda-vs-opencl-")
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)  # the run under nsys inherits the environment of this process
    try:
        ranges = timeline.analyse(timeline.profile(scenario, folder))
    finally:
        os.environ.clear()
        os.environ.update(previous)
    span = sum(r["span_ns"] for r in ranges.values())
    busy = sum(r["busy_ns"] for r in ranges.values())
    return {"method": "timeline", "wall_s": span * 1e-9, "gpu_s": busy * 1e-9, "busy_fraction": busy / span,
            "ranges": {name: {"span_ns": r["span_ns"], "busy_ns": r["busy_ns"]} for name, r in ranges.items()}}


# ---------------------------------------------------------------------------- reading the results

def load(folder=RESULTS):
    """``(benchmarks, busy)``: ``{(backend, mode): scenarios of the benchmark file}`` and
    ``{(backend, mode, scenario): busy fraction}``."""
    benchmarks, busy = {}, {}
    for backend in BACKENDS:
        for mode in MODES:
            with open(os.path.join(folder, f"benchmark-{backend}-{mode}.json")) as f:
                benchmarks[backend, mode] = json.load(f)["scenarios"]
            for scenario in BUSY_SCENARIOS:
                with open(os.path.join(folder, f"busy-{backend}-{mode}-{scenario}.json")) as f:
                    busy[backend, mode, scenario] = json.load(f)["busy_fraction"]
    return benchmarks, busy


def release_memory(folder=RELEASE):
    """``{configuration: {scenario: peak GPU memory in MB}}`` of the CUDA runs of release 0.4.0."""
    labels = {"default": "fork-0.4.0-cuda-fixed", "boundary saving": "fork-0.4.0-cuda-boundary-saving",
              "FP16": "fork-0.4.0-cuda-fp16", "both": "fork-0.4.0-cuda-boundary-saving-fp16"}
    memory = {}
    for name, label in labels.items():
        with open(os.path.join(folder, f"{label}.json")) as f:
            memory[name] = {s: r["device_peak_mb"] for s, r in json.load(f)["scenarios"].items()}
    return memory


def time_of(benchmarks, backend, mode, scenario, kind):
    """The median time (s) of one kind (``forward_s``, ``gradient_s``), scaled to the shots of the scenario."""
    return benchmarks[backend, mode][scenario][kind]["median"]


def speedup(benchmarks, scenario, kind, opencl_mode, cuda_mode):
    """How many times CUDA is faster than OpenCL: OpenCL time over CUDA time."""
    return time_of(benchmarks, "opencl", opencl_mode, scenario, kind) / time_of(benchmarks, "cuda", cuda_mode,
                                                                                 scenario, kind)


def _times(value):
    """A speed-up as the document writes it: ``12×``, ``3.4×``, ``1.25×``."""
    return f"{value:.0f}×" if value >= 20 else f"{value:.1f}×" if value >= 2 else f"{value:.2f}×"


def _seconds(value):
    """A time as the document writes it: ``0.018 s``, ``0.41 s``, ``17.9 s``, ``529 s``."""
    if value < 0.1:
        return f"{value:.3f} s"
    return f"{value:.0f} s" if value >= 100 else f"{value:.1f} s" if value >= 10 else f"{value:.2f} s"


def facts(benchmarks=None, busy=None, memory=None):
    """Every number the document cites, ``{name: text}``, formatted as the document writes it.

    Names: ``speedup-<opencl mode>-<cuda mode>-<scenario>-<kind>`` (``kind``: ``forward`` or ``gradient``;
    the like-for-like comparisons and ``speedup-legacy-fixed-...``, the defaults), ``time-<backend>-<mode>-
    <scenario>-<kind>``, ``host-<backend>-<mode>-<scenario>-<kind>`` (milliseconds of the host per time step),
    ``commands-opencl-<mode>-<kind>`` and ``copies-opencl-<kind>`` (OpenCL calls per time step, the same in
    every scenario but for the gradient of the large one), ``busy-<backend>-<mode>-<scenario>``,
    ``memory-<configuration>-<scenario>`` (MB or GB), ``memory-ratio-<configuration>-<scenario>`` (how many
    times less than the default) and ``inversion-backend`` (the largest difference of the estimates of CUDA
    and OpenCL in the same mode after complete inversions, over what the inversion recovers).
    """
    if benchmarks is None or busy is None:
        benchmarks, busy = load()
    memory = release_memory() if memory is None else memory
    numbers = {}
    for scenario in SCENARIOS:
        for kind, _ in KINDS:
            short = kind[:-2]
            for opencl_mode, cuda_mode in (("legacy", "legacy"), ("fixed", "fixed"), ("legacy", "fixed")):
                numbers[f"speedup-{opencl_mode}-{cuda_mode}-{scenario}-{short}"] = _times(
                    speedup(benchmarks, scenario, kind, opencl_mode, cuda_mode))
            for backend in BACKENDS:
                for mode in MODES:
                    numbers[f"time-{backend}-{mode}-{scenario}-{short}"] = _seconds(
                        time_of(benchmarks, backend, mode, scenario, kind))
                    host = benchmarks[backend, mode][scenario]["host_ms_per_step"][short]
                    numbers[f"host-{backend}-{mode}-{scenario}-{short}"] = f"{host:.3f} ms"
    for mode in MODES:
        steps = benchmarks["opencl", mode]["small"]["per_step"]
        for short in ("forward", "gradient"):
            numbers[f"commands-opencl-{mode}-{short}"] = f"{steps[f'{short}_launches']:.1f} commands"
            numbers[f"copies-opencl-{short}"] = f"{steps[f'{short}_copies']:.1f} copies"
    for (backend, mode, scenario), value in busy.items():
        numbers[f"busy-{backend}-{mode}-{scenario}"] = f"{100 * value:.0f} %"
    for name, scenarios in memory.items():
        for scenario, mb in scenarios.items():
            key = name.replace(" ", "-")
            numbers[f"memory-{key}-{scenario}"] = f"{mb / 1024:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"
            if name != "default":
                numbers[f"memory-ratio-{key}-{scenario}"] = _times(memory["default"][scenario] / mb)
    with open(INVERSIONS) as f:
        agreement = json.load(f)["agreement"]
    numbers["inversion-backend"] = f"{100 * max(v for pairs in agreement.values() for v in pairs['backend'].values()):.1f} %"
    return numbers


# ---------------------------------------------------------------------------- figures

def _style(ax, grid_axis="y"):
    """Recessive chrome: solid hairline grid on one axis, no top and right spines."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _speedup_bars(ax, values, labels, colour, title):
    """Horizontal bars of speed-ups, one per scenario and kind, labelled with their value."""
    _style(ax, grid_axis="x")
    y = range(len(values))
    ax.barh(list(y), values, color=colour, edgecolor=SURFACE, linewidth=1.5, height=0.62)
    for i, value in zip(y, values):
        ax.text(value, i, f"  {_times(value)}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels, color=INK)
    ax.set_ylim(len(values) - 0.5, -0.5)  # the first label on top
    ax.axvline(1, color=INK_MUTED, linewidth=1)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def _labels():
    return [f"{scenario.capitalize()}, {name.lower()}" for scenario in SCENARIOS for _, name in KINDS]


def figure_like_for_like(benchmarks, out):
    """``like_for_like.png``: how many times CUDA is faster than OpenCL in the same numeric mode."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, mode in zip(axes, MODES):
        values = [speedup(benchmarks, scenario, kind, mode, mode) for scenario in SCENARIOS for kind, _ in KINDS]
        _speedup_bars(ax, values, _labels(), BLUE if mode == "fixed" else INK_MUTED,
                      f"Both backends in the {mode} mode")
        ax.set_xlabel("times faster on CUDA (1 = as fast as OpenCL)", color=INK_SECONDARY, fontsize=9)
    axes[1].set_yticklabels([])
    fig.suptitle("Same GPU, same mathematics: how many times faster CUDA is than OpenCL", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "like_for_like.png")


def figure_defaults(benchmarks, out):
    """``defaults.png``: the time of the default of each backend (OpenCL legacy, CUDA fixed), log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    width = 0.36
    for ax, (kind, name) in zip(axes, KINDS):
        _style(ax)
        for b, (backend, colour) in enumerate((("opencl", INK_MUTED), ("cuda", BLUE))):
            mode = DEFAULT_MODE[backend]
            for s, scenario in enumerate(SCENARIOS):
                value = time_of(benchmarks, backend, mode, scenario, kind)
                x = s + (b - 0.5) * width
                ax.bar(x, value, width, color=colour, edgecolor=SURFACE, linewidth=1.5,
                       label=f"{NAMES[backend]} ({mode}, its default)" if s == 0 else None)
                ax.text(x, value * 1.15, _seconds(value), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(SCENARIOS)), [f"{s.capitalize()} scenario" for s in SCENARIOS], color=INK)
        ax.set_ylabel("seconds (log scale; lower is better)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_ylim(top=ax.get_ylim()[1] * 3)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    fig.suptitle("What a user gets by default on this GPU", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "defaults.png")


def figure_busy(busy, out):
    """``gpu_busy.png``: the share of a forward modelling and a gradient during which the GPU works."""
    fig, axes = plt.subplots(1, 1, figsize=(7.5, 3.8), facecolor=SURFACE)
    ax = axes
    _style(ax)
    configurations = [(b, m) for b in BACKENDS for m in MODES]
    colours = {("opencl", "legacy"): INK_MUTED, ("opencl", "fixed"): "#b9b8b0", ("cuda", "legacy"): ORANGE,
               ("cuda", "fixed"): BLUE}
    width = 0.2
    for c, (backend, mode) in enumerate(configurations):
        for s, scenario in enumerate(BUSY_SCENARIOS):
            value = 100 * busy[backend, mode, scenario]
            x = s + (c - 1.5) * width
            ax.bar(x, value, width, color=colours[backend, mode], edgecolor=SURFACE, linewidth=1.5,
                   label=f"{NAMES[backend]}, {mode}" if s == 0 else None)
            ax.text(x, value + 1.5, f"{value:.0f} %", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(BUSY_SCENARIOS)), [f"{s.capitalize()} scenario" for s in BUSY_SCENARIOS], color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_ylabel("share of the time the GPU works", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=4, loc="lower left", bbox_to_anchor=(0, 1))
    fig.suptitle("How busy the GPU is (higher is better)", color=INK, fontsize=12, x=0.01, ha="left", y=1.08)
    fig.tight_layout()
    return _save(fig, out, "gpu_busy.png")


def figure_memory(memory, out):
    """``memory.png``: the peak GPU memory of the CUDA default and of the features only CUDA has."""
    fig, axes = plt.subplots(1, 1, figsize=(7.5, 3.8), facecolor=SURFACE)
    ax = axes
    _style(ax)
    configurations = list(memory)
    colours = {"default": BLUE, "boundary saving": ORANGE, "FP16": "#1baf7a", "both": INK_MUTED}
    width = 0.2
    for c, name in enumerate(configurations):
        for s, scenario in enumerate(SCENARIOS):
            value = memory[name][scenario] / 1024
            x = s + (c - 1.5) * width
            ax.bar(x, value, width, color=colours[name], edgecolor=SURFACE, linewidth=1.5,
                   label=name if s == 0 else None)
            mb = memory[name][scenario]
            ax.text(x, value + 0.05, f"{mb / 1024:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB", ha="center",
                    va="bottom", fontsize=7, color=INK, rotation=90)
    ax.set_xticks(range(len(SCENARIOS)), [f"{s.capitalize()} scenario" for s in SCENARIOS], color=INK)
    ax.set_ylim(0, max(max(v.values()) for v in memory.values()) / 1024 * 1.3)
    ax.set_ylabel("peak GPU memory (GB; lower is better)", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=4, loc="lower left", bbox_to_anchor=(0, 1))
    fig.suptitle("Memory of the card with the options only CUDA has", color=INK, fontsize=12, x=0.01, ha="left",
                 y=1.08)
    fig.tight_layout()
    return _save(fig, out, "memory.png")


def figures(out=FIGURES, folder=RESULTS, release=RELEASE):
    """Every figure of the document: ``[paths]``."""
    benchmarks, busy = load(folder)
    memory = release_memory(release)
    return [figure_like_for_like(benchmarks, out), figure_defaults(benchmarks, out), figure_busy(busy, out),
            figure_memory(memory, out)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true", help="run the session on the GPU first (25 minutes)")
    parser.add_argument("--out", default=FIGURES, help="folder of the figures")
    args = parser.parse_args(argv)
    if args.measure:
        measure()
    for path in figures(args.out):
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
