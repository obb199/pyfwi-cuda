"""Figures of the project overview (``docs/cuda-migration/overview.md``), drawn from recorded results.

Every number comes from a file of the repository or from a measurement recorded in the roadmap and
its reports (the source is given next to each table below), so the figures can be drawn again at any
time without a GPU:

* ``speedup_by_release.png``: forward modelling and gradient of the three benchmark scenarios, each
  release against PyFWI 0.1.10 (``bench/``), with the targets of gate G3;
* ``phase3_steps.png``: how the steps of Phase 3 added up, from the before / after measurements of
  each step (``execution-roadmap.md``, "Results of Phase 3");
* ``gpu_time_per_step.png``: GPU time of one time step of one shot in the medium scenario, by role of
  the kernels, 0.1.10 (``bench/baseline/profile-medium-ref-0.1.10.json``) against 0.3.0
  (``reports/timeline-0.3.0.md``);
* ``gpu_busy.png``: the share of the run during which the GPU works, 0.1.10 against 0.3.0;
* ``accuracy.png``: the checks of the mathematical baseline in the ``legacy`` mode (the arithmetic of
  0.1.10, ``test/golden/math_baseline.json``) and in the ``fixed`` mode (``gates/G2.md``).

Colours follow the reference palette of the data-visualisation guidelines used for the project
(validated for colour-vision deficiencies): the release being presented in blue, older releases in
greys; categories in the fixed order blue, orange, aqua.

Usage::

    ../envs/dev/bin/python tools/plot_overview.py              # writes docs/cuda-migration/figures/
    ../envs/dev/bin/python tools/plot_overview.py --out /tmp/figures

Used by
-------
``docs/cuda-migration/overview.md``. Tested by ``test/test_plot_overview.py``.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # after choosing the backend

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(REPO, "bench")
FIGURES = os.path.join(REPO, "docs", "cuda-migration", "figures")

# ---------------------------------------------------------------------------- style

#: Chart surface, inks, grid and axis of the reference palette (light mode).
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
#: Categorical slots, in their fixed order: blue, orange, aqua.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SCENARIOS = ("small", "medium", "large")
KINDS = (("forward_s", "Forward modelling"), ("gradient_s", "Gradient"))


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


def _figure(columns, width=10.0, height=4.2):
    fig, axes = plt.subplots(1, columns, figsize=(width, height), facecolor=SURFACE)
    return fig, axes if columns > 1 else [axes]


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- data

def stored(path):
    """The scenarios of a result file of ``tools/benchmark.py``, ``bench/<path>.json``."""
    with open(os.path.join(BENCH, f"{path}.json")) as f:
        return json.load(f)["scenarios"]


#: The releases of the first figure: label, result file, and the run of 0.1.10 measured under the same
#: conditions (the stored baseline of Oct 25 for the first two; for 0.3.0, a run of the same session,
#: when the GPU was in a slower state: see reports/benchmark-0.3.0.md).
RELEASES = (
    ("0.1.11 (OpenCL)", "release-0.1.11/fork-0.1.11-pyopencl-2026", "baseline/ref-0.1.10-pyopencl-2022", AXIS),
    ("0.2.0 (CUDA, fixed mode)", "release-0.2.0/fork-0.2.0-cuda-fixed", "baseline/ref-0.1.10-pyopencl-2022",
     INK_MUTED),
    ("0.3.0 (default: CUDA, fixed mode)", "release-0.3.0/fork-0.3.0-cuda-fixed",
     "release-0.3.0/ref-0.1.10-pyopencl-2022", BLUE),
)

#: The speed-ups that gate G3 set against 0.1.10, per scenario (low, high).
G3_TARGETS = {"small": (20, 100), "medium": (10, 40), "large": (3, 10)}


def release_speedups():
    """``{release label: {scenario: {kind: speed-up against 0.1.10}}}`` (``RELEASES``)."""
    result = {}
    for label, run, reference, _ in RELEASES:
        measured, ref = stored(run), stored(reference)
        result[label] = {s: {k: ref[s][k]["median"] / measured[s][k]["median"] for k, _ in KINDS} for s in SCENARIOS}
    return result


#: Phase 3, step by step, in the order the steps were done: the forward and gradient times (s) before
#: and after each step, measured in the same session (execution-roadmap.md, "Results of Phase 3"; the
#: small scenario with two decimals until step 3.10). None: no change measured.
PHASE3_STEPS = (
    ("3.1", "block size", None, None, None, None),
    ("3.2", "native kernels", None, None, None, None),
    ("3.3", "1/dx", None, None, (3.06, 3.03), (5.86, 5.57)),
    ("3.4", "1-D injection", None, None, (3.03, 3.05), None),
    ("3.5", "fusion", (0.20, 0.17), (0.19, 0.15), (3.05, 3.04), (5.57, 5.06)),
    ("3.6", "time on GPU", (0.17, 0.14), None, (3.04, 2.90), (5.06, 5.11)),
    ("3.7", "CUDA graphs", (0.14, 0.08), (0.15, 0.09), (2.90, 2.76), (5.11, 5.00)),
    ("3.8", "checkpoints on GPU", (0.08, 0.04), (0.09, 0.07), (2.76, 1.82), (5.00, 4.10)),
    ("3.10", "damping at borders", None, (0.07, 0.065), (1.82, 1.60), (4.10, 3.84)),
    ("3.9", "shots in batches", (0.047, 0.023), (0.073, 0.066), (1.78, 1.65), (4.31, 4.16)),
    ("3.11", "idle gaps", (0.0223, 0.0219), None, None, None),
)


def phase3_cumulative():
    """``{(scenario, kind): [speed-up over 0.2.0 after each step]}``: the product of the step ratios."""
    series = {}
    for column, key in enumerate((("small", "forward_s"), ("small", "gradient_s"), ("medium", "forward_s"),
                                  ("medium", "gradient_s")), start=2):
        total, values = 1.0, []
        for step in PHASE3_STEPS:
            pair = step[column]
            if pair is not None:
                total *= pair[0] / pair[1]
            values.append(total)
        series[key] = values
    return series


#: GPU time of the kernels of 0.3.0 in the medium scenario, mean per launch (us), each launch running
#: the 2 shots of the scenario, and launches per run (reports/timeline-0.3.0.md, medium scenario).
TIMELINE_MEDIUM_0_3_0 = {
    "forward": {"update_v": (49.78, 2998), "update_tau": (68.16, 2998), "inject_and_record": (2.80, 2998),
                "advance": (1.26, 2998), "copy_checkpoint": (61.40, 599)},
    "gradient": {"grad_adjoint_v": (166.49, 2997), "update_tau": (76.90, 2399), "Adj_update_tau": (55.05, 2997),
                 "update_v": (55.83, 2399), "copy_checkpoint": (59.34, 598), "inject_residual": (1.95, 2997),
                 "advance": (1.03, 2997)},
}

#: The role of every kernel, for the third figure.
ROLES = {
    "Field updates (forward, or backward in the gradient)": {
        "update_velx", "update_velz", "update_tauz", "update_tauxz", "update_v", "update_tau"},
    "Adjoint field and gradient": {"Grad", "Adj_update_v", "Adj_update_tau", "grad_adjoint_v"},
    "Sources, receivers, checkpoints, time step": {
        "injSrc", "Adj_injSrc", "inject_and_record", "inject_residual", "advance", "copy_checkpoint"},
}


def gpu_time_per_step():
    """``{version: {kind: {role: us of GPU time per time step and shot}}}``, medium scenario."""
    with open(os.path.join(BENCH, "baseline", "profile-medium-ref-0.1.10.json")) as f:
        profile = dict(json.load(f)["kernels"])
    # 0.1.10 ran one forward modelling and one gradient of the 2 shots of the scenario, one shot per launch:
    # the gradient steps are the launches of Grad (nt - 1 per shot), the forward steps 2 more (nt per shot).
    gradient_steps = profile["Grad"]["launches"]
    steps = gradient_steps + 2
    old = {kind: {role: 0.0 for role in ROLES} for kind in ("forward", "gradient")}
    updates = ("update_velx", "update_velz", "update_tauz", "update_tauxz")
    for name in updates:
        old["forward"][_role(name)] += profile[name]["mean_us"]
        # In the gradient, the forward field is rebuilt backwards except at the restored checkpoint steps.
        backward_share = (profile[name]["launches"] - steps) / gradient_steps
        old["gradient"][_role(name)] += profile[name]["mean_us"] * backward_share
    old["forward"][_role("injSrc")] += profile["injSrc"]["mean_us"] * profile["injSrc"]["launches"] / steps
    for name in ("Grad", "Adj_update_v", "Adj_update_tau"):
        old["gradient"][_role(name)] += profile[name]["mean_us"]
    old["gradient"][_role("Adj_injSrc")] += (profile["Adj_injSrc"]["mean_us"] * profile["Adj_injSrc"]["launches"]
                                             / gradient_steps)
    new = {}
    for kind, kernels in TIMELINE_MEDIUM_0_3_0.items():
        steps_of_run = kernels["advance"][1]
        new[kind] = {role: 0.0 for role in ROLES}
        for name, (mean, launches) in kernels.items():
            new[kind][_role(name)] += mean * launches / steps_of_run / 2  # 2 shots per launch
    return {"0.1.10": old, "0.3.0": new}


def _role(kernel):
    return next(role for role, names in ROLES.items() if kernel in names)


def gpu_busy():
    """``{scenario: {version: share of the run with the GPU busy}}``.

    0.1.10: forward modellings and gradient of the benchmark run (``tools/opencl_profile.py``, stored in
    ``bench/baseline/profile-*-ref-0.1.10.json``). 0.3.0: one forward modelling and one gradient, the
    time of each range weighted by its busy share (reports/timeline-0.3.0.md: the small scenario with the
    medians of its runs).
    """
    result = {}
    timeline = {"small": ((24.36, 0.856), (65.55, 0.954)), "medium": ((415.76, 0.970), (1040.77, 0.988))}
    for scenario in ("small", "medium"):
        with open(os.path.join(BENCH, "baseline", f"profile-{scenario}-ref-0.1.10.json")) as f:
            old = json.load(f)["gpu_fraction"]
        ranges = timeline[scenario]
        new = sum(length * busy for length, busy in ranges) / sum(length for length, _ in ranges)
        result[scenario] = {"0.1.10": old, "0.3.0": new}
    return result


def accuracy():
    """The checks of the mathematical baseline, ``{check: (legacy, fixed)}``, and the convergence orders.

    Legacy: ``test/golden/math_baseline.json`` (0.1.10), but for the adjoint dot product, which 0.1.10
    cannot run and the fork measured in its legacy mode; fixed: gates/G2.md.
    """
    with open(os.path.join(REPO, "test", "golden", "math_baseline.json")) as f:
        metrics = json.load(f)["metrics"]
    errors = {
        "vp gradient against\nfinite differences": (metrics["gradient_fd"]["median_abs_error"], 0.006),
        "vs gradient against\nfinite differences": (metrics["gradient_fd_vs"]["median_abs_error"], 0.0027),
        "Density gradient against\nfinite differences": (metrics["gradient_fd_rho"]["median_abs_error"], 0.014),
        "Adjoint dot-product\nmismatch (one step)": (4.3e-3, 1.1e-7),
    }
    orders = (metrics["convergence"]["order_8"]["orders"][-1], 7.5)
    return errors, orders


# ---------------------------------------------------------------------------- figures

def plot_speedup_by_release(out):
    """Small multiples by scenario: speed-up of each release against 0.1.10, log scale, with the G3 band."""
    data = release_speedups()
    fig, axes = _figure(3, height=4.4)
    width = 0.26
    for ax, scenario in zip(axes, SCENARIOS):
        _style(ax)
        low, high = G3_TARGETS[scenario]
        ax.axhspan(low, high, color=GRID, alpha=0.6, linewidth=0,
                   label="Target of gate G3 (range)" if scenario == "small" else None)
        for r, (label, _, _, colour) in enumerate(RELEASES):
            for k, (kind, _) in enumerate(KINDS):
                x = k + (r - 1) * width
                value = data[label][scenario][kind]
                ax.bar(x, value, width, color=colour, edgecolor=SURFACE, linewidth=1.5,
                       label=label if (k == 0 and scenario == "small") else None)
                if colour == BLUE:  # the release presented: its value on the bar end
                    ax.text(x, value * 1.12, f"{value:.0f}×" if value >= 10 else f"{value:.1f}×",
                            ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 200)
        ax.axhline(1, color=AXIS, linewidth=1)
        ax.set_xticks([0, 1], [name for _, name in KINDS], color=INK_SECONDARY)
        ax.set_title(f"{scenario.capitalize()} scenario", color=INK, fontsize=11, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    axes[0].set_ylabel("Times faster than PyFWI 0.1.10", color=INK_SECONDARY)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04),
               labelcolor=INK_SECONDARY)
    fig.suptitle("Each release against PyFWI 0.1.10 (higher is better)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out, "speedup_by_release.png")


def plot_phase3_steps(out):
    """How the steps of Phase 3 added up: speed-up over 0.2.0 after each step (small and medium)."""
    series = phase3_cumulative()
    labels = ["0.2.0"] + [f"{step}\n{name.replace(' ', chr(10), 1)}" for step, name, *_ in PHASE3_STEPS]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), facecolor=SURFACE)
    for ax, scenario in zip(axes, ("small", "medium")):
        _style(ax)
        for (kind, name), colour in zip(KINDS, (BLUE, ORANGE)):
            values = [1.0] + series[(scenario, kind)]
            ax.step(range(len(values)), values, where="mid", color=colour, linewidth=2, label=name)
            ax.plot(range(len(values)), values, "o", color=colour, markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
            ax.annotate(f"{values[-1]:.1f}×", (len(values) - 1, values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK, fontweight="bold")
        ax.axhline(1, color=AXIS, linewidth=1)
        ax.set_xticks(range(len(labels)), labels, fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(labels) - 0.2)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{scenario.capitalize()} scenario", color=INK, fontsize=11, loc="left")
        ax.set_ylabel("Times faster than 0.2.0", color=INK_SECONDARY)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    fig.suptitle("Phase 3 step by step: speed-up over 0.2.0 after each step (in the order they were done)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "phase3_steps.png")


def plot_gpu_time_per_step(out):
    """Stacked horizontal bars: GPU time of one step of one shot, by role, 0.1.10 against 0.3.0."""
    data = gpu_time_per_step()
    rows = [("0.1.10", "forward"), ("0.3.0", "forward"), ("0.1.10", "gradient"), ("0.3.0", "gradient")]
    fig, axes = _figure(1, width=10, height=3.6)
    ax = axes[0]
    _style(ax, grid_axis="x")
    colours = (BLUE, ORANGE, AQUA)
    for y, (version, kind) in enumerate(rows):
        left = 0.0
        for (role, colour) in zip(ROLES, colours):
            value = data[version][kind][role]
            ax.barh(y, value, 0.6, left=left, color=colour, edgecolor=SURFACE, linewidth=1.5,
                    label=role if y == 0 else None)
            left += value
        ax.text(left + 15, y, f"{left:.0f} µs", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [f"{kind.capitalize()}, {version}" for version, kind in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("GPU time of one time step of one shot (µs), medium scenario: 440 × 440 cells",
                  color=INK_SECONDARY)
    ax.set_xlim(0, 1750)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="lower right")
    fig.suptitle("Where the GPU time of a time step goes (lower is better)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout()
    return _save(fig, out, "gpu_time_per_step.png")


def plot_gpu_busy(out):
    """Bars: share of the run with the GPU busy, 0.1.10 against 0.3.0, small and medium."""
    data = gpu_busy()
    fig, axes = _figure(1, width=6.5, height=3.6)
    ax = axes[0]
    _style(ax)
    width = 0.34
    for v, (version, colour) in enumerate((("0.1.10", INK_MUTED), ("0.3.0", BLUE))):
        for s, scenario in enumerate(data):
            x = s + (v - 0.5) * width
            value = 100 * data[scenario][version]
            ax.bar(x, value, width, color=colour, edgecolor=SURFACE, linewidth=1.5, label=version if s == 0 else None)
            ax.text(x, value + 1.5, f"{value:.0f} %", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(data)), [f"{s.capitalize()} scenario" for s in data], color=INK)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_ylabel("Share of the run with the GPU working", color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left", ncol=2, bbox_to_anchor=(0, 1))
    fig.suptitle("How busy the GPU is (higher is better)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "gpu_busy.png")


def plot_accuracy(out):
    """Two panels: errors of the legacy and fixed modes (log scale), and the convergence order of order 8."""
    errors, orders = accuracy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 1.1]})
    ax = axes[0]
    _style(ax)
    width = 0.36
    for m, (mode, colour) in enumerate((("legacy (arithmetic of 0.1.10)", INK_MUTED), ("fixed", BLUE))):
        for c, (check, values) in enumerate(errors.items()):
            x = c + (m - 0.5) * width
            ax.bar(x, values[m], width, color=colour, edgecolor=SURFACE, linewidth=1.5,
                   label=mode if c == 0 else None)
            text = _percent(values[m]) if c < 3 else f"{values[m]:.1e}"
            ax.text(x, values[m] * 1.25, text, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(3e-8, 20)
    ax.set_xticks(range(len(errors)), list(errors), fontsize=8.5, color=INK)
    ax.set_ylabel("Relative error (lower is better)", color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.12))
    ax.set_title("Gradient and adjoint checks", color=INK, fontsize=11, loc="left")
    ax = axes[1]
    _style(ax)
    for m, colour in enumerate((INK_MUTED, BLUE)):
        ax.bar(m, orders[m], 0.6, color=colour, edgecolor=SURFACE, linewidth=1.5)
        ax.text(m, orders[m] - 0.25, f"{orders[m]:.1f}", ha="center", va="top", fontsize=9, color="#ffffff",
                fontweight="bold")
    ax.axhline(8, color=INK_SECONDARY, linewidth=1)
    ax.text(1.45, 8.15, "ideal: 8", ha="right", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks([0, 1], ["legacy", "fixed"], color=INK)
    ax.set_ylim(0, 9.5)
    ax.set_title("Convergence order of\nthe order-8 scheme", color=INK, fontsize=11, loc="left")
    fig.suptitle("The fixed numeric mode is also more accurate", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "accuracy.png")


def _percent(share):
    """A share as a percentage with two significant digits at most, without exponents (1.35 -> "135 %")."""
    value = 100 * share
    return f"{value:.0f} %" if value >= 10 else f"{value:.2g} %"


FIGURES_DRAWN = (plot_speedup_by_release, plot_phase3_steps, plot_gpu_time_per_step, plot_gpu_busy, plot_accuracy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=FIGURES, help="folder of the figures (default: %(default)s)")
    args = parser.parse_args(argv)
    paths = [draw(args.out) for draw in FIGURES_DRAWN]
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
