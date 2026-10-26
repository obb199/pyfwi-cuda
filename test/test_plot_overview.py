"""The figures of the project overview (``tools/plot_overview.py``, ``docs/cuda-migration/overview.md``).

The figures are drawn from recorded results, without a GPU: they must draw, the document must show each
of them, and the step-by-step data of Phase 3 must agree with the release benchmarks it summarises.
"""
import os
import re

import pytest

from helpers import running_reference
from tools import plot_overview

pytestmark = pytest.mark.skipif(running_reference(), reason="the overview is about the fork")

OVERVIEW = os.path.join(plot_overview.REPO, "docs", "cuda-migration", "overview.md")


def test_every_figure_draws(tmp_path):
    paths = plot_overview.main(["--out", str(tmp_path)])
    assert [os.path.basename(p) for p in paths] == ["speedup_by_release.png", "phase3_steps.png",
                                                    "gpu_time_per_step.png", "gpu_busy.png", "accuracy.png"]
    assert all(os.path.getsize(p) > 10_000 for p in paths)


def test_the_overview_shows_every_figure_of_the_repository():
    with open(OVERVIEW) as f:
        shown = set(re.findall(r"\]\(figures/([\w.]+\.png)\)", f.read()))
    stored = set(os.listdir(plot_overview.FIGURES))
    assert shown == stored == {f"{draw.__name__[len('plot_'):]}.png" for draw in plot_overview.FIGURES_DRAWN}


def test_the_steps_of_phase_3_add_up_to_the_release_gain():
    # The product of the step ratios (same-session measurements) and the ratio of the release runs
    # (0.2.0 and 0.3.0, measured hours apart with the GPU in different states) agree within 20 %.
    steps = plot_overview.phase3_cumulative()
    old, new = (plot_overview.stored(run) for run in ("release-0.2.0/fork-0.2.0-cuda-fixed",
                                                      "release-0.3.0/fork-0.3.0-cuda-fixed"))
    for (scenario, kind), values in steps.items():
        release = old[scenario][kind]["median"] / new[scenario][kind]["median"]
        assert values[-1] == pytest.approx(release, rel=0.2), (scenario, kind)


def test_the_gpu_time_per_step_and_the_targets():
    times = plot_overview.gpu_time_per_step()
    totals = {version: {kind: sum(roles.values()) for kind, roles in kinds.items()} for version, kinds in times.items()}
    assert totals["0.1.10"]["forward"] == pytest.approx(640, abs=1) and totals["0.1.10"]["gradient"] == pytest.approx(1511, abs=1)
    assert totals["0.3.0"]["forward"] == pytest.approx(67, abs=1) and totals["0.3.0"]["gradient"] == pytest.approx(171, abs=1)
    speedups = plot_overview.release_speedups()["0.3.0 (default: CUDA, fixed mode)"]
    for scenario, (low, _) in plot_overview.G3_TARGETS.items():
        assert min(speedups[scenario].values()) > low, scenario
