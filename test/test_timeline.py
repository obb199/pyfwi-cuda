"""The GPU timeline tool of gate G3 (``tools/timeline.py``, roadmap step 3.11, test type PRF).

``busy_and_gaps`` is checked on hand-made intervals; on a machine with a GPU and Nsight Systems, the
whole tool runs on the tiny scenario.
"""
import shutil

import pytest

from helpers import running_reference
from tools.timeline import SIGNIFICANT_GAP_NS, busy_and_gaps, render


def test_busy_time_is_the_union_of_the_activities():
    # A kernel 0-10, a copy 5-12 beside it, a kernel 12-20 right after; then idle until 50, a kernel
    # 50-60 and one inside it 52-55; idle until 61, a last kernel 61-62.
    intervals = [(0, 10), (5, 12), (12, 20), (50, 60), (52, 55), (61, 62)]
    busy, gaps = busy_and_gaps(intervals)
    assert busy == 20 + 10 + 1
    # The gap after 20 follows the interval 2 (12-20) and precedes 3; the gap after 60 follows 3, not 4.
    assert gaps == [(30, 2, 3), (1, 3, 5)]
    assert busy_and_gaps([]) == (0, [])


def test_the_report_lists_the_ranges_and_their_large_gaps():
    results = {"forward": {"span_ns": 2_000_000, "busy_ns": 1_500_000, "busy_fraction": 0.75, "activities": 3,
                           "gaps": 2, "median_gap_ns": 250_000, "largest_gap_ns": 400_000,
                           "significant": [(400_000, 100_000, "update_v", "copy 4 B")], "small_gaps_ns": 5,
                           "kernels": {"update_v": (2, 1_000_000, 100_000), "copy 4 B": (1, 500_000, 0)}}}
    text = render(results, "Timeline")
    assert "| forward | 2.00 | 75.0 % | 2 | 250.00 | 1 | 0.40 | 0.00 |" in text
    assert "| 400.0 | 0.10 | update_v | copy 4 B |" in text
    # update_v moves 8 arrays of 100,000 cells (3.2 MB) in 500 us: 6.4 GB/s.
    assert "| update_v | 2 | 1.00 | 500.00 | 100000 | 6 |" in text and "| copy 4 B | 1 | 0.50 | 500.00 |  |  |" in text
    assert SIGNIFICANT_GAP_NS == 10_000


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("nsys") is None, reason="Nsight Systems (nsys) is not installed")
@pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no CUDA backend")
def test_the_tool_measures_the_tiny_scenario(cuda_device, monkeypatch, tmp_path):
    from tools.timeline import main

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.delenv("PYFWI_NUMERICS", raising=False)
    results = main(["--scenario", "tiny", "--out", str(tmp_path)])
    assert set(results) == {"forward", "gradient"}
    for name, r in results.items():
        assert 0 < r["busy_ns"] <= r["span_ns"] and r["activities"] > 100, name
        assert "update_v" in r["kernels"] and "update_tau" in r["kernels"], name  # graph nodes are traced
        assert r["kernels"]["update_v"][2] >= 40 * 40, name  # threads: at least the cells of the model
    assert "grad_adjoint_v" in results["gradient"]["kernels"]
