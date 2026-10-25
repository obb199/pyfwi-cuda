"""Tests of the benchmark script (``tools/benchmark.py``).

The launch and copy counters are what later phases will be judged on (fewer launches
and blocking copies per step), so they are checked against the exact structure of the
time loops of PyFWI 0.1.10:

* forward, per shot: one ``MakeAllZero``, 4 field updates per internal step, one
  ``injSrc`` every ``dt_scale`` steps; 5 blocking copies per checkpoint and 5 for the
  seismograms;
* gradient, per shot: ``Grad`` (and both adjoint updates) at every step of the
  backward loop, i.e. ``nt - 1`` times.
"""
import json

import pytest

from tools.benchmark import SCENARIOS, count_opencl_calls, main, run_once

#: Internal time step of PyFWI for a recording step of 4 ms (split into 7 sub-steps of at most 0.6 ms).
INTERNAL_DT = 0.004 / 7


def test_scenarios_match_the_analysis_document():
    def steps(name):
        return int(1 + SCENARIOS[name]["t"] // INTERNAL_DT)
    small, medium, large = SCENARIOS["small"], SCENARIOS["medium"], SCENARIOS["large"]
    # PyFWI computes 1 + 0.6 // (0.004 / 7) = 1050 steps (the analysis document rounds it to 1,051).
    assert (small["nz"], small["nx"], small["ns_target"], steps("small")) == (100, 100, 4, 1050)
    assert (medium["nz"], medium["nx"], medium["ns_target"]) == (400, 400, 8) and abs(steps("medium") - 3000) < 10
    assert large["nz"] * large["nx"] == 450_000 and large["ns_target"] == 50 and abs(steps("large") - 6000) < 10


def test_counter_counts_every_launch_and_copy(opencl_platform):
    _, _, forward, gradient, W = run_once("tiny")
    ns, nt, scale, nchp = int(W.ns), int(W.nt), int(W.dt_scale), int(W.nchp)
    injections = len(range(0, nt, scale))  # the source is injected when t % dt_scale == 0
    assert forward["launch:injSrc"] == ns * injections
    assert forward["launches"] == ns * (1 + 4 * nt + injections)
    assert forward["copies"] == forward["blocking_copies"] == ns * (5 * nchp + 5)
    assert gradient["launch:Grad"] == gradient["launch:Adj_update_v"] == ns * (nt - 1)


def test_counter_restores_pyopencl():
    import pyopencl as cl
    call, copy = cl.Kernel.__call__, cl.enqueue_copy
    with count_opencl_calls():
        assert cl.Kernel.__call__ is not call and cl.enqueue_copy is not copy
    assert cl.Kernel.__call__ is call and cl.enqueue_copy is copy


def test_cli_writes_a_complete_result_file(opencl_platform, tmp_path):
    main(["--scenarios", "tiny", "--repeats", "2", "--label", "check", "--out", str(tmp_path)])
    results = json.loads((tmp_path / "check.json").read_text())
    assert {"label", "date", "machine", "repeats", "scenarios"} <= set(results)
    tiny = results["scenarios"]["tiny"]
    assert len(tiny["forward_s"]["runs"]) == 2 and tiny["forward_s"]["median"] > 0
    assert tiny["per_step"]["forward_launches"] == pytest.approx(4.15, abs=0.05)  # 4 updates + 1/7 injection + zeroing
    assert {"opencl_device", "pyopencl", "pyfwi"} <= set(results["machine"])


def _fake_run(label, forward, gradient):
    """A minimal result dict in the format of tools/benchmark.py, for the report tests."""
    scenario = {"forward_s": {"median": forward}, "gradient_s": {"median": gradient}, "scale": 1,
                "host_ms_per_step": {"forward": 0.3, "gradient": 0.4},
                "per_step": {"forward_launches": 4.15, "gradient_launches": 6.3, "blocking_copies": 2.7},
                "cv_forward": 0.01, "cv_gradient": 0.02, "gpu_before": {"temperature_c": 50.0}, "gpu_after": {}}
    machine = {"opencl_device": "GPU", "driver": "1", "pyfwi": "0.1.10", "pyopencl": "x", "numpy": "y"}
    return {"label": label, "date": "today", "machine": machine, "scenarios": {"small": scenario}}


def test_report_shows_speed_ups_relative_to_the_first_run():
    from tools.benchmark_report import render
    text = render([_fake_run("before", 2.0, 4.0), _fake_run("after", 1.0, 1.0)])
    assert "| small | before | 2.00 | 4.00 | 1.00 / 1.00 |" in text
    assert "| small | after | 1.00 | 1.00 | 2.00 / 4.00 |" in text
    assert text.count("\n|") >= 6  # header rows + one row per run and scenario


def test_report_orders_scenarios_by_size_and_states_the_scaling():
    from tools.benchmark_report import render
    run = _fake_run("only", 1.0, 1.0)
    run["scenarios"]["large"] = {**run["scenarios"]["small"], "scale": 50, "scaled_to_shots": 50}
    run["scenarios"]["medium"] = {**run["scenarios"]["small"], "scale": 4, "scaled_to_shots": 8}
    text = render([run])
    assert text.index("| small |") < text.index("| medium (scaled x4 to 8 shots) |") < text.index("| large (scaled x50")


@pytest.mark.parametrize("label", ["ref-0.1.10-pyopencl-2022", "dev-pyopencl-2026"])
def test_stored_baseline_is_complete_and_stable(label):
    # Roadmap step 0.7: the baseline of 0.1.10, with a coefficient of variation below 3 % everywhere.
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "bench", "baseline", f"{label}.json")
    with open(path) as f:
        baseline = json.load(f)
    assert baseline["machine"]["pyfwi"] == "0.1.10" and baseline["repeats"] == 5
    assert set(baseline["scenarios"]) == {"small", "medium", "large"}
    for name, scenario in baseline["scenarios"].items():
        assert scenario["cv_forward"] < 0.03 and scenario["cv_gradient"] < 0.03, name
        assert scenario["guard_bands"] is True
