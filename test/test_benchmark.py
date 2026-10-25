"""Tests of the benchmark script (``tools/benchmark.py``).

The launch and copy counters are what later phases will be judged on (fewer launches
and blocking copies per step), so they are checked against the exact structure of the
time loops of PyFWI 0.1.10:

* forward, per shot: one ``MakeAllZero``, 4 field updates per internal step, one
  ``injSrc`` every ``dt_scale`` steps; 5 copies per checkpoint (blocking in 0.1.10, not
  since step 1.5) and 5 for the seismograms (not blocking since step 1.4);
* gradient, per shot: ``Grad`` (and both adjoint updates) at every step of the
  backward loop, i.e. ``nt - 1`` times; 5 copies per restored checkpoint (blocking in
  0.1.10, not since step 1.5), and the residual: 5 blocking copies per injection in
  0.1.10, 5 per shot since step 1.4.
"""
import json

import pytest

from helpers import running_reference
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
    assert gradient["launch:Grad"] == gradient["launch:Adj_update_v"] == ns * (nt - 1)
    # Copies: 5 per checkpoint in the forward loop and 5 seismograms per shot; in the gradient,
    # 5 per checkpoint restored (the backward loop stops at t = 1), the residual, and 3 reads of
    # the gradient at the end.
    restores = int((W.chp >= 1).sum())
    adjoint_injections = len([t for t in range(nt - 1, 0, -1) if t % scale == 0])
    if running_reference():
        # 0.1.10: every copy blocks, and one residual sample is sent before each injection.
        checkpoint_blocking, seismograms_blocking, residual, on_device = 5, 5, 5 * adjoint_injections, 0
    else:
        # Step 1.4: seismograms read without blocking; the residual of a shot is sent once (a
        # blocking copy), then each sample is copied on the device. Step 1.5: the checkpoints are
        # copied without blocking.
        checkpoint_blocking, seismograms_blocking, residual, on_device = 0, 0, 5, 5 * adjoint_injections
    assert forward["copies"] == ns * (5 * nchp + 5)
    assert forward["blocking_copies"] == ns * (checkpoint_blocking * nchp + seismograms_blocking)
    assert gradient["copies"] == ns * (5 * restores + residual + on_device) + 3
    assert gradient["blocking_copies"] == ns * (checkpoint_blocking * restores + residual) + 3


def test_counter_restores_pyopencl():
    import pyopencl as cl
    call, copy, enqueue = cl.Kernel.__call__, cl.enqueue_copy, cl.enqueue_nd_range_kernel
    with count_opencl_calls():
        assert cl.Kernel.__call__ is not call and cl.enqueue_copy is not copy
        assert cl.enqueue_nd_range_kernel is not enqueue
    assert cl.Kernel.__call__ is call and cl.enqueue_copy is copy and cl.enqueue_nd_range_kernel is enqueue


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


def _stored(folder, label):
    """A result file stored in ``bench/<folder>/<label>.json``."""
    import os
    with open(os.path.join(os.path.dirname(__file__), "..", "bench", folder, f"{label}.json")) as f:
        return json.load(f)


def test_step_1_3_made_each_launch_cheaper():
    # Roadmap step 1.3 (docs/cuda-migration/reports/step-1.3-kernel-launches.md): kernels retrieved
    # once and arguments set once. Checks the recorded results, not this machine.
    baseline = _stored("baseline", "ref-0.1.10-pyopencl-2022")["scenarios"]["small"]
    new = {v: _stored("step-1.3", f"fork-1.3-pyopencl-{v}")["scenarios"]["small"] for v in ("2022", "2026")}
    for run in new.values():
        assert baseline["forward_s"]["median"] / run["forward_s"]["median"] > 2.5
        assert baseline["gradient_s"]["median"] / run["gradient_s"]["median"] > 2
        assert run["per_step"] == pytest.approx(baseline["per_step"])  # same launches and copies per step
        assert run["cv_forward"] < 0.03 and run["cv_gradient"] < 0.03
    # The slowdown of PyOpenCL 2026 (+44 % forward in the baseline) is gone: within 10 %.
    for kind in ("forward_s", "gradient_s"):
        assert abs(new["2026"][kind]["median"] / new["2022"][kind]["median"] - 1) < 0.1


def test_step_1_4_removed_the_blocking_residual_copies():
    # Roadmap step 1.4: the residual of a shot goes to the device once. In the small scenario
    # (1,050 steps, an injection every 7) that removes 5 x 149 blocking copies per shot from the
    # gradient: 0.71 per step, from 2.71 to 2.00.
    before = _stored("step-1.3", "fork-1.3-pyopencl-2026")["scenarios"]["small"]
    after = _stored("step-1.4", "fork-1.4-pyopencl-2026")["scenarios"]["small"]
    assert before["per_step"]["blocking_copies"] == pytest.approx(2.71, abs=0.01)
    assert after["per_step"]["blocking_copies"] == pytest.approx(2.00, abs=0.01)
    assert after["gradient_s"]["median"] <= before["gradient_s"]["median"]


def test_step_1_5_the_host_no_longer_waits_for_checkpoint_copies():
    # Roadmap step 1.5: contiguous checkpoints copied without blocking. Recorded results.
    baseline = _stored("baseline", "ref-0.1.10-pyopencl-2022")["scenarios"]
    before = _stored("step-1.3", "fork-1.3-pyopencl-2026")["scenarios"]
    after = _stored("step-1.5", "fork-1.5-pyopencl-2026")["scenarios"]
    for name in ("small", "medium"):
        assert after[name]["per_step"]["blocking_copies"] < 0.01, name
    # Medium scenario: the copies dominated after step 1.3.
    assert before["medium"]["forward_s"]["median"] / after["medium"]["forward_s"]["median"] > 2.5
    assert baseline["medium"]["gradient_s"]["median"] / after["medium"]["gradient_s"]["median"] > 1.4
    assert after["small"]["forward_s"]["median"] <= before["small"]["forward_s"]["median"]
