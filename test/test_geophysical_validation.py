"""The geophysical validation (roadmap step 4.6): ``tools/geophysical_validation.py``.

The tool runs complete inversions (multi-scale FWI and the eight time-lapse strategies) with OpenCL and
CUDA and writes the figures and the report that a geophysicist reviews. Tested here: its models are those
of the examples of the original repository, its metrics and its recording of the misfit, the figures and
the report (from synthetic results, without a GPU), and, on the GPU, the reduced version: every inversion
lowers its misfit in every configuration, and CUDA gives the estimates of OpenCL.
"""
import copy
import os
import sys

import numpy as np
import pytest

from helpers import running_reference

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import geophysical_validation as gv

pytestmark = pytest.mark.skipif(running_reference(), reason="the tool runs the fork's backends and modes")


def test_the_models_are_those_of_the_examples():
    # At 100 x 100: the louboutin model of PyFWI, the monitor of the time-lapse notebook (vp 20 % higher in
    # rows 20:30, columns 35:55) and the background vp of ModelGenerator(smoothing=1) as the starting vp.
    import PyFWI.model_dataset as md

    baseline, monitor, start, inclusion, box = gv.make_models(100, 100)
    louboutin = md.ModelGenerator("louboutin")()
    for par in ("vp", "vs", "rho"):
        np.testing.assert_array_equal(baseline[par], louboutin[par], err_msg=par)
    notebook = copy.deepcopy(louboutin)
    notebook["vp"][20:30, 35:55] += 0.2 * notebook["vp"][20:30, 35:55]
    np.testing.assert_array_equal(monitor["vp"], notebook["vp"])
    np.testing.assert_array_equal(start["vp"], md.ModelGenerator("louboutin")(smoothing=1)["vp"])
    for par in ("vs", "rho"):  # only vp is unknown
        np.testing.assert_array_equal(start[par], baseline[par])
        np.testing.assert_array_equal(monitor[par], baseline[par])
    assert box.sum() == 200 and np.array_equal(box, monitor["vp"] != baseline["vp"])
    assert np.array_equal(inclusion, louboutin["vp"] == 3000)


def test_the_small_models_scale_the_geometry():
    baseline, monitor, _, inclusion, box = gv.make_models(50, 50)
    assert baseline["vp"].shape == (50, 50) and baseline["vp"].dtype == np.float32
    assert inclusion[25, 25] and inclusion.sum() < np.pi * 5 ** 2 + 1  # radius 5 at the centre
    assert box.sum() == 5 * 10 and box[10:15, 17:27].all()
    np.testing.assert_allclose(monitor["vp"][box], 1.2 * baseline["vp"][box], rtol=1e-6)


def test_the_fwi_metrics():
    true, _, start, inclusion, _ = gv.make_models(50, 50)
    assert gv.fwi_metrics(true["vp"], true["vp"], start["vp"], inclusion) == {"error_ratio": 0.0, "contrast": 1.0}
    assert gv.fwi_metrics(start["vp"], true["vp"], start["vp"], inclusion) == {"error_ratio": 1.0, "contrast": 0.0}
    half = gv.fwi_metrics(start["vp"] + 0.5 * (true["vp"] - start["vp"]), true["vp"], start["vp"], inclusion)
    assert half == pytest.approx({"error_ratio": 0.5, "contrast": 0.5})


def test_the_time_lapse_metrics():
    baseline, monitor, _, _, box = gv.make_models(50, 50)
    true_dm = monitor["vp"] - baseline["vp"]
    assert gv.time_lapse_metrics(true_dm, true_dm, box) == pytest.approx(
        {"recovered": 1.0, "correlation": 1.0, "artefacts": 0.0, "error": 0.0})
    assert gv.time_lapse_metrics(np.zeros_like(true_dm), true_dm, box) == pytest.approx(
        {"recovered": 0.0, "correlation": 0.0, "artefacts": 0.0, "error": 1.0})
    artefact = np.zeros_like(true_dm)
    artefact[30:] = 25.0  # below the box: the lower 20 of 50 rows
    m = gv.time_lapse_metrics(0.5 * true_dm + artefact, true_dm, box)  # half the change, and the artefact
    assert m["recovered"] == pytest.approx(0.5)
    assert m["artefacts"] == pytest.approx(25.0 * np.sqrt(20 * 50 / (2500 - box.sum())) / true_dm[box].mean())
    assert 0 < m["correlation"] < 1
    assert gv.agreement(true_dm, true_dm, true_dm) == 0.0
    assert gv.agreement(true_dm, 0.9 * true_dm, true_dm) == pytest.approx(0.1)


class _Inversion:
    """Stands for ``FWI`` / ``TimeLapse``: a stage evaluates the misfit a few times."""

    def lbfgs(self, model, iterations, freq):
        return [self.fprime(model, freq)[0] for _ in range(iterations)]

    def fprime(self, model, freq):
        self.calls = getattr(self, "calls", 0) + 1
        return 10.0 / self.calls, np.zeros(1)


def test_the_recorder_keeps_every_evaluation_stage_by_stage():
    recorder, inversion, joint = gv.Recorder(n_freqs=2), _Inversion(), _Inversion()
    recorder.watch(inversion, "lbfgs", "fprime", "fwi")
    recorder.watch(joint, "lbfgs", "fprime", "joint")
    assert inversion.lbfgs(None, 3, 15) == [10.0, 5.0, 10.0 / 3]  # the wrapped methods still answer
    inversion.lbfgs(None, 2, 45)
    joint.lbfgs(None, 1, 15)
    assert [(s["inversion"], s["kind"], s["freq"], len(s["misfit"])) for s in recorder.stages] == [
        (1, "fwi", 15.0, 3), (1, "fwi", 45.0, 2), (2, "joint", 15.0, 1)]
    assert sum(len(s["misfit"]) for s in recorder.stages) == 6
    assert gv.converged(recorder.stages) == [(2, recorder.stages[2])]  # one evaluation: no decrease


def _synthetic(size="small", configurations=gv.CONFIGURATIONS):
    """Results and models of every case, as ``run`` returns them, without running an inversion."""
    p = gv.SIZES[size]
    true, monitor, start, _, _ = gv.make_models(p["nz"], p["nx"])
    true_dm = monitor["vp"] - true["vp"]
    rng = np.random.default_rng(0)
    results = {"size": size, "parameters": p, "source_amplitude": gv.SOURCE_AMPLITUDE,
               "configurations": {c[0]: {"label": c[1], "backend": c[2], "numerics": c[3]} for c in configurations},
               "environment": {"PyFWI": "test"}, "cases": {}}
    models = {}
    for case in gv.CASES:
        results["cases"][case] = {}
        for i, (name, *_) in enumerate(configurations):
            keys = ["vp"] if case == "fwi" else sorted({"dm", gv.STRATEGIES[case][2], "dm0"})
            for key in keys:
                noise = rng.normal(0, 1 + i, true_dm.shape).astype(np.float32)
                models[f"{case}/{name}/{key}"] = (start["vp"] + 0.6 * (true["vp"] - start["vp"]) if case == "fwi"
                                                  else 0.4 * true_dm) + noise
            stages = [{"inversion": 1 + s // len(p["freqs"]), "kind": "fwi", "freq": float(p["freqs"][s % 2]),
                       "misfit": list(np.geomspace(1.0, 0.1 * (1 + i), 4))} for s in range(4)]
            results["cases"][case][name] = {"time_s": 1.0 + i, "stages": stages, "keys": keys, "evaluations": 16}
    return gv.evaluate(results, models, p), models


def test_the_figures_and_the_report(tmp_path):
    results, models = _synthetic()
    paths = gv.figures(results, models, str(tmp_path))
    assert [os.path.basename(path) for path in paths] == [
        "validation_fwi.png", "validation_time_lapse.png", "validation_misfit.png"]
    assert all(os.path.getsize(path) > 10_000 for path in paths)
    report = gv.render(results, "figures")
    for text in ("## Multi-scale FWI", "## Time-lapse strategies", "## Misfit curves", "## For the reviewer",
                 "](figures/validation_fwi.png)", "Review: *(name, date"):
        assert text in report, text
    for strategy in gv.STRATEGIES:
        assert f"| `{strategy}` |" in report, strategy
    findings = gv.findings(results)
    assert findings[0] == "The misfit decreased in every stage of every inversion (108 stages)."
    assert any(line.startswith("Largest difference for the backend") for line in findings)
    largest = gv.metric_difference(results, "cuda-fixed", "opencl-fixed")
    assert 0 < largest < 0.1 and gv.metric_difference(results, "cuda-fixed", "cuda-fixed") == 0.0


def test_a_stage_that_did_not_decrease_is_reported():
    results, _ = _synthetic()
    results["cases"]["cc"]["opencl-legacy"]["stages"][1]["misfit"] = [2.0, 2.0]
    assert gv.findings(results)[0] == ("The misfit decreased in 107 of the 108 stages; it did not in: "
                                       "cc opencl-legacy stage 2 (45 Hz).")


def test_saved_results_redraw_the_figures_and_the_report(tmp_path):
    results, models = _synthetic()
    gv.save_results(results, models, str(tmp_path))
    loaded, loaded_models = gv.load_results(str(tmp_path))
    assert loaded == results and sorted(loaded_models) == sorted(models)
    gv.main(["--redraw", "--out", str(tmp_path)])
    with open(tmp_path / "report.md") as f:
        assert "](./validation_fwi.png)" in f.read()
    assert (tmp_path / "validation_misfit.png").exists()


def test_one_configuration_alone(tmp_path):
    # A run of CUDA only (--configurations cuda-fixed): no differences, figures without them.
    results, models = _synthetic(configurations=gv.CONFIGURATIONS[:1])
    assert results["agreement"]["fwi"] == {}
    assert "Differences between configurations, FWI" not in gv.render(results, ".")
    assert len(gv.figures(results, models, str(tmp_path))) == 3


@pytest.mark.slow
@pytest.mark.gpu
def test_the_reduced_validation(cuda_device, opencl_platform, monkeypatch):
    # The small size: every case in every configuration. Every stage of every inversion lowers the misfit,
    # FWI recovers part of the inclusion (measured: error ratio 0.72 to 0.73), every strategy's main estimate
    # correlates with the true change (0.32 to 0.44), and CUDA gives the estimates of OpenCL in the same
    # numeric mode (at most 2.4e-4 of what the inversion recovers for FWI, 1e-5 for time-lapse; the full size
    # reaches 1.7e-2 after 45 iterations: bench/geophysical-*/).
    for name in [n for n in os.environ if n.startswith("PYFWI_")]:
        monkeypatch.setenv(name, os.environ[name])  # run() removes them: monkeypatch puts them back
    monkeypatch.setenv("PYFWI_FP16", "1")  # the defaults are validated: the opt-in variables are removed
    results, _ = gv.run("small", progress=lambda line: None)
    assert "PYFWI_FP16" not in os.environ
    assert [(case, c, s["freq"]) for case, runs in results["cases"].items() for c, r in runs.items()
            for _, s in gv.converged(r["stages"])] == []
    for c, run in results["cases"]["fwi"].items():
        assert run["metrics"]["vp"]["error_ratio"] < 0.9, c
    for strategy, (_, _, key) in gv.STRATEGIES.items():
        for c, run in results["cases"][strategy].items():
            assert run["metrics"][key]["correlation"] > 0.2, (strategy, c)
    backend = [value for pairs in results["agreement"].values() for value in pairs["backend"].values()]
    assert max(backend) < 1e-2
