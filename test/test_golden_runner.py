"""Tests of the golden-master runner (``test/golden_runner.py``) on a real OpenCL device."""
import numpy as np
import pytest

from golden_cases import PropagationCase, propagation_cases
from golden_runner import COMPONENTS, case_inputs, run_propagation_case


def test_case_inputs_follow_the_case():
    case = propagation_cases()[0]
    x = case_inputs(case)
    assert x["inpa"]["sdo"] == case.order and x["inpa"]["acq_type"] == case.acq_type
    assert x["inpa"]["energy_balancing"] == case.energy_balancing
    assert x["true"]["vp"].shape == (case.nz, case.nx) and x["start"]["vp"].shape == (case.nz, case.nx)


@pytest.mark.parametrize("case", [
    PropagationCase(acq_type=1, components=0, order=4, energy_balancing=False, chpr=20, parameterization="dv"),
    PropagationCase(acq_type=2, components=4, order=8, energy_balancing=True, chpr=0, parameterization="lmd"),
], ids=lambda c: c.name)
def test_runner_outputs_are_complete_and_deterministic(opencl_platform, case):
    first, second = run_propagation_case(case), run_propagation_case(case)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
    # Layers present, with the expected shapes.
    nr = case_inputs(case)["rec_loc"].shape[0]
    nt = int(1 + case.t // 0.004)
    for c in COMPONENTS:
        assert first[f"raw_seismogram/{c}"].shape == (nt, nr, case.ns)
    gradient_names = ("vp", "vs", "rho") if case.parameterization == "dv" else ("lam", "mu", "rho")
    for p in gradient_names:
        assert first[f"gradient/{p}"].shape == (case.nz, case.nx)
    for p in ("lam", "mu", "rho"):
        assert first[f"raw_gradient/{p}"].shape == (case.nz, case.nx)
    assert first["misfit"].shape == () and first["misfit"] > 0
    assert all(np.isfinite(v).all() for v in first.values())
