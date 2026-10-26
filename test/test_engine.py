"""The propagation engine without a device (roadmap step 2.1): ``PyFWI.engine.propagation``.

The engine runs the time loops of PyFWI 0.1.10 through a backend. Here it runs on the fake NumPy
backend of ``fake_backend.py``, whose toy kernels have effects simple enough to predict by hand,
so that every rule of the loops is checked without OpenCL: when the source is injected and the
receivers recorded, which kernels run in which order with which arguments, where the checkpoints
go and when they come back, how the residual is injected backwards, and what the gradient
accumulates. That the real kernels give the results of 0.1.10 is checked by the
characterisation tests; that these unit tests alone cover the engine is checked at the end.

The case: a 6 x 5 grid, 2 shots with their sources on cells (2, 1) and (2, 3), 3 receivers,
10 internal steps with an injection every 3 (recorded samples 0 to 3), checkpoints at steps
0, 4 and 9 (``chpr`` = 30).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from fake_backend import FakeBackend
from helpers import running_reference

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no engine")

TNZ, TNX, NS, NR, NT, NT_EXT, DT_SCALE = 6, 5, 2, 3, 10, 4, 3
SRCX, SRCZ = np.array([1, 3], np.int32), np.array([2, 2], np.int32)
RHO_INVERSE, LAM, MU = 1.0, 2.0, 3.0
COMPONENTS = ("vx", "vz", "taux", "tauz", "tauxz")


def source(t):
    """The five source samples of recorded sample ``t``: 10 (t + 1) each."""
    return [10.0 * (t + 1)] * 5


def make_engine(injection="surface", chpr=30):
    """An engine on the fake backend, with a model (1 / rho = 1, lambda = 2, mu = 3) and a boundary."""
    from PyFWI.engine.propagation import PropagationEngine, checkpoint_steps
    backend = FakeBackend(nr=NR, tnx=TNX)
    sources = {"elastic": "elastic source", "crosswell": "crosswell source", "surface": "surface source"}
    engine = PropagationEngine(backend, sources, injection, tnz=TNZ, tnx=TNX, ns=NS, nr=NR, nt=NT, nt_ext=NT_EXT,
                               dt_scale=DT_SCALE, chp=checkpoint_steps(chpr, NT), dxr=np.int32(1), srcx=SRCX,
                               srcz=SRCZ, source=source, rec_top_right_var=0)
    field = np.ones((TNZ, TNX), np.float32)
    engine.set_pml(0 * field, 0 * field)
    engine.set_model(RHO_INVERSE * field, MU * field, LAM * field)
    return engine


def residual():
    """Residual of both shots: sample k of shot s at receiver r is (k + 1) (s + 1) + r / 10."""
    samples = np.arange(NT_EXT)[:, None] + 1
    columns = [(s + 1) * samples + np.arange(NR) / 10 for s in range(NS)]
    one = np.hstack(columns).astype(np.float32)  # (NT_EXT, NR * NS)
    return {par: one.copy() for par in COMPONENTS}


def injected(step):
    """Total of the source samples injected at the source cell up to internal step ``step`` included."""
    return sum(source(k)[0] for k in range(NT_EXT) if k * DT_SCALE <= step)


def launches(engine):
    """Names of the kernels launched, in order, from the log of the fake backend."""
    return [entry[1] for entry in engine.backend.log if entry[0] == "launch"]


# ---------------------------------------------------------------------------------- helpers

def test_checkpoint_steps_follow_0_1_10():
    from PyFWI.engine.propagation import checkpoint_steps
    assert list(checkpoint_steps(30, 10)) == [0, 4, 9]
    assert list(checkpoint_steps(100, 10)) == list(range(10))
    assert list(checkpoint_steps(0, 10)) == [1, 9]  # the default chpr=0: two checkpoints only


def test_margins_cover_the_farthest_access_and_are_aligned():
    from PyFWI.engine.propagation import MIN_MARGIN_BYTES, margin_bytes
    assert margin_bytes(10, 10, 0, 256) == MIN_MARGIN_BYTES  # small grids: the minimum
    assert margin_bytes(10, 1000, 30, 256) == -(-(38 * 1000 * 4) // 256) * 256  # right-well rows
    assert margin_bytes(20000, 10, 0, 512) == -(-(20008 * 4) // 512) * 512  # tnz values past a record


# ---------------------------------------------------------------------------------- construction

def test_kernels_come_from_the_program_of_the_acquisition_and_are_retrieved_once():
    engine = make_engine("surface")
    assert engine.knl.injSrc is engine.prg.injSrc and engine.knl.Adj_injSrc is engine.prg.Adj_injSrc
    retrieved = [entry[1] for entry in engine.backend.log if entry[0] == "kernel"]
    assert sorted(retrieved) == sorted(["injSrc", "Adj_injSrc", "MakeAllZero", "MakeGradZero", "update_velx",
                                        "update_velz", "update_tauz", "update_tauxz", "Adj_update_tau",
                                        "Adj_update_v", "Grad"])
    elastic = make_engine("elastic")  # acq_type 2: the injection kernels of elastic.cl
    assert "injSrc" not in vars(elastic.prg)


def test_buffers_live_inside_allocations_with_margins():
    engine = make_engine()
    assert engine.margins_changed() == []
    engine._allocations["vx_b"][0] = 5.0  # a write before the buffer
    assert engine.margins_changed() == ["vx_b"]
    assert engine.vx_b.size == TNZ * TNX and engine.seismogramid_vx_b.size == NT_EXT * NR


# ---------------------------------------------------------------------------------- forward

def test_forward_records_the_receivers_of_every_shot_at_every_injection():
    engine = make_engine()
    seismogram = engine.forward()
    for s in range(NS):
        for t in range(NT_EXT):
            before = t * DT_SCALE  # updates made before the injection of sample t
            offset = np.arange(NR) + 100 * SRCX[s]
            expected = {"vx": before * RHO_INVERSE, "vz": before * RHO_INVERSE, "taux": before * LAM,
                        "tauz": before * LAM, "tauxz": before * MU}
            for par, value in expected.items():
                inj = injected(before) if par != "tauxz" else 0
                np.testing.assert_allclose(seismogram[par][t, :, s], value + inj + offset, err_msg=f"{par} {s} {t}")


def test_forward_stores_the_field_after_every_checkpoint_step():
    engine = make_engine()
    engine.forward()
    for s in range(NS):
        cell = (SRCZ[s], SRCX[s])
        for c, step in enumerate(engine.chp):
            vx = engine.checkpoints["vx"][s, c]
            assert vx[0, 0] == (step + 1) * RHO_INVERSE
            assert vx[cell] == (step + 1) * RHO_INVERSE + injected(step)
            assert (engine.checkpoints["tauxz"][s, c] == (step + 1) * MU).all()


def test_forward_launch_sequence():
    engine = make_engine()
    engine.backend.log.clear()
    engine.forward()
    names = launches(engine)
    per_shot = ["MakeAllZero"]
    for t in range(NT):
        per_shot += (["injSrc"] if t % DT_SCALE == 0 else []) + ["update_velx", "update_velz", "update_tauz",
                                                                 "update_tauxz"]
    assert names == per_shot * NS
    copies = [entry for entry in engine.backend.log if entry[0] == "to_host"]
    # Five checkpoint copies per checkpoint and five seismograms per shot, none of them blocking.
    assert copies == [("to_host", False)] * (NS * (5 * engine.nchp + 5))
    assert engine.backend.log[-1] == ("finish",)


def test_update_kernels_get_the_coefficient_of_each_run():
    engine = make_engine()
    engine.forward()
    assert engine.knl.update_velx.args[0] == 1
    engine.prepare_gradient()
    engine.gradient(residual())
    assert engine.knl.update_velx.args[0] == -1
    engine.forward()
    assert engine.knl.update_velx.args[0] == 1


# ---------------------------------------------------------------------------------- gradient

def _rebuilt(cell_value):
    """The forward field the gradient loop rebuilds, step by step from NT - 1 down to 1.

    ``cell_value(step)`` is the forward field after ``step``: the loop takes it at the checkpoints
    and goes one update back (minus 1 / density) at every other step.
    """
    values, current = {}, None
    for t in range(NT - 1, 0, -1):
        current = cell_value(t) if t in (0, 4, 9) else current - RHO_INVERSE
        values[t] = current
    return values


def test_gradient_rebuilds_the_forward_field_from_the_checkpoints():
    engine = make_engine()
    engine.forward()
    engine.prepare_gradient()
    engine.gradient(residual())
    g_mu, _, g_rho, mu_precond, lam_precond, rho_precond = engine.read_gradient()
    # Away from the sources the updates are undone exactly; at a source the injections are not,
    # so the rebuilt field drifts between checkpoints, as in 0.1.10.
    away = sum(_rebuilt(lambda step: (step + 1) * RHO_INVERSE).values())
    assert g_mu[0, 0] == NS * away
    for s in range(NS):
        at_source = sum(_rebuilt(lambda step: (step + 1) * RHO_INVERSE + injected(step)).values())
        other_source = sum(_rebuilt(lambda step: (step + 1) * RHO_INVERSE).values())
        assert g_mu[SRCZ[s], SRCX[s]] == at_source + other_source
    assert (g_rho == NS * (NT - 1)).all() and (mu_precond == NS * (NT - 1)).all()
    assert (lam_precond == 0).all() and (rho_precond == 0).all()


def test_gradient_injects_the_residual_backwards_in_time():
    engine = make_engine()
    engine.forward()
    engine.prepare_gradient()
    res = residual()
    engine.gradient(res)
    g_lam = engine.read_gradient()[1]
    expected = 0.0
    for s in range(NS):
        adjoint, t_src = 0.0, NT_EXT
        for t in range(NT - 1, 0, -1):
            if t % DT_SCALE == 0:
                adjoint += res["vx"][t_src - 1, s * NR]  # the first receiver of the shot
                t_src -= 1
            expected += adjoint
    assert g_lam[0, 0] == pytest.approx(expected)


def test_gradient_launch_sequence():
    engine = make_engine()
    engine.forward()
    engine.prepare_gradient()
    engine.backend.log.clear()
    engine.gradient(residual())
    names = launches(engine)
    per_shot = ["MakeAllZero", "MakeAllZero"]
    for t in range(NT - 1, 0, -1):
        if t not in (4, 9):  # checkpoint steps restore the field instead of updating it
            per_shot += ["update_tauxz", "update_tauz", "update_velz", "update_velx"]
        per_shot += (["Adj_injSrc"] if t % DT_SCALE == 0 else []) + ["Adj_update_tau", "Adj_update_v", "Grad"]
    assert names == ["MakeGradZero"] + per_shot * NS
    restores = [entry for entry in engine.backend.log if entry == ("to_device", False)]
    assert len(restores) == NS * 5 * 2  # checkpoints 9 and 4 of each shot, without blocking
    uploads = [entry for entry in engine.backend.log if entry == ("to_device", True)]
    assert len(uploads) == NS * 5  # the residual of each shot, once
    samples = [entry[1] for entry in engine.backend.log if entry[0] == "copy"]
    row = NR * 4
    assert samples == [k * row for k in (3, 2, 1) for _ in range(5)] * NS  # samples 3, 2, 1, backwards


# ---------------------------------------------------------------------------------- the rest

def test_observers_see_every_step():
    engine = make_engine()
    forward, backward = [], []
    engine.forward(lambda s, t: forward.append((s, int(t))))
    assert forward == [(s, t) for s in range(NS) for t in range(NT)]
    engine.prepare_gradient()
    engine.gradient(residual(), lambda s, t, t_src: backward.append((s, t, int(t_src))))
    assert [entry[:2] for entry in backward] == [(s, t) for s in range(NS) for t in range(NT - 1, 0, -1)]
    assert backward[0][2] == NT_EXT - 1 and backward[NT - 2][2] == 1  # after the injections of steps 9 ... 3


def test_reading_fields_and_gradients():
    engine = make_engine()
    engine.forward()
    host = np.zeros((TNZ, TNX), np.float32)
    assert engine.read_field("tauxz_b", host) is host and (host == NT * MU).all()
    engine.prepare_gradient()
    engine.gradient(residual())
    _, _, g_rho, *ones = engine.read_gradient(preconditioners=False)
    assert all((array == 1).all() for array in ones) and (g_rho == NS * (NT - 1)).all()


def test_residual_helpers():
    engine = make_engine()
    res = residual()
    engine.make_residual(res, 1, 2)  # 0.1.10: sample 1 of shot 1 from the host
    np.testing.assert_allclose(engine.res_vx_b, res["vx"][1, NR:2 * NR])
    engine.residual_to_device(res, 0)
    engine.residual_sample(0)  # sample -1: the last one, as NumPy indexing did in 0.1.10
    np.testing.assert_allclose(engine.res_vz_b, res["vz"][-1, :NR])
    first = engine.res_shot_b["vx"]
    engine.residual_to_device(res, 1)
    assert engine.res_shot_b["vx"] is first  # same size: the buffer is reused


def test_the_unit_tests_alone_cover_the_engine(tmp_path):
    # Roadmap criterion of step 2.1: engine coverage above 90 % with these tests alone.
    env = {**os.environ, "COVERAGE_FILE": str(tmp_path / ".coverage")}
    here = os.path.dirname(__file__)
    run = subprocess.run([sys.executable, "-m", "coverage", "run", "--source=PyFWI.engine", "-m", "pytest", "-q",
                          "-p", "no:cacheprovider", os.path.join(here, "test_engine.py"), "-k", "not the_unit_tests_alone"],
                         cwd=os.path.dirname(here), env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    report = subprocess.run([sys.executable, "-m", "coverage", "report", "--fail-under=90"], cwd=os.path.dirname(here),
                            env=env, capture_output=True, text=True, check=False)
    assert report.returncode == 0, report.stdout
