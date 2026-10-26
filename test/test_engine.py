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
0, 4 and 9 (``chpr`` = 30). Unless a test says otherwise, the engine runs in the ``legacy`` numeric
mode; the tests of the ``fixed`` mode (step 2.4) are at the end, with ``internal_source``.
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


def make_engine(injection="surface", chpr=30, numerics="legacy", nt_ext=NT_EXT):
    """An engine on the fake backend, with a model (1 / rho = 1, lambda = 2, mu = 3) and a boundary.

    ``nt_ext`` (rows of the seismograms) can differ from the 4 recorded steps, as rounding makes it
    in PyFWI for some ``dt`` and ``t``.
    """
    from PyFWI.engine.propagation import PropagationEngine, checkpoint_steps
    backend = FakeBackend(nr=NR, tnx=TNX)
    sources = {"elastic": "elastic source", "crosswell": "crosswell source", "surface": "surface source"}
    engine = PropagationEngine(backend, sources, injection, tnz=TNZ, tnx=TNX, ns=NS, nr=NR, nt=NT, nt_ext=nt_ext,
                               dt_scale=DT_SCALE, chp=checkpoint_steps(chpr, NT), dxr=np.int32(1), srcx=SRCX,
                               srcz=SRCZ, source=source, rec_top_right_var=0, numerics=numerics)
    field = np.ones((TNZ, TNX), np.float32)
    engine.set_pml(0 * field, 0 * field)
    engine.set_model(RHO_INVERSE * field, MU * field, LAM * field)
    return engine


def residual(nt_ext=NT_EXT):
    """Residual of both shots: sample k of shot s at receiver r is (k + 1) (s + 1) + r / 10."""
    samples = np.arange(nt_ext)[:, None] + 1
    columns = [(s + 1) * samples + np.arange(NR) / 10 for s in range(NS)]
    one = np.hstack(columns).astype(np.float32)  # (NT_EXT, NR * NS)
    return {par: one.copy() for par in COMPONENTS}


def injected(step):
    """Total of the source samples injected at the source cell up to internal step ``step`` included."""
    return sum(source(k)[0] for k in range(NT_EXT) if k * DT_SCALE <= step)


def launches(engine):
    """Names of the kernels launched, in order, from the log of the fake backend."""
    return [entry[1] for entry in engine.backend.log if entry[0] == "launch"]


def recorded_rows(engine):
    """A list that receives the seismogram row (last argument) of every injSrc launch of ``engine``."""
    rows, play = [], engine.knl.injSrc.function

    def inject(backend, *args):
        rows.append(int(args[-1]))
        play(backend, *args)
    engine.knl.injSrc.function = inject
    return rows


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


@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
def test_gradient_injects_the_residual_backwards_in_time(numerics):
    # Here nt_ext is the number of recorded steps: both modes inject the same samples, at the same
    # steps. Grad adds the adjoint field: in the legacy mode after the injection of the step, in the
    # fixed mode before it (the residual of step t goes into the adjoint field of step t, which
    # the gradient of step t - 1 uses).
    engine = make_engine(numerics=numerics)
    engine.forward()
    engine.prepare_gradient()
    res = residual()
    engine.gradient(res)
    g_lam = engine.read_gradient()[1]
    expected = 0.0
    for s in range(NS):
        adjoint, t_src = 0.0, NT_EXT
        for t in range(NT - 1, 0, -1):
            if numerics == "fixed":
                expected += adjoint
            if t % DT_SCALE == 0:
                adjoint += res["vx"][t_src - 1, s * NR]  # the first receiver of the shot
                t_src -= 1
            if numerics == "legacy":
                expected += adjoint
    assert g_lam[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
def test_gradient_launch_sequence(numerics):
    engine = make_engine(numerics=numerics)
    engine.forward()
    engine.prepare_gradient()
    engine.backend.log.clear()
    engine.gradient(residual())
    names = launches(engine)
    per_shot = ["MakeAllZero", "MakeAllZero"]
    for t in range(NT - 1, 0, -1):
        if t not in (4, 9):  # checkpoint steps restore the field instead of updating it
            per_shot += ["update_tauxz", "update_tauz", "update_velz", "update_velx"]
        injection = ["Adj_injSrc"] if t % DT_SCALE == 0 else []
        if numerics == "legacy":
            per_shot += injection + ["Adj_update_tau", "Adj_update_v", "Grad"]
        else:  # the exact adjoint of the forward step (PropagationEngine._adjoint_step)
            per_shot += ["Grad", "Adj_update_v", "Adj_update_tau"] + injection
    assert names == ["MakeGradZero"] + per_shot * NS
    restores = [entry for entry in engine.backend.log if entry == ("to_device", False)]
    assert len(restores) == NS * 5 * 2  # checkpoints 9 and 4 of each shot, without blocking
    uploads = [entry for entry in engine.backend.log if entry == ("to_device", True)]
    assert len(uploads) == NS * 5  # the residual of each shot, once
    samples = [entry[1] for entry in engine.backend.log if entry[0] == "copy"]
    row = NR * 4
    assert samples == [k * row for k in (3, 2, 1) for _ in range(5)] * NS  # samples 3, 2, 1, backwards


# ---------------------------------------------------------------------------------- the rest

@pytest.mark.parametrize("numerics", ["legacy", "fixed"])
def test_observers_see_every_step(numerics):
    engine = make_engine(numerics=numerics)
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


# ---------------------------------------------------------------------------------- fixed mode

def _band_limited(time):
    """A smooth wavelet with no energy near 125 Hz: a 15 Hz cosine under a Gaussian centred on 0.15 s.

    It starts at 1e-11: nothing of its interpolation before time 0 is lost.
    """
    return np.cos(2 * np.pi * 15 * (time - 0.15)) * np.exp(-((time - 0.15) / 0.03) ** 2)


def _sampled(k):
    """The five samples of ``_band_limited`` at recorded sample ``k`` (dt = 4 ms)."""
    return [_band_limited(k * 0.004)] * 5


def test_internal_source_without_resampling_is_the_source():
    # dt_scale = 1 (a user dt of 0.6 ms or less): both modes inject the samples as they are.
    from PyFWI.engine.propagation import internal_source
    expected = np.array([source(k) for k in range(NT)], np.float32)
    np.testing.assert_array_equal(internal_source(source, NT, 1), expected)


def test_internal_source_is_the_wavelet_at_every_internal_step():
    # dt = 4 ms, 7 internal steps per sample (the examples). Times dt_scale, the source of every
    # step is the wavelet at that time; at the recorded steps, the recorded samples.
    from PyFWI.engine.propagation import internal_source
    scale, nt = 7, 700
    steps = internal_source(_sampled, nt, scale)
    assert steps.shape == (nt, 5) and steps.dtype == np.float32
    exact = _band_limited(np.arange(nt) * 0.004 / scale)
    assert np.abs(steps[:, 2] * scale - exact).max() < 1e-5
    # Spread over the steps, the source of each recorded interval is the same as in 0.1.10.
    total = sum(_sampled(k)[2] for k in range(nt // scale + 1))
    assert float(steps[:, 2].astype(np.float64).sum()) == pytest.approx(total, rel=1e-6)


def test_internal_source_has_no_images_of_the_wavelet():
    # The impulse train of 0.1.10 (sample k at step 7 k) repeats the spectrum of the wavelet around
    # every multiple of 1 / dt = 250 Hz: the grid noise of the known bug. The resampled source keeps
    # it below the recorded Nyquist frequency (125 Hz).
    from PyFWI.engine.propagation import internal_source
    scale, nt = 7, 1400
    resampled = internal_source(_sampled, nt, scale)[:, 2].astype(np.float64)
    train = np.zeros(nt)
    train[::scale] = [_sampled(k)[2] for k in range(len(train[::scale]))]
    above = np.fft.rfftfreq(nt, 0.004 / scale) > 1.3 * 125.0
    for signal, largest in ((resampled, 1e-5), (train, None)):
        spectrum = np.abs(np.fft.rfft(signal))
        ratio = spectrum[above].max() / spectrum.max()
        assert ratio < largest if largest else ratio > 0.5


def test_fixed_forward_injects_at_every_step_and_records_at_the_recorded_steps():
    from PyFWI.engine.propagation import internal_source
    engine = make_engine(numerics="fixed")
    rows = recorded_rows(engine)
    engine.backend.log.clear()
    seismogram = engine.forward()
    per_shot = ["MakeAllZero"] + ["injSrc", "update_velx", "update_velz", "update_tauz", "update_tauxz"] * NT
    assert launches(engine) == per_shot * NS
    # The receivers at steps 0, 3, 6 and 9 (rows 0 to 3), nothing in between.
    assert rows == [t // DT_SCALE if t % DT_SCALE == 0 else -1 for t in range(NT)] * NS
    # At the source, vx gets 1 / rho per step and the resampled source of every step so far.
    added = np.cumsum(internal_source(source, NT, DT_SCALE)[:, 0].astype(np.float64))
    for s in range(NS):
        for t in range(NT_EXT):
            before = t * DT_SCALE
            expected = before * RHO_INVERSE + added[before] + np.arange(NR) + 100 * SRCX[s]
            np.testing.assert_allclose(seismogram["vx"][t, :, s], expected, rtol=1e-6, err_msg=f"{s} {t}")


def test_fixed_forward_records_only_rows_that_exist():
    # Rounding can give fewer rows than recorded steps (dt = 5 ms, t = 0.25 s: 50 rows, 51 steps).
    # Here 3 rows for steps 0, 3, 6 and 9: 0.1.10 records step 9 in row 3, past the end of the
    # seismograms; the fixed mode does not record it.
    legacy, fixed = make_engine(numerics="legacy", nt_ext=3), make_engine(numerics="fixed", nt_ext=3)
    legacy_rows, fixed_rows = recorded_rows(legacy), recorded_rows(fixed)
    legacy.forward()
    fixed.forward()
    assert legacy_rows == [0, 1, 2, 3] * NS
    assert sorted(legacy.margins_changed()) == sorted(f"seismogramid_{par}_b" for par in COMPONENTS)
    assert max(fixed_rows) == 2 and fixed_rows.count(-1) == NS * (NT - 3)
    assert fixed.margins_changed() == []


@pytest.mark.parametrize("numerics, samples", [("legacy", (4, 3, 2)), ("fixed", (3, 2, 1))])
def test_gradient_injects_each_residual_sample_where_it_was_recorded(numerics, samples):
    # Rounding can give one row more than recorded steps (dt = 4 ms, t = 0.4 s): here 5 rows for
    # samples 0 to 3. 0.1.10 counts down from row 4 at step 9, where sample 3 was recorded: every
    # sample is injected one recorded sample too early. The fixed mode injects 3, 2, 1 at 9, 6, 3.
    engine = make_engine(numerics=numerics, nt_ext=5)
    engine.forward()
    engine.prepare_gradient()
    engine.backend.log.clear()
    engine.gradient(residual(nt_ext=5))
    copied = [entry[1] for entry in engine.backend.log if entry[0] == "copy"]
    assert copied == [k * NR * 4 for k in samples for _ in range(5)] * NS


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
