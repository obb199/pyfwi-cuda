"""The public API on both backends (roadmap step 2.5, test type API).

The public objects of PyFWI keep their signatures and return values on every backend of the
fork: ``WavePropagator`` (forward modelling, gradient, ``W.seismogram``), the ``show=True`` path
that plots the wavefield during the time loops, and the formats of the seismic sections. Every
test takes the ``backend`` fixture of ``conftest.py`` and runs on OpenCL and on CUDA. The OpenCL
case also runs in the reference environment, against the published 0.1.10, so what is checked
here is what 0.1.10 does.

The rest of the API suite is parametrised the same way in its own files: ``W.W`` and the
checkpoints (``test_checkpoints.py``), the example scripts with their physical checks, including
``torchfwi.Fwi`` with CPU tensors (``test_examples.py``), and the original example script of
0.1.10 (``test_release.py``).

One difference is documented: the CUDA backend has no OpenCL context or queue (``W.ctx`` and
``W.queue`` are None); its buffers and kernels are CuPy arrays and CUDA kernels under the same
attribute names (``W.vx_b``, ``W.knl``...).

The published 0.1.10 runs inside zero guard bands (``test/oob_guard.py``), as in the other test
files: they give its out-of-bounds accesses a defined result and keep its OpenCL contexts alive
(releasing one while another exists breaks the NVIDIA driver for the rest of the process).
"""
import contextlib

import numpy as np
import pytest

from helpers import running_reference, tiny_case

#: The recorded components of each value of ``components`` (``PyFWI.acquisition.seismic_section``);
#: 0 records the pressure (taux + tauz) / 2, returned under both names.
SECTIONS = {0: ["taux", "tauz"], 1: ["taux"], 2: ["vx", "vz"], 3: ["taux", "tauz", "tauxz"],
            4: ["vx", "vz", "taux", "tauz", "tauxz"]}


@pytest.fixture(autouse=True)
def guard_bands_for_0_1_10():
    """Run every test of this file inside zero guard bands in the reference environment only."""
    from oob_guard import zero_guard_bands

    with zero_guard_bands() if running_reference() else contextlib.nullcontext():
        yield


@pytest.mark.parametrize("components", sorted(SECTIONS))
def test_forward_modelling_returns_the_sections_of_0_1_10(backend, components):
    case = tiny_case(ns=2, components=components)
    W = case.W
    data = W.forward_modeling(case.model, False)
    nt, nr, ns = int(W.nt_ext), int(W.nr), int(W.ns)
    assert sorted(data) == sorted(SECTIONS[components])
    for name, section in data.items():  # the default shape: "2d", every shot side by side
        assert section.shape == (nt, nr * ns) and section.dtype == np.float32, name
        assert np.abs(section).max() > 0, name
    # W.seismogram keeps the five components of every shot, (nt_ext, nr, ns), from which the
    # section is made (Fortran order: the shots side by side).
    assert sorted(W.seismogram) == sorted(SECTIONS[4])
    for name, seismogram in W.seismogram.items():
        assert seismogram.shape == (nt, nr, ns) and seismogram.dtype == np.float32, name
    if components == 4:
        for name in data:
            np.testing.assert_array_equal(data[name], np.reshape(W.seismogram[name], (nt, nr * ns), "F"))


def test_three_dimensional_sections(backend):
    case = tiny_case(ns=2, components=1, inpa_extra={"seimogram_shape": "3d"})  # PyFWI's spelling of the key
    data = case.W.forward_modeling(case.model, False)
    assert data["taux"].shape == (int(case.W.nt_ext), int(case.W.nr), 2)
    np.testing.assert_array_equal(data["taux"], case.W.seismogram["taux"])


@pytest.mark.parametrize("parameterization, names", [("dv", ["vp", "vs", "rho"]), ("lmd", ["lam", "mu", "rho"])])
def test_gradient_returns_maps_of_the_model(backend, parameterization, names):
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process

    case = tiny_case()
    d_obs = process.prepare_residual(case.W.forward_modeling(case.model, False), 1)
    d_est = process.prepare_residual(case.W.forward_modeling(case.m0, False), 1)
    _, adjoint_source = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))
    gradient = case.W.gradient(adjoint_source, False, parameterization=parameterization)
    assert sorted(gradient) == sorted(names)
    for name, value in gradient.items():
        assert value.shape == case.model["vp"].shape and value.dtype == np.float32, name
        assert np.isfinite(value).all() and np.abs(value).max() > 0, name


@pytest.fixture
def plots(monkeypatch):
    """Draw with the non-interactive Agg backend, without the 0.1 s pauses; returns the list of pauses.

    ``WavePropagator.plot_propagation`` updates its figure and calls ``plt.pause(0.1)`` every 20
    time steps; the figures are really drawn, only the waits are skipped (and counted).
    """
    import matplotlib.pyplot as plt

    import PyFWI.wave_propagation as wave

    plt.switch_backend("Agg")
    pauses = []
    monkeypatch.setattr(wave.plt, "pause", lambda interval: pauses.append(interval))
    yield pauses
    plt.close("all")


def test_show_path_plots_the_wavefield_and_changes_nothing(backend, plots):
    import PyFWI.fwi_tools as tools
    import PyFWI.processing as process

    case = tiny_case()
    W, shown = case.W, []
    draw = W.plot_propagation

    def plot_propagation(wave1, t, wave2=None):
        shown.append((float(np.abs(wave1).max()), None if wave2 is None else float(np.abs(wave2).max())))
        draw(wave1, t, wave2)
    W.plot_propagation = plot_propagation

    plain = W.forward_modeling(case.model, False)
    with_plots = W.forward_modeling(case.model, True)
    # One plot every 20 internal steps of each shot (after the step), of a wavefield that grows.
    assert len(shown) == len(plots) == len(range(0, int(W.nt), 20))
    assert 0 < shown[0][0] < shown[-1][0]
    for name in plain:
        np.testing.assert_array_equal(with_plots[name], plain[name], err_msg=name)

    d_obs = process.prepare_residual(plain, 1)
    d_est = process.prepare_residual(W.forward_modeling(case.m0, False), 1)
    _, adjoint_source = tools.cost_seismic(d_est, d_obs, fun=tools.CostFunction("l2"))
    gradient = W.gradient(adjoint_source, False)
    shown.clear()
    with_plots = W.gradient(adjoint_source, True)
    # The backward run plots the rebuilt forward field and the adjoint field side by side.
    assert shown and any(adjoint > 0 for _, adjoint in shown)
    for name in gradient:
        np.testing.assert_array_equal(with_plots[name], gradient[name], err_msg=name)


def test_opencl_handles_exist_only_on_opencl(backend):
    W = tiny_case().W
    if backend == "opencl":
        import pyopencl as cl
        assert isinstance(W.ctx, cl.Context) and isinstance(W.queue, cl.CommandQueue)
    else:
        assert W.ctx is None and W.queue is None
        import cupy
        assert isinstance(W.vx_b, cupy.ndarray)
