"""The cost function, its filters and the smoothing on the GPU (roadmap step 4.1, ``PyFWI._device_arrays``).

``PYFWI_CUDA_COST=1`` moves the L2 cost, the multi-scale low-pass filters and the Gaussian smoothing of
the gradient to the GPU. Test type UNI: the results agree with those of the host within 1e-6 relative
(the roadmap's tolerance), on data shaped as those of an inversion; FWI: a multi-scale inversion with
the variable gives the result of the default within the tolerance of the optimiser's path.
"""
import sys

import numpy as np
import pytest

from helpers import running_reference

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no GPU cost function")

#: Relative tolerance of the roadmap for the cost and the adjoint residual on the GPU.
TOLERANCE = 1e-6


def relative(expected, actual):
    """``||actual - expected|| / ||expected||`` in float64."""
    e, a = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    return float(np.linalg.norm(a - e) / np.linalg.norm(e))


def data(seed, samples=151, traces=120):
    """Seismic-looking data ``{component: (samples, traces)}``: random wavelets, float32."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples)[:, None]
    arrivals = rng.uniform(20, samples - 20, traces)
    traces_ = np.exp(-((t - arrivals) / 4.0) ** 2) * np.cos((t - arrivals) / 2.0) * rng.uniform(0.5, 2, traces)
    return {c: (k + 1) * traces_.astype(np.float32) for k, c in enumerate(("vx", "vz", "taux", "tauz", "tauxz"))}


def test_the_variable_is_off_by_default_and_checked():
    from PyFWI._device_arrays import cost_on_gpu
    assert cost_on_gpu({}) is False and cost_on_gpu({"PYFWI_CUDA_COST": " 0 "}) is False
    assert cost_on_gpu({"PYFWI_CUDA_COST": ""}) is False
    for wrong in ("yes", "2", "true"):
        with pytest.raises(ValueError, match="PYFWI_CUDA_COST must be 0 or 1"):
            cost_on_gpu({"PYFWI_CUDA_COST": wrong})


def test_the_gpu_needs_cupy(monkeypatch):
    from PyFWI._device_arrays import cost_on_gpu
    monkeypatch.setitem(sys.modules, "cupy", None)  # its import fails
    with pytest.raises(RuntimeError, match="needs CuPy"):
        cost_on_gpu({"PYFWI_CUDA_COST": "1"})


def test_numpy_arrays_stay_on_the_host():
    from PyFWI._device_arrays import array_module, gaussian_filter, to_host
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert array_module(x) is np and to_host(x) is x
    np.testing.assert_array_equal(gaussian_filter(x, 1), __import__("scipy.ndimage").ndimage.gaussian_filter(x, 1))


@pytest.mark.gpu
@pytest.mark.parametrize("freq", [None, 12.0])
@pytest.mark.parametrize("weights", [False, True])
def test_the_l2_cost_and_its_residual_agree_with_the_host(cuda_device, monkeypatch, freq, weights):
    import PyFWI.fwi_tools as tools

    predicted, observed = data(1), data(2)
    params_oh = np.linspace(0.5, 1.5, 120, dtype=np.float32) if weights else None
    kwargs = {"fun": tools.CostFunction("l2"), "fn": 125.0, "freq": freq, "order": 3, "axis": 1,
              "params_oh": params_oh}
    monkeypatch.setenv("PYFWI_CUDA_COST", "0")
    host_cost, host_residual = tools.cost_seismic(predicted, observed, **kwargs)
    monkeypatch.setenv("PYFWI_CUDA_COST", "1")
    gpu_cost, gpu_residual = tools.cost_seismic(predicted, observed, **kwargs)
    assert isinstance(gpu_cost, np.float64) and abs(gpu_cost - host_cost) <= TOLERANCE * host_cost
    assert gpu_residual.keys() == host_residual.keys()
    for key in host_residual:
        assert type(gpu_residual[key]) is np.ndarray and gpu_residual[key].dtype == np.float32
        assert relative(host_residual[key], gpu_residual[key]) <= TOLERANCE, key


@pytest.mark.gpu
def test_the_low_pass_filter_agrees_with_the_host(cuda_device):
    import cupy

    from PyFWI.fwi_tools import lowpass
    x = np.array(list(data(3).values()))
    host = lowpass(x, 20.0, 125.0, order=3, axis=1)
    gpu = lowpass(cupy.asarray(x), 20.0, 125.0, order=3, axis=1)
    # On the GPU in float32 (FP64 is 64 times slower on GeForce cards): 1.1e-7 measured.
    assert isinstance(gpu, cupy.ndarray) and gpu.dtype == np.float32 and relative(host, gpu.get()) <= 5e-7


@pytest.mark.gpu
def test_other_cost_functions_stay_on_the_host(cuda_device, monkeypatch):
    import PyFWI.fwi_tools as tools

    predicted, observed = data(4), data(5)
    results = []
    for value in ("0", "1"):
        monkeypatch.setenv("PYFWI_CUDA_COST", value)
        results.append(tools.cost_seismic(predicted, observed, fun=tools.CostFunction("l1")))
    assert results[0][0] == results[1][0]
    for key in results[0][1]:
        np.testing.assert_array_equal(results[0][1][key], results[1][1][key])


@pytest.mark.gpu
@pytest.mark.parametrize("sigma", [0, 2])
def test_the_smoothing_agrees_with_the_host(cuda_device, monkeypatch, sigma):
    from scipy.ndimage import gaussian_filter as host_filter

    from PyFWI._device_arrays import gaussian_filter
    rng = np.random.default_rng(6)
    gradient = rng.normal(size=(60, 80)).astype(np.float32)
    monkeypatch.setenv("PYFWI_CUDA_COST", "1")
    smoothed = gaussian_filter(gradient, sigma)
    assert type(smoothed) is np.ndarray and smoothed.dtype == np.float32
    assert relative(host_filter(gradient, sigma), smoothed) <= TOLERANCE


@pytest.mark.gpu
def test_an_inversion_on_the_gpu_gives_the_result_of_the_host(cuda_device, monkeypatch):
    # FWI: the multi-scale inversion of the example script (vp, L-BFGS, two stages with low-pass
    # filters, gradient smoothing) with the cost on the GPU and on the host.
    sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "example", "scripts"))
    import fwi

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    runs = {}
    for value in ("0", "1"):
        monkeypatch.setenv("PYFWI_CUDA_COST", value)
        runs[value] = fwi.main(size="small")
    host, gpu = runs["0"], runs["1"]
    assert gpu["error_end"] < 0.9 * gpu["error_start"]  # the inversion works
    assert relative(host["estimate"]["vp"], gpu["estimate"]["vp"]) <= 1e-4
    assert gpu["misfit"] == pytest.approx(host["misfit"], rel=1e-4)
