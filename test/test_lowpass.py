"""The low-pass filter of the multi-scale inversion (roadmap step 1.6): ``fwi_tools.lowpass`` and ``adj_lowpass``.

Multi-scale FWI (``FWI(..., freqs=[...])``) filters observed and predicted data with a zero-phase
Butterworth low-pass filter before the misfit, and the residual with its adjoint before the
gradient. PyFWI 0.1.10 applied the filter in the frequency domain as products with ``(n, n)``
diagonal matrices (``np.diag(h) @ X``): O(n^2) per trace. Since step 1.6 the spectrum is multiplied
element by element. The versions of 0.1.10 are kept below, verbatim, as the reference.
"""
import time

import numpy as np
import pytest
from numpy import fft
from scipy.signal import butter, freqz

from helpers import running_reference
from PyFWI import fwi_tools

#: Nyquist frequency of the tests (Hz): 4 ms sampling.
FN = 125.0


def lowpass_0_1_10(x1, highcut, fn, order=1, axis=1):
    """``fwi_tools.lowpass`` of PyFWI 0.1.10 (without its unused ``show`` argument and deep copy)."""
    padding = 512
    x = np.hstack((x1, np.zeros((x1.shape[0], padding, x1.shape[2]))))
    nt = x.shape[axis]
    x_fft = fft.fft(x, n=nt, axis=axis)
    b, a = butter(order, 2 * highcut / fn, btype='lowpass', output="ba")
    _, h1 = freqz(b, a, worN=nt, whole=True)
    fd = np.diag(h1) @ x_fft
    fd = np.diag(np.conjugate(h1)) @ fd
    return fft.ifft(fd, n=nt, axis=axis).real[:, :-padding, :]


def adj_lowpass_0_1_10(x, highcut, fn, order, axis=1):
    """``fwi_tools.adj_lowpass`` of PyFWI 0.1.10."""
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))
    nt = x.shape[axis]
    x_fft = np.fft.fft(x, n=nt, axis=axis)
    b, a = butter(order, 2 * highcut / fn, btype='lowpass', output="ba")
    _, h = freqz(b, a, worN=nt, whole=True)
    fd = np.diag(np.conjugate(h)) @ x_fft
    fd = np.diag(h) @ fd
    return np.fft.ifft(fd, axis=axis).real[:, :-padding, :]


def _data(nt, traces, dtype, seed=0):
    """Random data of the shape the inversion filters: (5 components, nt samples, traces)."""
    return np.random.default_rng(seed).standard_normal((5, nt, traces)).astype(dtype)


def _relative(new, old):
    """Largest difference relative to the largest value of the reference."""
    return np.abs(new - old).max() / np.abs(old).max()


@pytest.mark.parametrize("nt, traces, order, highcut", [
    (151, 400, 3, 10.0),   # small scenario of the benchmark, order used by FWI
    (51, 7, 1, 40.0),
    (300, 1, 2, 5.0),
    (751, 20, 5, 25.0),
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_same_result_as_0_1_10(nt, traces, order, highcut, dtype):
    x = _data(nt, traces, dtype)
    new, old = fwi_tools.lowpass(x, highcut, FN, order=order, axis=1), lowpass_0_1_10(x, highcut, FN, order=order)
    assert new.shape == old.shape == x.shape and new.dtype == old.dtype == np.float64
    assert _relative(new, old) <= 1e-12
    new, old = fwi_tools.adj_lowpass(x, highcut, FN, order, axis=1), adj_lowpass_0_1_10(x, highcut, FN, order)
    assert _relative(new, old) <= 1e-12


@pytest.mark.xfail(condition=running_reference(), strict=True, raises=ValueError,
                   reason="0.1.10 pads along axis 1 whatever the axis: other axes fail in the matrix product")
def test_any_time_axis():
    # 0.1.10 padded along axis 1 whatever the axis, so only axis=1 worked; now any axis does.
    x = _data(120, 6, np.float64)
    along_1 = fwi_tools.lowpass(x, 15.0, FN, order=3, axis=1)
    along_0 = fwi_tools.lowpass(np.moveaxis(x, 1, 0), 15.0, FN, order=3, axis=0)
    np.testing.assert_allclose(np.moveaxis(along_0, 0, 1), along_1, rtol=0, atol=1e-12 * np.abs(along_1).max())


def test_the_input_is_not_modified():
    x = _data(101, 5, np.float32)
    before = x.copy()
    fwi_tools.lowpass(x, 10.0, FN, order=3, axis=1)
    fwi_tools.adj_lowpass(x, 10.0, FN, 3, axis=1)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("order", [1, 3])
def test_adjoint_identity(order):
    # <lowpass x, y> = <x, adj_lowpass y>: what makes the gradient of the filtered misfit correct.
    x, y = _data(201, 9, np.float64, seed=1), _data(201, 9, np.float64, seed=2)
    left = np.vdot(fwi_tools.lowpass(x, 12.0, FN, order=order, axis=1), y)
    right = np.vdot(x, fwi_tools.adj_lowpass(y, 12.0, FN, order, axis=1))
    assert abs(left - right) <= 1e-12 * abs(left)


def test_the_filter_keeps_low_and_removes_high_frequencies():
    t = np.arange(400) * 0.004
    signal = np.sin(2 * np.pi * 3 * t) + np.sin(2 * np.pi * 60 * t)
    x = np.tile(signal[None, :, None], (5, 1, 2))
    filtered = fwi_tools.lowpass(x, 10.0, FN, order=3, axis=1)[0, 50:350, 0]  # away from the ends
    np.testing.assert_allclose(filtered, np.sin(2 * np.pi * 3 * t[50:350]), atol=0.02)


@pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has the O(n^2) filter")
def test_element_wise_filter_is_much_faster():
    # DES: O(n) instead of O(n^2) per trace. The FFT now dominates the cost.
    x = _data(1000, 50, np.float32)

    def seconds(function):
        start = time.perf_counter()
        function(x, 10.0, FN, order=3)
        return time.perf_counter() - start

    seconds(fwi_tools.lowpass)  # warm-up
    old = min(seconds(lowpass_0_1_10) for _ in range(2))
    new = min(seconds(fwi_tools.lowpass) for _ in range(3))
    assert old / new > 5, (old, new)
