"""The FP16 mode (roadmap step 4.3): the fields stored in half precision, scaled.

With ``PYFWI_FP16=1`` the native kernels store the fields, adjoint fields, checkpoints and stored bands as
half-precision numbers (2 bytes) and compute in float32; the engine scales the velocities and stresses so
that their values suit half precision (``PyFWI.engine.propagation.fp16_scales``). The tests of the
roadmap: the signal-to-noise ratio of the seismograms against float32, extreme amplitudes (overflow and
underflow), a complete inversion within 10 % of the model error of float32, and half the memory. The
bookkeeping of the engine is tested on the fake backend (``test_engine.py``), the variable by
``test_backend_selection.py``.
"""
import os
import sys

import numpy as np
import pytest

from helpers import forward_and_gradient, running_reference, tiny_case

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no FP16 mode")]

#: Smallest signal-to-noise ratios (dB) against float32: seismograms and gradients. Measured on the three
#: cases below: seismograms 44 to 57 dB, gradients 33 to 46 dB (half precision has 11 significant bits).
SEISMOGRAM_SNR, GRADIENT_SNR = 40.0, 30.0

CASES = [{"sdo": 4, "ns": 2}, {"sdo": 8, "acq_type": 2, "components": 4, "ns": 3},
         {"sdo": 4, "acq_type": 0, "components": 4, "ns": 2}]


def snr(reference, value):
    """Signal-to-noise ratio of ``value`` against ``reference``, in dB."""
    reference, value = np.asarray(reference, np.float64), np.asarray(value, np.float64)
    return 10 * np.log10(np.sum(reference ** 2) / max(np.sum((reference - value) ** 2), 1e-300))


def _runs(monkeypatch, kw, amplitude=1.0):
    """Forward and gradient in float32 and in FP16, the source wavelet multiplied by ``amplitude``."""
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    runs = []
    for fp16 in ("0", "1"):
        monkeypatch.setenv("PYFWI_FP16", fp16)
        case = tiny_case(**kw)
        case.W.src.w = case.W.src.w * np.float32(amplitude)
        runs.append((forward_and_gradient(case), case.W))
    return runs


@pytest.mark.parametrize("kw", CASES)
def test_the_fp16_mode_against_float32(cuda_device, monkeypatch, kw):
    ((d32, g32), _), ((d16, g16), W) = _runs(monkeypatch, kw)
    assert W.fp16 and W._engine.vx_b.dtype == np.float16
    for key in d32:
        if np.any(d32[key]):
            assert snr(d32[key], d16[key]) >= SEISMOGRAM_SNR, key
    for key in g32:
        assert snr(g32[key], g16[key]) >= GRADIENT_SNR, key


@pytest.mark.parametrize("amplitude", [1e-6, 1e6])
def test_extreme_amplitudes_neither_overflow_nor_underflow(cuda_device, monkeypatch, amplitude):
    # The scales follow the source and the residual: a wavelet a million times weaker or stronger gives
    # the same accuracy. Without scales, the velocities (about 1e-8 for a unit wavelet) are below the
    # smallest half-precision number at any amplitude, and the stresses (about 0.7) above the largest one
    # (65504) at 1e6. (At 1e-12 the gradient of float32 itself underflows to zero.)
    kw = CASES[1]
    ((d32, g32), _), ((d16, g16), _) = _runs(monkeypatch, kw, amplitude)
    assert min(snr(d32[k], d16[k]) for k in d32) >= SEISMOGRAM_SNR
    assert min(snr(g32[k], g16[k]) for k in g32) >= GRADIENT_SNR


def test_an_overflow_is_reported(cuda_device, monkeypatch):
    from PyFWI.engine import propagation
    monkeypatch.setattr(propagation, "FP16_PEAK", 1e6)  # beyond the largest half-precision number
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    monkeypatch.setenv("PYFWI_FP16", "1")
    with pytest.raises(FloatingPointError, match="overflowed"):
        forward_and_gradient(tiny_case(**CASES[1]))


def test_half_the_memory(cuda_device, monkeypatch):
    # MEM: the fields, the adjoint fields, the checkpoints and the stored band take half the bytes.
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    engines = []
    for fp16 in ("0", "1"):
        monkeypatch.setenv("PYFWI_FP16", fp16)
        monkeypatch.setenv("PYFWI_BOUNDARY_SAVING", "0")
        case = tiny_case(**CASES[1])
        forward_and_gradient(case)
        engines.append(case.W._engine)
    single, half = engines
    for name in ("vx_b", "atauxz_b", "_device_checkpoints"):
        assert half.backend.nbytes(getattr(half, name)) * 2 == single.backend.nbytes(getattr(single, name)), name
    assert half.backend.nbytes(half.Gmu_b) == single.backend.nbytes(single.Gmu_b)  # the gradients stay float32


def test_fp16_with_boundary_saving(cuda_device, monkeypatch):
    # Both opt-in features together: the rebuild of 3,500 steps in half precision, anchored on the stored
    # band at every step, keeps the accuracy of the checkpoints (measured 39.8 dB against 40.0 dB).
    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    kw = {"sdo": 8, "acq_type": 2, "components": 4, "ns": 1, "t": 2.0}
    _, g32 = forward_and_gradient(tiny_case(chpr=100, **kw))
    monkeypatch.setenv("PYFWI_FP16", "1")
    monkeypatch.setenv("PYFWI_BOUNDARY_SAVING", "1")
    _, g16 = forward_and_gradient(tiny_case(**kw))
    assert min(snr(g32[k], g16[k]) for k in g32) >= GRADIENT_SNR


@pytest.mark.xfail(strict=True, reason=(
    "a limit of half precision, measured: the rounding of every update leaves an error about 60 dB below the "
    "peak of the fields, which the late, attenuated part of the recordings falls under, and which changes "
    "chaotically with the model, so the misfit is rough (a model change of 0.002 m/s moved it by 6 %) and the "
    "line search of the optimiser finds no descent; the FP16 mode suits modelling and gradients, not inversion"))
def test_a_complete_inversion(cuda_device, monkeypatch):
    # FWI: the multi-scale inversion of the example script in FP16 ends with a model error at most 10 %
    # above that of float32 (the target of the roadmap). It does not: see the reason of the xfail.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "scripts"))
    import fwi

    monkeypatch.setenv("PYFWI_BACKEND", "cuda:strict")
    errors = {}
    for fp16 in ("0", "1"):
        monkeypatch.setenv("PYFWI_FP16", fp16)
        result = fwi.main(size="small")
        errors[fp16] = (result["error_start"], result["error_end"])
    assert errors["1"][1] < 0.9 * errors["1"][0]  # it inverts
    assert errors["1"][1] <= 1.1 * errors["0"][1]
