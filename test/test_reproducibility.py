"""Reproducibility of PyFWI runs: the properties the golden masters of Phase 0 rely on.

PyFWI 0.1.10 has two sources of irreproducibility (see ``test_known_bugs.py``):
out-of-bounds buffer accesses, whose results depend on the memory next to each
buffer, and race conditions at some source/receiver positions. The golden masters
are therefore produced under two rules:

1. every run happens inside **zero guard bands** (``oob_guard.zero_guard_bands``),
   which gives the out-of-bounds accesses a defined result;
2. the acquisition is **race-free** (``helpers.race_free_geometry``).

Under those rules a run must be deterministic, and NVIDIA OpenCL and POCL (OpenCL
on the CPU) must agree bit for bit, which lets the CPU CI check results against
golden masters made on the GPU. For the surface acquisition the rules change
nothing: a plain run of 0.1.10 gives the same bits.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import _opencl_platforms, _pyopencl_ctx
from helpers import forward_and_gradient, tiny_case
from oob_guard import zero_guard_bands

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _guarded_run(acq_type, sdo):
    """Forward + gradient of a race-free tiny case inside zero guard bands.

    Returns a flat dict ``{"d_<component>": seismogram, "g_<parameter>": gradient}``.
    """
    with zero_guard_bands():
        d, g = forward_and_gradient(tiny_case(acq_type=acq_type, sdo=sdo, components=4, ns=2))
    return {**{f"d_{k}": v for k, v in d.items()}, **{f"g_{k}": v for k, v in g.items()}}


def _assert_same_bits(a, b):
    """Assert that two result dicts have the same keys and bit-identical arrays."""
    assert a.keys() == b.keys()
    different = [k for k in a if not np.array_equal(a[k], b[k])]
    assert not different, f"arrays that differ: {different}"


@pytest.mark.parametrize("acq_type", [0, 1, 2])
@pytest.mark.parametrize("sdo", [4, 8])
def test_guarded_race_free_runs_are_deterministic(opencl_platform, acq_type, sdo):
    _assert_same_bits(_guarded_run(acq_type, sdo), _guarded_run(acq_type, sdo))


@pytest.mark.opencl
@pytest.mark.parametrize("acq_type", [0, 1, 2])
def test_nvidia_and_pocl_agree_bit_for_bit(monkeypatch, acq_type):
    platforms = _opencl_platforms()
    nvidia = next((p for p in platforms if "NVIDIA" in p[1]), None)
    pocl = next((p for p in platforms if "Portable" in p[1]), None)
    if nvidia is None or pocl is None:
        pytest.skip("needs both NVIDIA OpenCL and POCL")
    results = []
    for index, _, n_devices in (nvidia, pocl):
        monkeypatch.setenv("PYOPENCL_CTX", _pyopencl_ctx(index, n_devices))
        results.append(_guarded_run(acq_type, sdo=8))
    _assert_same_bits(*results)


#: Program run in a fresh process by the next test: a plain run (no guard bands) of the
#: surface case, saved to the .npz file given as first argument. A fresh process matters:
#: plain runs of 0.1.10 can crash the NVIDIA kernel after a few runs in the same process.
PLAIN_RUN = """
import sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = [sys.argv[2], sys.argv[3]]
import numpy as np
from helpers import forward_and_gradient, tiny_case
d, g = forward_and_gradient(tiny_case(acq_type=1, sdo=8, components=4, ns=2))
np.savez(sys.argv[1], **{f"d_{k}": v for k, v in d.items()}, **{f"g_{k}": v for k, v in g.items()})
"""


@pytest.mark.opencl
def test_plain_surface_run_equals_guarded_run(nvidia_opencl_platform, monkeypatch, tmp_path):
    monkeypatch.setenv("PYOPENCL_CTX", str(nvidia_opencl_platform))
    out = tmp_path / "plain.npz"
    subprocess.run([sys.executable, "-c", PLAIN_RUN, str(out), os.path.join(REPO, "test"), REPO],
                   check=True, capture_output=True, text=True, timeout=300)
    plain = dict(np.load(out))
    _assert_same_bits(plain, _guarded_run(acq_type=1, sdo=8))
