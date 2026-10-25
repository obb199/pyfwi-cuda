"""Optional dependencies: PyFWI must import without them and explain how to get them.

``segyio`` is only needed to read SEG-Y files (``seismic_io.read_segy`` and the Marmousi model of
``model_dataset``). Since step 1.1 it is imported by those functions only, so the rest of PyFWI
works without it. In the reference environment (the published 0.1.10) these tests are skipped:
there segyio is still imported at module level (``test_known_bugs.py`` records that bug).
"""
import importlib
import sys

import numpy as np
import pytest

from helpers import running_reference

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 imports segyio at module level")


@pytest.fixture
def without_segyio(monkeypatch):
    """Make ``import segyio`` fail and force fresh imports of the modules that use it."""
    monkeypatch.setitem(sys.modules, "segyio", None)
    for name in ("PyFWI.model_dataset", "PyFWI.seismic_io"):
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("module", ["PyFWI.seismic_io", "PyFWI.model_dataset"])
def test_modules_import_without_segyio(without_segyio, module):
    importlib.import_module(module)


def test_reading_segy_without_segyio_explains_how_to_install_it(without_segyio):
    io = importlib.import_module("PyFWI.seismic_io")
    with pytest.raises(ImportError, match="pip install segyio"):
        io.read_segy("any.segy")


def test_read_segy_returns_samples_by_traces(tmp_path):
    segyio = pytest.importorskip("segyio")
    import PyFWI.seismic_io as io
    traces = np.arange(24, dtype=np.float32).reshape(4, 6)  # 4 traces of 6 samples
    path = str(tmp_path / "small.segy")
    segyio.tools.from_array(path, traces)
    data = io.read_segy(path)
    assert data.dtype == np.float32 and data.shape == (6, 4)
    np.testing.assert_array_equal(data, traces.T)
