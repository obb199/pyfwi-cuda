"""Tests of the golden-master storage (``test/golden_store.py``) and generator (``tools/make_golden.py``)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from golden_cases import propagation_cases
from golden_store import array_sha256, load_case, load_manifest, save_case, verify_case, write_manifest

#: Absolute path of the repository root and of the generator script.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAKE_GOLDEN = os.path.join(REPO, "tools", "make_golden.py")


def test_array_sha256_depends_on_bytes_dtype_and_shape():
    a = np.arange(6, dtype=np.float32)
    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a.reshape(2, 3))       # same bytes, other shape
    assert array_sha256(a) != array_sha256(a.view(np.int32))      # same bytes, other dtype
    b = a.copy()
    b[3] = np.nextafter(b[3], np.float32(10))                    # one ulp
    assert array_sha256(a) != array_sha256(b)


def test_save_load_verify_round_trip(tmp_path):
    arrays = {"gradient/vp": np.ones((3, 4), np.float32), "misfit": np.asarray(1.5)}
    relative, hashes = save_case("case_x", arrays, tmp_path)
    assert relative == "data/case_x.npz" and set(hashes) == set(arrays)
    loaded = load_case("case_x", tmp_path)
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[key], value)
        assert loaded[key].dtype == value.dtype
    manifest = {"cases": {"case_x": {"sha256": hashes}}}
    write_manifest(manifest, tmp_path)
    assert load_manifest(tmp_path) == manifest
    assert verify_case("case_x", manifest, tmp_path) == []
    # Tampering is detected.
    manifest["cases"]["case_x"]["sha256"]["misfit"] = "0" * 64
    assert verify_case("case_x", manifest, tmp_path) == ["misfit"]


def test_load_manifest_of_an_empty_folder_is_none(tmp_path):
    assert load_manifest(tmp_path) is None


def _make_golden(*args):
    """Run tools/make_golden.py with the current interpreter; return the CompletedProcess."""
    return subprocess.run([sys.executable, "-W", "ignore", MAKE_GOLDEN, *args], check=False, capture_output=True,
                          text=True, timeout=900)


def _running_the_fork():
    import PyFWI
    return os.path.abspath(PyFWI.__file__).startswith(REPO + os.sep)


@pytest.mark.skipif(not _running_the_fork(), reason="this environment has the published 0.1.10")
def test_generator_refuses_the_fork_code():
    r = _make_golden("--cases", propagation_cases()[0].name)
    assert r.returncode == 2 and "is not the published 0.1.10" in r.stderr


def test_allow_fork_requires_an_output_folder():
    r = _make_golden("--allow-fork")
    assert r.returncode == 2 and "--allow-fork needs --out" in r.stderr


def test_generator_writes_a_complete_manifest(opencl_platform, tmp_path):
    name = propagation_cases()[1].name
    r = _make_golden("--allow-fork", "--out", str(tmp_path), "--cases", name, "--check-plain")
    assert r.returncode == 0, r.stderr[-3000:]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format"] == 1 and manifest["generated"] and manifest["rules"]["guard_bands"] == "zero"
    assert {"python", "numpy", "scipy", "pyopencl", "pyfwi", "pyfwi_path", "opencl_device", "driver"} <= set(
        manifest["environment"])
    entry = manifest["cases"][name]
    assert entry["kind"] == "propagation" and entry["deterministic"] is True
    assert entry["plain_run_matches"] in (True, False)
    assert entry["params"]["acq_type"] == propagation_cases()[1].acq_type
    assert verify_case(name, manifest, tmp_path) == []
