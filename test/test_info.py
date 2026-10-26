"""Diagnostics (roadmap step 1.8): ``PyFWI.info()`` and the summary logged at the first run.

``info()`` must work on any machine (without a GPU, without OpenCL), report what really runs (the
fork, its git commit, the device a propagator would choose, the versions), and ``import PyFWI``
must stay free of side effects, as in 0.1.10: no heavy import, no logging set-up, no environment
change.
"""
import logging
import os
import shutil
import subprocess
import sys

import pytest

from helpers import running_reference, tiny_case

fork_only = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 has no PyFWI.info")

#: Program that imports PyFWI and prints what the import changed.
IMPORT_ONLY = """
import json, logging, os, sys
before = dict(os.environ)
import PyFWI
print(json.dumps({"heavy": [m for m in ("numpy", "scipy", "pyopencl", "matplotlib") if m in sys.modules],
                  "handlers": len(logging.getLogger().handlers) + len(logging.getLogger("PyFWI").handlers),
                  "environment_changed": dict(os.environ) != before}))
"""


def test_import_has_no_side_effects():
    import json
    run = subprocess.run([sys.executable, "-c", IMPORT_ONLY], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == {"heavy": [], "handlers": 0, "environment_changed": False}


@fork_only
def test_info_reports_the_package_and_versions(monkeypatch):
    monkeypatch.delenv("PYFWI_BACKEND", raising=False)  # the default backend
    import numpy
    import scipy

    import PyFWI
    from PyFWI._info import NUMERIC_MODE, ORIGIN
    summary = PyFWI.info()
    assert set(summary) == {"pyfwi", "origin", "source", "commit", "backend", "numeric_mode", "device",
                            "opencl_platforms", "opencl_error", "versions"}
    assert summary["pyfwi"] == PyFWI.__version__ and summary["origin"] == ORIGIN
    assert (summary["backend"], summary["numeric_mode"]) == ("opencl", NUMERIC_MODE) == ("opencl", "legacy")
    assert summary["source"] == os.path.dirname(os.path.abspath(PyFWI.__file__))
    assert summary["versions"]["numpy"] == numpy.__version__ and summary["versions"]["scipy"] == scipy.__version__
    assert repr(summary).startswith(f"PyFWI {PyFWI.__version__}: pyfwi-cuda")  # printed as the summary


@fork_only
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_info_gives_the_commit_of_a_checkout_and_none_elsewhere(tmp_path):
    import PyFWI
    from PyFWI._info import _commit
    folder = os.path.dirname(os.path.abspath(PyFWI.__file__))
    head = subprocess.run(["git", "-C", folder, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          check=False).stdout.strip()
    if not head:
        pytest.skip("PyFWI is not imported from a git checkout")
    assert PyFWI.info()["commit"].split("-")[0] == head
    # A folder of another repository (e.g. a virtual environment inside a project) is not the fork.
    other = tmp_path / "project" / "venv" / "PyFWI"
    other.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tmp_path / "project")], check=True)
    assert _commit(str(other)) is None and _commit(str(tmp_path)) is None


@fork_only
def test_info_names_the_device_a_propagator_would_use(opencl_platform):
    import pyopencl as cl

    import PyFWI
    platforms = cl.get_platforms()
    for index, platform in enumerate(platforms):
        expected = f"{platform.get_devices()[0].name} on {platform.name}"
        assert PyFWI.info(platform=index, device=0)["device"] == expected
    assert len(PyFWI.info()["opencl_platforms"]) == len(platforms)


@fork_only
def test_info_works_without_opencl(monkeypatch):
    import pyopencl as cl

    import PyFWI
    monkeypatch.setattr(cl, "get_platforms", list)  # a machine where PyOpenCL finds no platform
    summary = PyFWI.info()
    assert summary["device"] is None and summary["opencl_platforms"] == []
    assert "No OpenCL platform" in summary["opencl_error"] and "device        none: " in repr(summary)


@fork_only
def test_the_summary_is_logged_once_at_the_first_run(opencl_platform, monkeypatch, caplog):
    import PyFWI._info as info
    monkeypatch.setattr(info, "_logged", False)
    case = tiny_case()
    with caplog.at_level(logging.WARNING, logger="PyFWI"):
        case.W.forward_modeling(case.model, False)  # INFO not enabled: nothing, and not marked as done
    assert not caplog.records and info._logged is False
    with caplog.at_level(logging.INFO, logger="PyFWI"):
        case.W.forward_modeling(case.model, False)
        case.W.forward_modeling(case.model, False)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "first propagation" in message and case.W.ctx.devices[0].name in message
