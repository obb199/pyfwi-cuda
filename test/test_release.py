"""Release checks of the fork (roadmap step 1.9): what a user of a new version does first.

* The original example script of PyFWI 0.1.10, ``example/fwi_example_surface.py`` (forward
  modelling and a two-band FWI of the documentation), runs unchanged as a program.
* The wheel installs in a fresh environment of every supported Python (3.10 to 3.13), with the
  dependencies resolved from the package index as a user gets them, and an example script then
  runs on OpenCL on the CPU (POCL).

Both are slow (minutes) and the second needs the network and uv: marked ``slow``, they run before a
release and on the GPU runner, not in the CPU CI.
"""
import os
import shutil
import subprocess
import sys

import pytest

from helpers import running_reference

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Python versions the fork supports (``requires-python`` and the classifiers of ``pyproject.toml``).
PYTHONS = ("3.10", "3.11", "3.12", "3.13")

#: Program that runs a script as ``__main__``, inside zero guard bands for the published 0.1.10 only
#: (the fork pads its buffers since step 1.7). argv: <test folder> <"guard" or "plain"> <script>.
PROGRAM = """
import contextlib, runpy, sys
sys.path.insert(0, sys.argv[1])
from oob_guard import zero_guard_bands
guard, script, sys.argv = sys.argv[2] == "guard", sys.argv[3], sys.argv[3:]
with zero_guard_bands() if guard else contextlib.nullcontext():
    runpy.run_path(script, run_name="__main__")
"""


@pytest.mark.slow
def test_original_example_script_runs_unchanged(opencl_platform):
    # The script imports from ./src (it was written to run from the repository root) and shows its
    # figures at the end: the Agg backend makes plt.show() return at once.
    env = {**os.environ, "MPLBACKEND": "Agg"}
    mode = "guard" if running_reference() else "plain"
    run = subprocess.run([sys.executable, "-c", PROGRAM, os.path.dirname(__file__), mode,
                          os.path.join("example", "fwi_example_surface.py")],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=1800, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
    assert "for f= 45" in run.stdout  # the second frequency band was inverted (FWI prints each evaluation)


@pytest.mark.slow
@pytest.mark.skipif(running_reference(), reason="packaging of the fork")
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to create the environments")
@pytest.mark.parametrize("python", PYTHONS)
def test_wheel_works_on_every_supported_python(built_distributions, tmp_path, python):
    env_dir = tmp_path / f"python{python}"
    subprocess.run(["uv", "venv", "--quiet", "--python", python, str(env_dir)], check=True)
    python_exe = str(env_dir / "bin" / "python")
    # The wheel with its dependencies, plus OpenCL on the CPU (not a dependency of PyFWI).
    install = subprocess.run(["uv", "pip", "install", "--quiet", "--python", python_exe, str(built_distributions["wheel"]),
                              "pocl-binary-distribution"], capture_output=True, text=True, check=False)
    assert install.returncode == 0, install.stderr[-3000:]
    # Outside the repository, so that only the installed copy can be imported.
    check = subprocess.run([python_exe, os.path.join(REPO, "tools", "check_install.py")], cwd=tmp_path,
                           capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stdout + check.stderr
    assert f"python {python}" in subprocess.run([python_exe, "-c", "import PyFWI; print(PyFWI.info())"],
                                                capture_output=True, text=True, check=True).stdout
    # A forward modelling on POCL, as a user runs an example.
    env = {**os.environ, "MPLBACKEND": "Agg", "PYOPENCL_CTX": "portable"}
    env.pop("PYTHONPATH", None)
    script = os.path.join(REPO, "example", "scripts", "forward_surface.py")
    run = subprocess.run([python_exe, script, "--size", "small", "--no-show"], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=900, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
