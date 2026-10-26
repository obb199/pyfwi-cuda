"""Packaging of the fork (roadmap step 1.1, PKG tests): distribution ``pyfwi-cuda``, import name ``PyFWI``.

The fork is installed with ``pip install pyfwi-cuda`` (once published) but imported as
``import PyFWI``, so that scripts written for PyFWI 0.1.10 run unchanged. These tests check:

* the metadata in ``pyproject.toml``: name, supported Python versions, the dependencies that
  0.1.10 forgot (``pyopencl``) or declared by mistake (``datetime``, ``setuptools``, ``wheel``),
  and the optional dependencies (extras ``segy``, ``torch``, and ``cuda13`` and ``cuda12`` for the
  CUDA backend, roadmap step 2.6);
* that the version is written in one place only (``src/PyFWI/version.py``);
* the content of the wheel and of the source distribution, built from a copy of the sources by
  the ``built_distributions`` fixture of ``conftest.py``;
* that the wheel installs in a fresh virtual environment and that every module then imports
  (``tools/check_install.py``), without segyio or torch; and, with each CUDA extra, that the CUDA
  backend then runs (slow).

Skipped in the reference environment, which has the published 0.1.10 and no build tools.
"""
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from helpers import running_reference
from tools.check_install import DATA_FILES, DISTRIBUTION

pytestmark = pytest.mark.skipif(running_reference(), reason="the reference environment has the published 0.1.10")

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#: Folder of the package sources.
PACKAGE = os.path.join(REPO, "src", "PyFWI")


def _pyproject():
    """``pyproject.toml`` as a dict (``tomllib`` is in the standard library from Python 3.11 on)."""
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def _name(requirement):
    """Project name of a requirement string, in lower case: ``"segyio>=1.9; extra == 'segy'"`` -> ``"segyio"``."""
    return re.split(r"[ ;<>=!~\[(]", requirement.strip(), maxsplit=1)[0].lower()


def _source_files():
    """Paths of every file of the package sources, relative to ``src/`` (``PyFWI/...``), without caches."""
    paths = []
    for root, dirs, files in os.walk(PACKAGE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        paths += [os.path.relpath(os.path.join(root, f), os.path.dirname(PACKAGE)).replace(os.sep, "/") for f in files]
    return sorted(paths)


def _version_in_sources():
    """The version written in ``src/PyFWI/version.py`` (what the wheel must carry)."""
    with open(os.path.join(PACKAGE, "version.py")) as f:
        return re.fullmatch(r"version = '(.+)'", f.read().strip()).group(1)


def test_a_release_version_has_a_dated_changelog_section():
    # A version without ".dev" is a release: its CHANGELOG section must exist, with its date.
    version = _version_in_sources()
    if ".dev" in version:
        pytest.skip(f"{version} is a development version")
    with open(os.path.join(REPO, "CHANGELOG.md"), encoding="utf-8") as f:
        changelog = f.read()
    assert re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE)


def test_pyproject_declares_the_fork():
    project = _pyproject()["project"]
    assert project["name"] == DISTRIBUTION
    assert project["requires-python"] == ">=3.10"
    assert project["dynamic"] == ["version"] and "version" not in project
    assert project["license"] == "GPL-3.0-or-later" and project["license-files"] == ["LICENSE"]
    dependencies = {_name(r) for r in project["dependencies"]}
    assert dependencies == {"numpy", "scipy", "matplotlib", "hdf5storage", "requests", "pyopencl"}
    extras = {extra: {_name(r) for r in reqs} for extra, reqs in project["optional-dependencies"].items()}
    assert extras == {"segy": {"segyio"}, "torch": {"torch"}, "cuda13": {"cupy-cuda13x"}, "cuda12": {"cupy-cuda12x"}}
    # [ctk]: the CUDA runtime and NVRTC from PyPI, so that only a driver is needed.
    assert all("[ctk]" in r for extra in ("cuda13", "cuda12") for r in project["optional-dependencies"][extra])


def test_the_setuptools_configuration_ships_the_kernels_and_the_data():
    setuptools = _pyproject()["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["dynamic"]["version"] == {"attr": "PyFWI.version.version"}
    assert set(setuptools["package-data"]["PyFWI"]) == {"*.cl", "data/*.mat", "kernels/*.h"}


def test_inherited_packaging_files_are_gone():
    # setup.py duplicated the metadata (with the wrong dependencies); requirements.txt pinned the
    # 0.1.10 stack, which is now in requirements/ref-0.1.10.txt.
    assert not os.path.exists(os.path.join(REPO, "setup.py"))
    assert not os.path.exists(os.path.join(REPO, "requirements.txt"))


def test_version_is_written_in_one_place():
    import PyFWI
    with open(os.path.join(PACKAGE, "version.py")) as f:
        assert f.read().strip() == f"version = '{PyFWI.__version__}'"
    # A release (0.1.11) or a development version after it (0.1.12.dev0): PEP 440, no local label.
    assert re.fullmatch(r"0\.\d+\.\d+(\.dev\d+)?", PyFWI.__version__), PyFWI.__version__
    with open(os.path.join(REPO, "docs", "conf.py")) as f:
        conf = f.read()
    assert "from PyFWI.version import version as release" in conf  # the docs do not repeat the number
    owners = importlib.metadata.packages_distributions().get("PyFWI", [])
    if DISTRIBUTION in owners:  # installed metadata (editable install) agrees
        assert importlib.metadata.version(DISTRIBUTION) == PyFWI.__version__


def test_wheel_metadata(built_distributions):
    with zipfile.ZipFile(built_distributions["wheel"]) as zf:
        info = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = zf.read(info).decode()
        names = zf.namelist()
    fields = [line.split(": ", 1) for line in metadata.split("\n\n", 1)[0].splitlines() if ": " in line]
    single = dict(fields)
    assert single["Name"] == DISTRIBUTION
    assert single["Version"] == _version_in_sources()
    assert single["Requires-Python"] == ">=3.10"
    assert single["License-Expression"] == "GPL-3.0-or-later"
    requires = [value for key, value in fields if key == "Requires-Dist"]
    base = {_name(r) for r in requires if "extra ==" not in r}
    assert base == {"numpy", "scipy", "matplotlib", "hdf5storage", "requests", "pyopencl"}
    assert any(_name(r) == "segyio" and "extra == \"segy\"" in r for r in requires)
    assert any(_name(r) == "torch" and "extra == \"torch\"" in r for r in requires)
    assert any(_name(r) == "cupy-cuda13x" and "extra == \"cuda13\"" in r for r in requires)
    assert any(_name(r) == "cupy-cuda12x" and "extra == \"cuda12\"" in r for r in requires)
    assert {value for key, value in fields if key == "Provides-Extra"} == {"segy", "torch", "cuda13", "cuda12"}
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in names)


def test_wheel_contains_every_file_of_the_package(built_distributions):
    with zipfile.ZipFile(built_distributions["wheel"]) as zf:
        names = set(zf.namelist())
    missing = [p for p in _source_files() if p not in names]
    assert not missing, f"files of src/PyFWI missing from the wheel: {missing}"
    assert {f"PyFWI/{name}" for name in DATA_FILES} <= names
    # Only the package and its metadata: no tests, tools or caches.
    assert all(n.startswith(("PyFWI/", "pyfwi_cuda-")) for n in names), sorted(names)
    assert not any("__pycache__" in n for n in names)


def test_sdist_contains_the_sources_licence_and_readme(built_distributions):
    with tarfile.open(built_distributions["sdist"]) as tf:
        names = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
    assert {"LICENSE", "README.md", "pyproject.toml"} <= names
    missing = [p for p in _source_files() if f"src/{p}" not in names]
    assert not missing, f"files of src/PyFWI missing from the sdist: {missing}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to create the environment offline")
def test_wheel_installs_and_imports_in_a_clean_environment(built_distributions, tmp_path):
    # A fresh environment with the wheel and its declared dependencies only (no segyio, no torch).
    # Dependencies come from uv's cache, at the versions of the development environment.
    env = tmp_path / "clean"
    subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(env)], check=True)
    install = subprocess.run(
        ["uv", "pip", "install", "--quiet", "--offline", "--python", str(env / "bin" / "python"),
         "--constraint", os.path.join(REPO, "requirements", "dev.txt"), str(built_distributions["wheel"])],
        capture_output=True, text=True, check=False)
    assert install.returncode == 0, install.stderr
    # Run from tmp_path, not from the repository, so that only the installed copy can be imported.
    check = subprocess.run([str(env / "bin" / "python"), os.path.join(REPO, "tools", "check_install.py")],
                           cwd=tmp_path, capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stdout + check.stderr
    assert str(env) in check.stdout  # PyFWI was imported from the clean environment
    assert "[skip] import PyFWI.torchfwi" in check.stdout  # torch is an extra, not installed
    assert "[  ok] import PyFWI.seismic_io" in check.stdout and "[  ok] import PyFWI.model_dataset" in check.stdout
    assert "[  ok] model 'dupuy'" in check.stdout  # the data file is installed and readable


#: The CUDA extras, with the requirements file whose versions they are installed at and the
#: environment of tools/setup_envs.sh that put those versions in uv's cache.
CUDA_EXTRAS = [("cuda13", "dev.txt", "dev"), ("cuda12", "cuda12.txt", "cuda12")]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to create the environment offline")
@pytest.mark.parametrize("extra, constraints, cached_by", CUDA_EXTRAS)
def test_cuda_extra_installs_and_runs_the_cuda_backend(built_distributions, tmp_path, cuda_device, extra,
                                                       constraints, cached_by):
    # The wheel with one CUDA extra in a fresh environment (offline: the CUDA wheels come from uv's
    # cache, filled when tools/setup_envs.sh created the environment), then the CUDA backend: the
    # selection reports it and an example script runs on it.
    if not os.path.isdir(os.path.join(os.path.dirname(REPO), "envs", cached_by)):
        pytest.skip(f"envs/{cached_by} not created: its CUDA wheels are not in uv's cache")
    env = tmp_path / extra
    python = str(env / "bin" / "python")
    subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(env)], check=True)
    install = subprocess.run(
        ["uv", "pip", "install", "--quiet", "--offline", "--python", python,
         "--constraint", os.path.join(REPO, "requirements", constraints), f"{built_distributions['wheel']}[{extra}]"],
        capture_output=True, text=True, check=False)
    assert install.returncode == 0, install.stderr
    check = subprocess.run([python, os.path.join(REPO, "tools", "check_install.py")], cwd=tmp_path,
                           capture_output=True, text=True, check=False)
    assert check.returncode == 0 and "[  ok] import PyFWI.backends.cuda" in check.stdout, check.stdout + check.stderr
    run_env = {**os.environ, "PYFWI_BACKEND": "cuda:strict", "MPLBACKEND": "Agg"}
    run_env.pop("PYTHONPATH", None)
    info = subprocess.run([python, "-c", "import PyFWI; print(PyFWI.info())"], cwd=tmp_path, env=run_env,
                          capture_output=True, text=True, check=False)
    assert info.returncode == 0 and "CUDA" in info.stdout, info.stdout + info.stderr
    script = os.path.join(REPO, "example", "scripts", "forward_surface.py")
    run = subprocess.run([python, script, "--size", "small", "--no-show"], cwd=tmp_path, env=run_env,
                         capture_output=True, text=True, timeout=900, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
