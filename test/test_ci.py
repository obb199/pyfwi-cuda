"""Checks of the continuous-integration setup (``.github/workflows/`` and ``requirements/ci-cpu.txt``).

The CI configuration is code that only runs on GitHub, so mistakes are found
late. These tests catch the important ones locally:

* no workflow may publish to PyPI (the fork must never upload under the name of
  the original package by accident), and the inherited workflows stay removed;
* the CI job runs the steps the roadmap promises (both lint passes, the GPU-less
  environment check, the test suite and the wheel installed in a fresh environment);
* the CI dependency file uses exactly the versions of the development
  environment, so "passes locally" and "passes in CI" mean the same thing;
* on a runner whose CPU PoCL cannot compile for, the OpenCL tests are reported
  as skipped with a warning, not as failures (hook in ``conftest.py``).
"""
import glob
import os
import re
import shutil
import subprocess
import sys

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#: Folder of the GitHub Actions workflows.
WORKFLOWS = os.path.join(REPO, ".github", "workflows")


def _pins(name):
    """Return ``{package (lower case): version}`` for the ``pkg==version`` lines of ``requirements/<name>``."""
    pins = {}
    with open(os.path.join(REPO, "requirements", name)) as f:
        for line in f:
            m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line.strip())
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def test_no_workflow_can_publish_to_pypi():
    for path in glob.glob(os.path.join(WORKFLOWS, "*.y*ml")):
        with open(path, encoding="utf-8") as f:
            text = f.read().lower()
        assert "pypi-publish" not in text and "twine upload" not in text, os.path.basename(path)


def test_inherited_workflows_are_removed():
    # pr.yml tested Python 3.8-3.10 with the old requirements; publish.yml uploaded to PyPI as "PyFWI".
    assert not os.path.exists(os.path.join(WORKFLOWS, "pr.yml"))
    assert not os.path.exists(os.path.join(WORKFLOWS, "publish.yml"))


def test_ci_workflow_runs_the_promised_steps():
    with open(os.path.join(WORKFLOWS, "ci.yml"), encoding="utf-8") as f:
        ci = f.read()
    assert "fetch-depth: 0" in ci  # test_docs.py needs the history to find cited commits
    assert "pip install -r requirements/ci-cpu.txt" in ci
    assert "ruff check --isolated --select E9,F63,F7,F82 --line-length 127 ." in ci
    assert re.search(r"run: ruff check \.\s*$", ci, re.MULTILINE)
    assert "tools/check_env.py --no-gpu" in ci
    assert 'pytest -m "not slow"' in ci
    # Packaging (PKG): wheel in a fresh environment, checked outside the checkout (step 1.1).
    assert "pip wheel --no-deps --no-build-isolation --wheel-dir dist ." in ci
    assert '-m pip install dist/pyfwi_cuda-*.whl' in ci
    assert 'cd "$RUNNER_TEMP" && "$RUNNER_TEMP/clean/bin/python" "$GITHUB_WORKSPACE/tools/check_install.py"' in ci


def test_ci_requirements_use_the_dev_versions():
    ci, dev = _pins("ci-cpu.txt"), _pins("dev.txt")
    different = {pkg: (ver, dev.get(pkg)) for pkg, ver in ci.items() if dev.get(pkg) != ver}
    assert not different, f"ci-cpu.txt and dev.txt disagree (ci, dev): {different}"
    assert {"numpy", "scipy", "pyopencl", "pocl-binary-distribution", "pytest", "ruff"} <= set(ci)
    assert "cupy-cuda13x" not in ci and "torch" not in ci  # the CPU job has no GPU stack


def test_gpu_workflow_targets_the_self_hosted_rtx_and_is_manual_until_the_runner_exists():
    with open(os.path.join(WORKFLOWS, "gpu.yml"), encoding="utf-8") as f:
        gpu = f.read()
    active = "\n".join(line for line in gpu.splitlines() if not line.lstrip().startswith("#"))
    assert "workflow_dispatch:" in active
    # Until a runner is registered (docs/cuda-migration/self-hosted-runner.md), no automatic trigger.
    assert "push:" not in active and "schedule:" not in active, "enable only once the runner exists"
    assert "runs-on: [self-hosted, linux, x64, gpu]" in active
    assert "tools/setup_envs.sh all" in active  # tests the checked-out commit, not a local checkout
    assert 'pytest -m "not slow"' in active and "pytest -m slow" in active
    assert "ref-0.1.10/bin/python -m pytest" in active


def test_pocl_unknown_cpu_failures_become_skips_with_a_warning(tmp_path):
    # A throw-away test session with a copy of conftest.py: one test fails with the PoCL error of
    # an unknown CPU, the other with a different build error, which must stay a failure.
    shutil.copy(os.path.join(REPO, "test", "conftest.py"), tmp_path)
    (tmp_path / "test_sample.py").write_text(
        "def test_unknown_cpu():\n"
        "    raise RuntimeError(\"clBuildProgram failed: BUILD_PROGRAM_FAILURE\\nerror: unknown target CPU 'generic'\")\n"
        "\n"
        "def test_other_build_error():\n"
        "    raise RuntimeError('clBuildProgram failed: BUILD_PROGRAM_FAILURE\\nerror: use of undeclared identifier')\n")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", str(tmp_path)],
                         cwd=tmp_path, capture_output=True, text=True, check=False,
                         env={**os.environ, "GITHUB_ACTIONS": "true"})
    assert "1 failed, 1 skipped" in run.stdout, run.stdout
    assert "FAILED test_sample.py::test_other_build_error" in run.stdout
    assert "SKIPPED [1] test_sample.py:1: PoCL 3.0 cannot compile OpenCL kernels for this CPU" in run.stdout
    assert "::warning title=PoCL cannot compile on this CPU::1 tests skipped" in run.stdout
