"""The notebooks of ``example/`` run with the fork (roadmap step 5.1, test type DOC).

The notebooks are the examples of the original PyFWI. Each one is run as its users run it, from the
``example/`` folder, but in a reduced size: the code of its cells is joined into one script, a few values are
replaced (shorter recordings, fewer iterations; ``REDUCTIONS``, each replacement required to match, so that a
change of a notebook is noticed here), and the script runs in a new Python process, with matplotlib drawing
to no window. The test passes when the notebook runs to its end; what the examples compute is checked by the
tests of ``example/scripts/`` (``test_examples.py``).

The notebooks run on both engines (the ``backend`` fixture): the CPU CI, which has POCL and no NVIDIA GPU,
runs them on OpenCL. ``grad_pytorch.ipynb`` needs PyTorch, which the CPU CI does not install.

The notebooks of ``example/cuda_vs_opencl/``, written for the fork, compare the two backends on one GPU; they
run the same way, from their folder (they import its ``comparison.py``), in a reduced size
(``COMPARISON_REDUCTIONS``), on a machine with CUDA. They are stored without outputs, so that a reader sees the
numbers of their own GPU, and the numbers they quote from the measurements of the repository are those of
the stored results.
"""
import json
import os
import subprocess
import sys

import pytest

from helpers import running_reference

pytestmark = pytest.mark.skipif(running_reference(), reason="the published 0.1.10 cannot import model_dataset "
                                                            "without segyio, which its environment does not install")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(REPO, "example")
COMPARISONS = os.path.join(EXAMPLES, "cuda_vs_opencl")

#: Replacements that make each notebook small: ``{notebook: [(text, replacement), ...]}``. Recordings of
#: 0.3 s instead of 0.6 or 0.8 s, one iteration per frequency instead of two, and wave snapshots that exist
#: in the shorter recording (``W.W`` keeps one snapshot every 5 internal steps with ``chpr=20``).
REDUCTIONS = {
    "fwi_example_crosswell.ipynb": [("'t': 0.6", "'t': 0.3"), ("iter=[2, 2]", "iter=[1, 1]")],
    "fwi_example_surface.ipynb": [("'t': 0.6", "'t': 0.3"), ("iter=[2, 2]", "iter=[1, 1]")],
    "gradient_example.ipynb": [("'t': 0.8", "'t': 0.3"),
                               ("snapshots = [40, 80, 130, 180]", "snapshots = [10, 20, 30, 40]")],
    "grad_pytorch.ipynb": [("'t': 0.8", "'t': 0.3")],
    "tlfwi_example_crosswell.ipynb": [("'t': 0.6", "'t': 0.3"), ("iter=[2, 2]", "iter=[1, 1]")],
}


#: The same for the notebooks of ``example/cuda_vs_opencl/``: models of 30 to 50 points, short recordings, one
#: repetition, two frequency bands of two iterations.
COMPARISON_REDUCTIONS = {
    "1_speed.ipynb": [('"small": {"nz": 100, "nx": 100, "ns": 4, "t": 0.6}', '"small": {"nz": 40, "nx": 40, "ns": 2, "t": 0.2}'),
                      ('"medium": {"nz": 200, "nx": 200, "ns": 4, "t": 1.0},', ""),
                      ('"large": {"nz": 300, "nx": 300, "ns": 2, "t": 1.4},', ""), ("REPEATS = 3", "REPEATS = 1")],
    "2_same_results.ipynb": [("NZ, NX, NS, T = 100, 100, 4, 0.6", "NZ, NX, NS, T = 40, 40, 2, 0.2"),
                             ("RECEIVER = 60", "RECEIVER = 10")],
    "3_inversion.ipynb": [("NZ, NX, NS, T = 100, 100, 4, 0.6", "NZ, NX, NS, T = 40, 40, 2, 0.3"),
                          ("FREQS = [15, 25, 45]", "FREQS = [15, 45]"), ("ITERATIONS = [15, 15, 15]", "ITERATIONS = [2, 2]")],
    "4_time_per_step.ipynb": [("SIZES = [50, 100, 200, 300, 400, 600]", "SIZES = [30, 50]"), ("T = 0.3", "T = 0.1"),
                              ("REPEATS = 3", "REPEATS = 1"), ("SOURCES = [1, 2, 4, 8]", "SOURCES = [1, 2]")],
    "5_memory.ipynb": [("NZ, NX, NS, T = 300, 300, 2, 1.4", "NZ, NX, NS, T = 40, 40, 2, 0.2"), ("REPEATS = 2", "REPEATS = 1")],
}


def reduced_script(name, folder=EXAMPLES, reductions=REDUCTIONS):
    """The code cells of ``<folder>/<name>`` joined into one script, with the replacements of ``reductions``.

    Raises
    ------
    AssertionError
        A replacement that matches nothing (the notebook changed).
    """
    with open(os.path.join(folder, name)) as f:
        cells = json.load(f)["cells"]
    script = "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    for text, replacement in reductions[name]:
        assert text in script, f"{name}: {text!r} not found: update the reductions"
        script = script.replace(text, replacement)
    return script


def _run(script, folder):
    """Run ``script`` in a new Python process from ``folder``, matplotlib drawing to no window."""
    return subprocess.run([sys.executable, "-c", script], cwd=folder, capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"), timeout=900, check=False)


def test_every_notebook_is_tested():
    assert sorted(REDUCTIONS) == sorted(n for n in os.listdir(EXAMPLES) if n.endswith(".ipynb"))


def test_every_notebook_has_the_note_of_the_fork():
    # The first cells say how the notebook runs with the fork and link the user guide.
    assert os.path.exists(os.path.join(EXAMPLES, "../docs/cuda-migration/user-guide.md"))
    for name in REDUCTIONS:
        with open(os.path.join(EXAMPLES, name)) as f:
            cells = json.load(f)["cells"]
        note = "".join(cells[1]["source"])
        assert cells[1]["cell_type"] == "markdown" and "PyFWI-CUDA" in note, name
        assert "(../docs/cuda-migration/user-guide.md)" in note, name


@pytest.mark.parametrize("name", sorted(REDUCTIONS))
def test_the_notebook_runs(backend, name):
    if name == "grad_pytorch.ipynb":
        pytest.importorskip("torch")
    run = _run(reduced_script(name), EXAMPLES)
    assert run.returncode == 0, run.stderr[-3000:]


# ---------------------------------------------------------------------------- example/cuda_vs_opencl

def _comparison_notebooks():
    return sorted(n for n in os.listdir(COMPARISONS) if n.endswith(".ipynb"))


def _cells(name, kind):
    with open(os.path.join(COMPARISONS, name)) as f:
        return [cell for cell in json.load(f)["cells"] if cell["cell_type"] == kind]


def test_every_comparison_notebook_is_tested_and_listed():
    assert sorted(COMPARISON_REDUCTIONS) == _comparison_notebooks()
    with open(os.path.join(COMPARISONS, "README.md")) as f:
        readme = f.read()
    assert all(f"]({name})" in readme for name in COMPARISON_REDUCTIONS)
    with open(os.path.join(REPO, "README.md")) as f:
        assert "(example/cuda_vs_opencl/)" in f.read()


@pytest.mark.parametrize("name", sorted(COMPARISON_REDUCTIONS))
def test_the_comparison_notebook_is_stored_without_outputs(name):
    # The reader sees the numbers of their own GPU, never stale ones; the first code cell checks the engines.
    code = _cells(name, "code")
    assert all(not cell["outputs"] and cell["execution_count"] is None for cell in code), name
    assert "check_engines()" in "".join(code[0]["source"]), name


#: Numbers with a unit in the text of the comparison notebooks that are not measurements: the recording lengths
#: of their settings, and what a signal-to-noise ratio in decibels means.
COMPARISON_DESCRIPTIVE = {"0.6 s", "1.4 s", "1 %", "10 %"}


def test_the_quoted_numbers_are_those_of_the_stored_results():
    # A number with a unit in the text of a notebook (8.8×, 1.7 %, 7.7 s...) quotes the final document or the
    # geophysical validation: it must be one that their stored results give.
    sys.path.insert(0, REPO)
    from test_cuda_vs_opencl import DESCRIPTIVE, QUANTITY

    from tools import cuda_vs_opencl as cvo

    with open(cvo.INVERSIONS) as f:
        inversions = json.load(f)["cases"]["fwi"]
    known = (set(cvo.facts().values()) | DESCRIPTIVE | COMPARISON_DESCRIPTIVE
             | {cvo._seconds(run["time_s"]) for run in inversions.values()})
    text = "\n".join("".join(cell["source"]) for name in _comparison_notebooks() for cell in _cells(name, "markdown"))
    quoted = QUANTITY.findall(text)
    assert len(quoted) >= 6 and not sorted(set(quoted) - known), sorted(set(quoted) - known)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(COMPARISON_REDUCTIONS))
def test_the_comparison_notebook_runs(cuda_device, name):
    run = _run(reduced_script(name, COMPARISONS, COMPARISON_REDUCTIONS), COMPARISONS)
    assert run.returncode == 0, run.stderr[-3000:]
