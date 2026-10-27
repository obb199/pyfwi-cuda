"""The notebooks of ``example/`` run with the fork (roadmap step 5.1, test type DOC).

The notebooks are the examples of the original PyFWI. Each one is run as its users run it, from the
``example/`` folder, but in a reduced size: the code of its cells is joined into one script, a few values are
replaced (shorter recordings, fewer iterations; ``REDUCTIONS``, each replacement required to match, so that a
change of a notebook is noticed here), and the script runs in a new Python process, with matplotlib drawing
to no window. The test passes when the notebook runs to its end; what the examples compute is checked by the
tests of ``example/scripts/`` (``test_examples.py``).

The notebooks run on both engines (the ``backend`` fixture): the CPU CI, which has POCL and no NVIDIA GPU,
runs them on OpenCL. ``grad_pytorch.ipynb`` needs PyTorch, which the CPU CI does not install.
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


def reduced_script(name):
    """The code cells of ``example/<name>`` joined into one script, with the replacements of ``REDUCTIONS``.

    Raises
    ------
    AssertionError
        A replacement that matches nothing (the notebook changed).
    """
    with open(os.path.join(EXAMPLES, name)) as f:
        cells = json.load(f)["cells"]
    script = "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    for text, replacement in REDUCTIONS[name]:
        assert text in script, f"{name}: {text!r} not found: update REDUCTIONS"
        script = script.replace(text, replacement)
    return script


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
    run = subprocess.run([sys.executable, "-c", reduced_script(name)], cwd=EXAMPLES, capture_output=True,
                         text=True, env=dict(os.environ, MPLBACKEND="Agg"), timeout=900, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
