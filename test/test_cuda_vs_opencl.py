"""The final document, CUDA against OpenCL on the project's GPU (roadmap step 6).

``docs/cuda-migration/cuda-vs-opencl.md`` is written for readers who know neither GPUs nor the code, and every
number in it must come from a result file of the repository (``tools/cuda_vs_opencl.py``). Tested here: the
figures are drawn from the stored results, every number with a unit in the document (times faster, percent,
seconds, memory) is one that ``facts()`` computes from those files, the stored session is complete and says
what the document says, and the README links the document.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import cuda_vs_opencl as cvo

DOCUMENT = os.path.join(cvo.REPO, "docs", "cuda-migration", "cuda-vs-opencl.md")

#: A number with a unit, as the document writes the numbers of ``facts()``: ``12×``, ``3.4×``, ``95 %``,
#: ``0.41 s``, ``0.095 ms``, ``418 MB``, ``4.4 GB`` (not ``GB/s``, a bandwidth), ``4.1 commands``, ``1.0 copies``.
QUANTITY = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:×| %| s(?![\w/])| ms\b| MB| GB(?!/)| commands| copies)")

#: Numbers with a unit that are not measurements: the memory of the card, and the share of its free memory the
#: checkpoints may take (``PyFWI.backends.cuda.CUDABackend.checkpoint_budget``, checked below).
DESCRIPTIVE = {"12 GB", "80 %"}


def test_the_checkpoint_budget_is_that_of_the_code():
    with open(os.path.join(cvo.REPO, "src", "PyFWI", "backends", "cuda.py")) as f:  # CuPy may be missing here
        assert "int(0.8 * (free" in f.read()


def test_numbers_are_written_as_the_document_writes_them():
    assert [cvo._times(v) for v in (31.4, 3.42, 1.254)] == ["31×", "3.4×", "1.25×"]
    assert [cvo._seconds(v) for v in (0.0181, 0.4111, 17.93, 528.9)] == ["0.018 s", "0.41 s", "17.9 s", "529 s"]
    assert QUANTITY.findall("12× and 3.4× in 0.41 s, 95 % busy, 418 MB, 4.4 GB, 360 GB/s, sm_86, 0.095 ms, "
                            "4.1 commands, 1.0 copies") == [
        "12×", "3.4×", "0.41 s", "95 %", "418 MB", "4.4 GB", "0.095 ms", "4.1 commands", "1.0 copies"]


def test_the_session_is_complete():
    benchmarks, busy = cvo.load()
    for (backend, mode), scenarios in benchmarks.items():
        assert set(scenarios) == set(cvo.SCENARIOS), (backend, mode)
        engine = scenarios["small"]["engine"]
        assert engine["numerics"] == mode and engine["backend"].startswith(backend), engine
        assert not engine["boundary_saving"] and not engine["fp16"]
    assert set(busy) == {(b, m, s) for b in cvo.BACKENDS for m in cvo.MODES for s in cvo.BUSY_SCENARIOS}


def test_the_session_says_what_the_document_says():
    # CUDA is faster than OpenCL in every scenario in the same numeric mode, and its default is faster than
    # the default of OpenCL; the GPU is busier on CUDA in the fixed mode.
    benchmarks, busy = cvo.load()
    for scenario in cvo.SCENARIOS:
        for kind, _ in cvo.KINDS:
            for opencl_mode, cuda_mode in (("legacy", "legacy"), ("fixed", "fixed"), ("legacy", "fixed")):
                assert cvo.speedup(benchmarks, scenario, kind, opencl_mode, cuda_mode) > 1, (scenario, kind)
    for scenario in cvo.BUSY_SCENARIOS:
        assert busy["cuda", "fixed", scenario] > busy["opencl", "legacy", scenario]


def test_the_figures_are_drawn_from_the_stored_results(tmp_path):
    paths = cvo.figures(str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["like_for_like.png", "defaults.png", "gpu_busy.png", "memory.png"]
    assert all(os.path.getsize(p) > 10_000 for p in paths)
    with open(DOCUMENT) as f:
        shown = set(re.findall(r"\]\(cuda-vs-opencl/([\w.]+\.png)\)", f.read()))
    assert shown == {os.path.basename(p) for p in paths} == set(os.listdir(cvo.FIGURES))


def test_every_number_of_the_document_comes_from_the_results():
    with open(DOCUMENT) as f:
        text = f.read()
    known = set(cvo.facts().values()) | DESCRIPTIVE
    unknown = sorted({q for q in QUANTITY.findall(text) if q not in known})
    assert not unknown, f"numbers of the document that no stored result gives: {unknown}"
    assert len(QUANTITY.findall(text)) >= 20  # the document is made of measurements


def test_the_readme_links_the_document():
    with open(os.path.join(cvo.REPO, "README.md")) as f:
        assert "(docs/cuda-migration/cuda-vs-opencl.md)" in f.read()


@pytest.mark.parametrize("name", ["default", "boundary saving", "FP16", "both"])
def test_the_memory_of_the_release_runs(name):
    memory = cvo.release_memory()[name]
    assert set(memory) == set(cvo.SCENARIOS) and all(v > 0 for v in memory.values())
