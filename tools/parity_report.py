"""Parity report of the CUDA backend (roadmap steps 2.7 and 3.11): CUDA against OpenCL, case by case.

In the ``legacy`` numeric mode the CUDA backend runs the kernel files of the OpenCL backend; in the
``fixed`` mode, since 0.3.0, kernels written for CUDA (``PyFWI/kernels/elastic.cu``) that compute the
same thing. The compilers (NVRTC and NVIDIA's OpenCL compiler) do not emit the same instructions,
and the native kernels order some operations differently, so the results differ by rounding. This
script measures by how much on the whole matrix of the golden masters (``test/golden_cases.py``), on
the same GPU:

* the 16 propagation cases (forward modelling and gradient: raw seismograms, data, misfit, raw and
  processed gradients, every component and parameter);
* the 10 inversion cases (FWI with L-BFGS and with the conjugate gradient, 2 iterations; the eight
  time-lapse strategies, 1 iteration): inverted models and misfits.

Every case runs on OpenCL and on CUDA in each numeric mode (``legacy``: the kernels of 0.1.10;
``fixed``: the corrections of step 2.4). The difference of an output is its relative L2 norm,
``||cuda - opencl|| / ||opencl||`` in float64. The report gives, per numeric mode, the largest
difference per kind of output (the tolerance of ``test/test_parity.py`` is set from it), then the
difference of every output of every case.

Usage::

    ../envs/dev/bin/python tools/parity_report.py      # both modes: reports/parity-<version of PyFWI>.md
    ../envs/dev/bin/python tools/parity_report.py --modes legacy --out /tmp/parity.md

Used by
-------
Releases 0.2.0 and 0.3.0 (``docs/cuda-migration/reports/parity-<version>.md``). ``render`` is tested
by ``test/test_parity.py``.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "test"), REPO]

#: Where the report of a release goes: ``REPORT.format(version)``.
REPORT = os.path.join(REPO, "docs", "cuda-migration", "reports", "parity-{}.md")

#: The backends compared: the reference, then the one measured against it (``PYFWI_BACKEND``).
BACKENDS = ("opencl", "cuda:strict")

#: The opening paragraphs of the report.
INTRODUCTION = (
    ("CUDA against OpenCL on the same GPU, on the whole matrix of the golden masters (`tools/parity_report.py`): "
     "relative L2 difference `||cuda - opencl|| / ||opencl||` of every output, in each numeric mode, each backend "
     "with its default settings. In the `legacy` mode both backends run the same kernel files; in the `fixed` mode "
     "the CUDA backend runs, since 0.3.0, the native kernels of Phase 3, which compute the same thing with some "
     "operations in another order (multiplications by 1 / dx instead of divisions, the shots in batches whose "
     "gradients are added at the end). The compilers also emit different instructions (contractions into fused "
     "multiply-adds), hence differences of the order of rounding, checked by the tolerances of `test/test_parity.py`."),
    ("The largest gradient differences, in the `fixed` mode, are those of one case (wells and surface, pressure "
     "data, energy balancing), whose gradient is the most sensitive to rounding: in 0.2.0, compiling the CUDA "
     "kernels without contraction into fused multiply-adds alone changed it by 7e-4, and NVIDIA OpenCL and POCL "
     "(OpenCL on the CPU) differ by 3e-5 there. The `legacy` mode differs less because its double-precision "
     "constants make part of its arithmetic FP64."),
)


def relative_l2(expected, actual):
    """``||actual - expected|| / ||expected||`` in float64 (0 when both are zero, 1 when only ``expected`` is)."""
    e, a = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    norm, diff = np.linalg.norm(e), np.linalg.norm(a - e)
    return float(diff / norm) if norm else float(diff > 0)


def kind_of(key):
    """The kind of an output key, for the summary: its first path element (``raw_gradient/vp`` -> ``raw_gradient``).

    The inversion outputs are grouped as ``model`` (the inverted models, ``model/vp``, ``b1/vp``,
    ``dm/vp``...) and ``misfit``.
    """
    head = key.split("/")[0]
    if head in ("raw_seismogram", "observed", "predicted", "misfit", "raw_gradient", "gradient"):
        return head
    return "misfit" if key.startswith("misfit") else "model"


def run(case, backend, numerics):
    """Outputs of one golden-master case on one backend and numeric mode (``test/golden_runner.py``)."""
    from golden_cases import PropagationCase
    from golden_runner import run_inversion_case, run_propagation_case

    os.environ["PYFWI_BACKEND"], os.environ["PYFWI_NUMERICS"] = backend, numerics
    with contextlib.redirect_stdout(io.StringIO()):
        if isinstance(case, PropagationCase):
            return run_propagation_case(case, guard=False)
        return run_inversion_case(case, guard=False)


def measure(modes, progress=print):
    """``{mode: {case name: {output key: relative L2 difference}}}`` on the whole matrix."""
    from golden_cases import inversion_cases, propagation_cases

    results = {}
    for mode in modes:
        results[mode] = {}
        for case in propagation_cases() + inversion_cases():
            reference, measured = (run(case, backend, mode) for backend in BACKENDS)
            results[mode][case.name] = {key: relative_l2(reference[key], measured[key]) for key in sorted(reference)}
            progress(f"{mode} {case.name}: largest {max(results[mode][case.name].values()):.1e}")
    return results


def summary(differences):
    """``{kind: (largest difference, case, key)}`` for the results of one numeric mode.

    The outputs of the inversion cases (``fwi_...``, ``tlfwi_...``) get their own kinds,
    ``inversion model`` and ``inversion misfit``.
    """
    largest = {}
    for case, outputs in differences.items():
        for key, value in outputs.items():
            kind = kind_of(key)
            if case.startswith(("fwi_", "tlfwi_")):
                kind = f"inversion {kind}"
            if kind not in largest or value > largest[kind][0]:
                largest[kind] = (value, case, key)
    return largest


def render(results, environment, release):
    """The Markdown report.

    Parameters
    ----------
    results : dict
        As returned by ``measure``.
    environment : dict
        ``{label: value}`` lines describing the machine and the versions.
    release : str
        The version of PyFWI measured, for the title.
    """
    lines = [f"# Parity of the CUDA backend with OpenCL (release {release})", ""]
    for paragraph in INTRODUCTION:
        lines += [paragraph, ""]
    lines += ["## Environment", ""]
    lines += [f"- {label}: {value}" for label, value in environment.items()]
    for mode, differences in results.items():
        lines += ["", f"## `{mode}` mode", "", "| Output | Largest difference | Case | Field |", "| --- | --- | --- | --- |"]
        for kind, (value, case, key) in sorted(summary(differences).items()):
            lines.append(f"| {kind} | {value:.1e} | {case} | `{key}` |")
        lines += ["", "Every output:", "", "| Case | Field | Difference |", "| --- | --- | --- |"]
        for case, outputs in differences.items():
            for key, value in outputs.items():
                lines.append(f"| {case} | `{key}` | {value:.1e} |")
    return "\n".join(lines) + "\n"


def environment():
    """Machine and versions, for the report."""
    import cupy
    import pyopencl

    import PyFWI
    from PyFWI.backends.cuda import CUDABackend
    return {"PyFWI": PyFWI.__version__, "GPU": CUDABackend().description,
            "CUDA runtime": cupy.cuda.runtime.runtimeGetVersion(), "NVRTC": ".".join(map(str, cupy.cuda.nvrtc.getVersion())),
            "CUDA driver": cupy.cuda.runtime.driverGetVersion(), "CuPy": cupy.__version__,
            "PyOpenCL": pyopencl.VERSION_TEXT, "NumPy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=["legacy", "fixed"], choices=["legacy", "fixed"])
    parser.add_argument("--out", help="the Markdown report (default: reports/parity-<version of PyFWI>.md)")
    args = parser.parse_args()
    import PyFWI
    out = args.out or REPORT.format(PyFWI.__version__)
    if "PYOPENCL_CTX" not in os.environ:
        from tools.make_golden import default_opencl_context
        default_opencl_context()
    results = measure(args.modes)
    with open(out, "w") as f:
        f.write(render(results, environment(), PyFWI.__version__))
    for mode, differences in results.items():
        for kind, (value, case, key) in sorted(summary(differences).items()):
            print(f"{mode:6} {kind:15} {value:.1e}  {case} {key}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
