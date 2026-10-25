"""Generate the golden masters of PyFWI 0.1.10 (roadmap step 0.4).

What it does
------------
For every case of ``test/golden_cases.py`` (or the ones selected with ``--cases`` or
``--kinds``): propagation cases (forward + gradient), inversion cases (short FWI and
time-lapse FWI) and example cases (the outputs of ``example/scripts/*.py``):

1. runs it twice with ``test/golden_runner.py`` (zero guard bands, race-free
   geometry) and stops with an error if the two runs differ in any bit;
2. optionally (``--check-plain``, propagation cases only) runs it once more
   *plainly*, without guard bands, in a fresh process, and records whether 0.1.10
   itself gives the same bits;
3. stores the arrays in ``<out>/data/<case>.npz`` and their SHA-256 in
   ``<out>/manifest.json``, together with the versions, the device and the commit
   (see ``test/golden_store.py``).

Safety
------
Golden masters must come from the **published** PyFWI 0.1.10, never from the
fork's code. The script refuses to run unless ``PyFWI`` is version 0.1.10 and is
imported from outside this repository, i.e. from the reference environment.
``--allow-fork`` lifts this check to test the generator itself; then ``--out``
is required, so that the real golden masters cannot be overwritten.

Usage
-----
::

    ../envs/ref-0.1.10/bin/python tools/make_golden.py --check-plain         # all cases
    ../envs/ref-0.1.10/bin/python tools/make_golden.py --cases a1_c0_o8_eb1_chpr20_lmd
    ../envs/dev/bin/python tools/make_golden.py --allow-fork --out /tmp/g --cases ...

The OpenCL device is the one selected by ``PYOPENCL_CTX`` (default here: the NVIDIA
platform if there is one).

Used by
-------
Developers, once per change of the case matrix. Tested by ``test/test_golden_store.py``.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(REPO, "test")
sys.path[:0] = [TEST_DIR, REPO]  # golden_cases, golden_runner, golden_store, helpers, oob_guard, tools

from golden_cases import FACTORS, ExampleCase, InversionCase, PropagationCase, all_cases
from golden_runner import run_case
from golden_store import FORMAT, GOLDEN_DIR, save_case, write_manifest

#: Program run in a fresh process for --check-plain: one plain run of a case, saved to a .npz.
#: argv: <test folder> <repository> <case name> <output .npz>
PLAIN_RUN = """
import sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
from golden_cases import propagation_cases
from golden_runner import run_propagation_case
case = {c.name: c for c in propagation_cases()}[sys.argv[3]]  # plain runs: propagation cases only
np.savez(sys.argv[4], **run_propagation_case(case, guard=False))
"""


#: Kind recorded in the manifest for each type of case.
KINDS = {PropagationCase: "propagation", InversionCase: "inversion", ExampleCase: "example"}


def pyfwi_origin():
    """Return ``(version, folder)`` of the PyFWI package that Python imports."""
    import PyFWI
    return PyFWI.__version__, os.path.dirname(os.path.abspath(PyFWI.__file__))


def default_opencl_context():
    """Set ``PYOPENCL_CTX`` to the NVIDIA platform when it is not set (first platform otherwise)."""
    if "PYOPENCL_CTX" in os.environ:
        return
    import pyopencl as cl
    platforms = cl.get_platforms()
    index = next((i for i, p in enumerate(platforms) if "NVIDIA" in p.name), 0)
    n_devices = len(platforms[index].get_devices())
    os.environ["PYOPENCL_CTX"] = f"{index}" if n_devices == 1 else f"{index}:0"


def environment():
    """Versions and device information recorded in the manifest."""
    import pyopencl as cl
    import scipy
    version, folder = pyfwi_origin()
    answer = os.environ["PYOPENCL_CTX"].split(":")
    plat = cl.get_platforms()[int(answer[0])]
    dev = plat.get_devices()[int(answer[1]) if len(answer) > 1 else 0]
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pyopencl": cl.VERSION_TEXT, "pyfwi": version, "pyfwi_path": folder,
            "opencl_platform": plat.name, "opencl_device": dev.name, "driver": dev.driver_version,
            "machine": platform.machine(), "system": platform.system()}


def git_commit():
    """Commit hash of this repository (``+dirty`` if it has uncommitted changes), or None."""
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], check=True, capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], check=True,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def plain_run_matches(case, arrays):
    """Run ``case`` plainly in a fresh process and tell whether it gives the same bits as ``arrays``."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "plain.npz")
        r = subprocess.run([sys.executable, "-c", PLAIN_RUN, TEST_DIR, REPO, case.name, out],
                           capture_output=True, text=True, timeout=600, check=False)
        if r.returncode != 0:
            return False  # a plain run that crashes does not match
        with np.load(out) as plain:
            return set(plain.files) == set(arrays) and all(np.array_equal(plain[k], arrays[k]) for k in arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", help="names of the cases to generate (default: all)")
    parser.add_argument("--kinds", nargs="*", choices=sorted(KINDS.values()),
                        help="kinds of cases to generate (default: all)")
    parser.add_argument("--check-plain", action="store_true", help="also compare with a plain run of each case")
    parser.add_argument("--out", help=f"output folder (default: {os.path.relpath(GOLDEN_DIR, REPO)})")
    parser.add_argument("--allow-fork", action="store_true", help="allow a PyFWI other than the published 0.1.10")
    args = parser.parse_args(argv)

    version, folder = pyfwi_origin()
    is_reference = version == "0.1.10" and not folder.startswith(REPO + os.sep)
    if not is_reference and not args.allow_fork:
        parser.error(f"PyFWI {version} from {folder} is not the published 0.1.10: run with the reference "
                     "environment (../envs/ref-0.1.10/bin/python), or pass --allow-fork --out <folder>")
    if args.allow_fork and not args.out:
        parser.error("--allow-fork needs --out, so that the real golden masters are never overwritten")
    out = os.path.abspath(args.out or GOLDEN_DIR)
    default_opencl_context()

    cases = all_cases()
    if args.kinds:
        cases = [c for c in cases if KINDS[type(c)] in args.kinds]
    if args.cases:
        by_name = {c.name: c for c in cases}
        unknown = [n for n in args.cases if n not in by_name]
        if unknown:
            parser.error(f"unknown cases: {unknown}")
        cases = [by_name[n] for n in args.cases]

    manifest = {"format": FORMAT, "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "generator_commit": git_commit(), "environment": environment(),
                "rules": {"guard_bands": "zero", "geometry": "race-free (test/helpers.py::race_free_geometry)"},
                "factors": {k: list(v) for k, v in FACTORS.items()}, "cases": {}}
    for case in cases:
        start = time.perf_counter()
        first = run_case(case)
        seconds = time.perf_counter() - start
        second = run_case(case)
        different = [k for k in first if not np.array_equal(first[k], second[k])]
        if different:
            sys.exit(f"{case.name}: two runs differ in {different}; refusing to store a non-deterministic golden master")
        plain = plain_run_matches(case, first) if args.check_plain and isinstance(case, PropagationCase) else None
        relative, hashes = save_case(case.name, first, out)
        manifest["cases"][case.name] = {"kind": KINDS[type(case)], "params": case.as_dict(), "file": relative,
                                        "sha256": hashes, "deterministic": True, "plain_run_matches": plain,
                                        "seconds": round(seconds, 3)}
        print(f"{case.name}: {len(first)} arrays, {seconds:.2f} s, plain run matches: {plain}", flush=True)
    write_manifest(manifest, out)
    print(f"wrote {len(cases)} case(s) to {out}")


if __name__ == "__main__":
    main()
