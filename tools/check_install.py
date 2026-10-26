"""Check an installed copy of the fork (distribution ``pyfwi-cuda``, import name ``PyFWI``).

``tools/check_env.py`` checks a whole development environment (GPU, CUDA tool chain, a PyFWI run).
This script checks only the *package*, as a user gets it from a wheel, so it runs anywhere,
without a GPU or an OpenCL device:

1. the ``PyFWI`` package belongs to the ``pyfwi-cuda`` distribution, and the version in the
   distribution metadata is ``PyFWI.__version__``;
2. the files that are not Python modules are installed: the three OpenCL kernels (``*.cl``) and
   the rock-physics data of ``model_dataset`` (``data/*.mat``);
3. every module imports. Modules that need an optional dependency (extra) that is not installed
   are reported as skipped, not as failures: ``PyFWI.torchfwi`` and ``PyFWI.torch.*`` need
   ``torch`` (extra ``torch``), ``PyFWI.backends.cuda`` needs CuPy. ``segyio`` (extra ``segy``) is not needed to import anything.
4. the installed data file is usable: ``ModelGenerator("dupuy")`` reads it (with the wheel of
   0.1.10 this failed with ``FileNotFoundError``).

Usage (the script must not be run from inside ``src/``, or it would check the sources instead
of the installed copy)::

    python tools/check_install.py           # exit code 0 if everything is fine, 1 otherwise

Used by
-------
* the CPU CI job (``.github/workflows/ci.yml``), on a wheel installed in a fresh virtual
  environment with dependencies resolved from PyPI;
* ``test/test_packaging.py``, on a wheel installed in a fresh virtual environment by uv.
"""
import importlib
import importlib.metadata
import importlib.util
import os
import sys

#: Distribution name of the fork on the package index (``pip install pyfwi-cuda``).
DISTRIBUTION = "pyfwi-cuda"

#: Files of the ``PyFWI`` package that are not Python modules, relative to the package folder.
DATA_FILES = ("elastic.cl", "elastic_crosswell.cl", "elastic_surface.cl", "data/Dupuy2011_Moduli_data.mat",
              "kernels/opencl_on_cuda.h", "kernels/elastic.cu")

#: Optional dependency (import name) that each module needs, for modules that need one.
OPTIONAL = {"PyFWI.torchfwi": "torch", "PyFWI.torch.wave_propagation": "torch", "PyFWI.backends.cuda": "cupy"}


def modules(folder):
    """Dotted names of every module of the package installed in ``folder`` (the ``PyFWI`` folder).

    Found from the files, not from a hard-coded list, so that a module added later is checked too.
    """
    names = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        package = ["PyFWI", *os.path.relpath(root, folder).split(os.sep)]
        package = [p for p in package if p != "."]
        for file in sorted(files):
            if file.endswith(".py"):
                stem = file[:-3]
                names.append(".".join(package if stem == "__init__" else [*package, stem]))
    return names


def check():
    """Run the checks; returns a list of ``(status, item, detail)`` with status "ok", "skip" or "FAIL"."""
    results = []
    import PyFWI
    folder = os.path.dirname(os.path.abspath(PyFWI.__file__))
    results.append(("ok", "PyFWI imported from", folder))

    owners = importlib.metadata.packages_distributions().get("PyFWI", [])
    if DISTRIBUTION in owners:
        version = importlib.metadata.version(DISTRIBUTION)
        same = version == PyFWI.__version__
        results.append(("ok" if same else "FAIL", f"distribution {DISTRIBUTION} {version}",
                         "same as PyFWI.__version__" if same else f"but PyFWI.__version__ is {PyFWI.__version__}"))
    else:
        results.append(("FAIL", f"distribution {DISTRIBUTION}", f"PyFWI belongs to {owners or 'no distribution'}"))

    for name in DATA_FILES:
        present = os.path.isfile(os.path.join(folder, name))
        results.append(("ok" if present else "FAIL", f"file PyFWI/{name}", "present" if present else "missing"))

    for module in modules(folder):
        needed = OPTIONAL.get(module)
        if needed and importlib.util.find_spec(needed) is None:
            results.append(("skip", f"import {module}", f"optional dependency {needed} not installed"))
            continue
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 (report every failure, not only the first one)
            results.append(("FAIL", f"import {module}", f"{type(error).__name__}: {error}"))
        else:
            results.append(("ok", f"import {module}", ""))

    item = "model 'dupuy' (reads data/*.mat)"
    try:
        from PyFWI.model_dataset import ModelGenerator
        vp = ModelGenerator("dupuy")()["vp"]
    except Exception as error:  # noqa: BLE001 (any error means the installed data is not usable)
        results.append(("FAIL", item, f"{type(error).__name__}: {error}"))
    else:
        results.append(("ok", item, f"vp of {vp.shape[0]} x {vp.shape[1]} cells"))
    return results


def main():
    results = check()
    width = max(len(item) for _, item, _ in results)
    for status, item, detail in results:
        print(f"[{status:>4}] {item:<{width}}  {detail}")
    failed = sum(status == "FAIL" for status, _, _ in results)
    print(f"{len(results)} checks, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
