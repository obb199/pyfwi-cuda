"""Storage of the golden masters: one ``.npz`` file per case and a JSON manifest.

Layout (``GOLDEN_DIR = test/golden``)::

    test/golden/manifest.json           what was generated, how, where and with what
    test/golden/data/<case name>.npz    the arrays of one case (keys such as "gradient/vp")

The manifest records, for reproducibility and for diagnosing a mismatch later:

* ``format`` -- version of this layout;
* ``generated`` -- UTC time stamp; ``generator_commit`` -- commit of this repository
  whose code generated the files;
* ``environment`` -- Python and package versions, where PyFWI was imported from, and
  the OpenCL platform, device and driver;
* ``rules`` -- the reproducibility rules (guard bands, geometry);
* ``cases`` -- per case: ``kind``, ``params`` (the case definition), ``file``,
  ``sha256`` of every array, ``deterministic`` (two runs gave the same bits),
  ``plain_run_matches`` (a plain run of 0.1.10 gave the same bits; null if not
  checked) and ``seconds`` (run time of one run).

Used by
-------
``tools/make_golden.py`` (writes) and the characterisation tests (read and verify).
"""
import hashlib
import json
import os

import numpy as np

#: Folder of the golden masters inside the repository.
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

#: Version of the storage layout described in the module docstring.
FORMAT = 1


def array_sha256(array):
    """SHA-256 of an array's dtype, shape and bytes (so that a reshaped or recast array changes the hash).

    Parameters
    ----------
    array : numpy.ndarray
        Any array.

    Returns
    -------
    str
        64 hexadecimal characters.
    """
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save_case(name, arrays, directory=GOLDEN_DIR):
    """Write the arrays of one case to ``<directory>/data/<name>.npz`` (compressed).

    Returns
    -------
    tuple
        ``(relative file path, {key: sha256})``, to be stored in the manifest.
    """
    relative = f"data/{name}.npz"
    path = os.path.join(directory, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return relative, {key: array_sha256(value) for key, value in arrays.items()}


def load_case(name, directory=GOLDEN_DIR):
    """Read the arrays of one case as a ``{key: numpy.ndarray}`` dict."""
    with np.load(os.path.join(directory, "data", f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def manifest_path(directory=GOLDEN_DIR):
    """Path of the manifest of a golden-master folder."""
    return os.path.join(directory, "manifest.json")


def load_manifest(directory=GOLDEN_DIR):
    """Read the manifest (a dict); returns None if the folder has no manifest yet."""
    path = manifest_path(directory)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_manifest(manifest, directory=GOLDEN_DIR):
    """Write the manifest as indented, key-sorted JSON (stable diffs in git)."""
    os.makedirs(directory, exist_ok=True)
    with open(manifest_path(directory), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")


def verify_case(name, manifest, directory=GOLDEN_DIR):
    """Check that the stored arrays of a case match the hashes of the manifest.

    Returns
    -------
    list of str
        Keys whose hash differs, or that are missing from either side (empty if all is well).
    """
    expected = manifest["cases"][name]["sha256"]
    arrays = load_case(name, directory)
    keys = set(expected) | set(arrays)
    return sorted(k for k in keys if k not in arrays or k not in expected or array_sha256(arrays[k]) != expected[k])
