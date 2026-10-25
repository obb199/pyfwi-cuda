"""Integrity of the stored golden masters (``test/golden/``), no device needed.

The golden masters are the reference every later phase is compared with, so they
must be complete, untampered and traceable to the published PyFWI 0.1.10.
"""
import os

import pytest

from golden_cases import all_cases
from golden_store import GOLDEN_DIR, load_manifest, verify_case

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Upper limit for the size of the golden-master folder (above it, use Git LFS; roadmap step 0.4).
MAX_BYTES = 20 * 1024 * 1024

MANIFEST = load_manifest()


def test_manifest_exists_and_records_how_it_was_made():
    assert MANIFEST is not None, "run ../envs/ref-0.1.10/bin/python tools/make_golden.py --check-plain"
    assert MANIFEST["format"] == 1 and MANIFEST["generated"]
    assert MANIFEST["generator_commit"] and not MANIFEST["generator_commit"].endswith("+dirty")
    assert MANIFEST["rules"]["guard_bands"] == "zero" and MANIFEST["rules"]["geometry"].startswith("race-free")


def test_golden_masters_come_from_the_published_0_1_10():
    env = MANIFEST["environment"]
    assert env["pyfwi"] == "0.1.10"
    assert not os.path.abspath(env["pyfwi_path"]).startswith(REPO + os.sep), "made from the fork's code"
    assert env["scipy"] == "1.7.3" and env["numpy"] == "1.21.6"  # the frozen reference environment


def test_every_case_of_the_matrix_is_stored_with_the_same_definition():
    stored = {name: entry["params"] for name, entry in MANIFEST["cases"].items()}
    expected = {case.name: case.as_dict() for case in all_cases()}
    assert stored == expected, "the case matrix changed: regenerate the golden masters"


def test_cases_are_of_the_three_kinds():
    kinds = {entry["kind"] for entry in MANIFEST["cases"].values()}
    assert kinds == {"propagation", "inversion", "example"}


@pytest.mark.parametrize("name", sorted(MANIFEST["cases"]) if MANIFEST else [])
def test_case_files_match_their_hashes(name):
    entry = MANIFEST["cases"][name]
    assert entry["deterministic"] is True and entry["plain_run_matches"] in (True, False, None)
    assert verify_case(name, MANIFEST) == []


def test_golden_folder_is_small_enough_for_git():
    size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(GOLDEN_DIR) for f in files)
    assert size < MAX_BYTES, f"{size / 2**20:.1f} MiB: move the golden masters to Git LFS"
