"""Repository hygiene of the fork: licence, modification notice, changelog, CI safety.

Why these tests exist
---------------------
PyFWI is licensed under the GPLv3, which requires a modified version to keep the
licence text intact and to "carry prominent notices stating that you modified it,
and giving a relevant date" (section 5a). The fork must also never publish to PyPI
under the name of the original package by accident. These tests turn those
obligations into checks that run on every commit (the CI-related checks are in
``test_ci.py``).
"""
import hashlib
import os
import re

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: SHA-256 of LICENSE and CITATION.cff in upstream PyFWI 0.1.10 (commit 1434a16).
#: Both files must stay byte-for-byte identical to the original.
UPSTREAM_SHA256 = {
    "LICENSE": "2fcbc59fd6474f7451a53b73bb38648034eeb71fcd3613330b0ffd6296d94784",
    "CITATION.cff": "8a4b6904ddfbe89b8c87c1c64562acb61d22b4bcb9e63eff58ef915a9b88490c",
}


def _read(name):
    """Return the text of a file given by its path relative to the repository root."""
    with open(os.path.join(REPO, name), encoding="utf-8") as f:
        return f.read()


def test_license_and_citation_are_unchanged_from_upstream():
    for name, expected in UPSTREAM_SHA256.items():
        with open(os.path.join(REPO, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == expected, f"{name} differs from upstream"


def test_readme_carries_the_gplv3_modification_notice():
    readme = _read("README.md")
    assert "modified version of PyFWI" in readme
    assert "https://github.com/AmirMardan/PyFWI" in readme  # where the original comes from
    assert re.search(r"since (January|February|March|April|May|June|July|August|September|October|"
                     r"November|December) \d{1,2}, \d{4}", readme), "the notice must give a date"
    assert "GNU General" in readme and "version 3" in readme
    assert "mardan2023pyfwi" in readme  # the citation of the original paper is kept


def test_changelog_has_an_unreleased_section_and_flags_numerical_changes():
    changelog = _read("CHANGELOG.md")
    assert "## [Unreleased]" in changelog
    # Every release section must say whether results changed.
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)[1:]
    assert sections and all("### Numerical changes" in s for s in sections)
