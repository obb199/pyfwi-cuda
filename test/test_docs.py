"""Consistency of the migration documentation: links, anchors and cited commits.

Documentation rots silently: a renamed file or heading breaks a link, and a
rebase can make a commit hash cited in the roadmap disappear. These tests catch
both without network access (external ``http(s)://`` links are not checked).

Checked files: every ``.md`` under ``docs/cuda-migration/``, plus ``README.md`` and
``CHANGELOG.md`` at the repository root.
"""
import glob
import os
import re
import shutil
import subprocess

import pytest

#: Absolute path of the repository root.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Markdown files whose links and commit hashes are checked.
DOCS = sorted(glob.glob(os.path.join(REPO, "docs", "cuda-migration", "**", "*.md"), recursive=True)) + [
    os.path.join(REPO, "README.md"), os.path.join(REPO, "CHANGELOG.md")]

#: A Markdown link ``[text](target)`` that is not an image (``![alt](src)``);
#: group 1 is the target.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading):
    """Return the anchor GitHub generates for a heading.

    Lower case, punctuation removed (letters with accents are kept), spaces
    turned into hyphens. Example: ``"Gate G0"`` -> ``"gate-g0"``.

    Used by ``anchors()`` and ``test_slug_matches_github_style``.
    """
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(path):
    """Return the set of anchors (slugs of every ``#`` heading) of a Markdown file."""
    with open(path, encoding="utf-8") as f:
        return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", f.read(), re.MULTILINE)}


def test_slug_matches_github_style():
    assert slug("Pending actions for the project owner") == "pending-actions-for-the-project-owner"
    assert slug("Phase 1: host fixes on OpenCL (fork version 0.1.11)") == \
        "phase-1-host-fixes-on-opencl-fork-version-0111"


def test_docs_exist():
    names = {os.path.basename(p) for p in DOCS if os.path.exists(p)}
    assert {"analysis-and-plan.md", "execution-roadmap.md", "README.md", "CHANGELOG.md"} <= names


@pytest.mark.parametrize("doc", DOCS, ids=os.path.basename)
def test_relative_links_and_anchors_resolve(doc):
    with open(doc, encoding="utf-8") as f:
        text = f.read()
    broken = []
    for target in LINK.findall(text):
        if re.match(r"[a-z]+://|mailto:", target):
            continue  # external links are not checked here (no network in CI)
        path, _, anchor = target.partition("#")
        dest = os.path.normpath(os.path.join(os.path.dirname(doc), path)) if path else doc
        missing_anchor = bool(anchor) and dest.endswith(".md") and os.path.exists(dest) and anchor not in anchors(dest)
        if not os.path.exists(dest) or missing_anchor:
            broken.append(target)
    assert not broken, f"broken links in {os.path.basename(doc)}: {broken}"


@pytest.mark.skipif(shutil.which("git") is None or not os.path.isdir(os.path.join(REPO, ".git")),
                    reason="no git")
@pytest.mark.parametrize("doc", DOCS, ids=os.path.basename)
def test_cited_commits_exist(doc):
    # A commit is cited as a 7- to 40-character hexadecimal string in backticks, e.g. `1434a16`.
    with open(doc, encoding="utf-8") as f:
        hashes = set(re.findall(r"`([0-9a-f]{7,40})`", f.read()))
    missing = [h for h in hashes
               if subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"{h}^{{commit}}"],
                                 check=False, capture_output=True).returncode != 0]
    assert not missing, f"cited commits that do not exist: {missing}"
