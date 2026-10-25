"""Consistência da documentação da migração: links, âncoras e commits citados."""
import glob
import os
import re
import shutil
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = sorted(glob.glob(os.path.join(REPO, "docs", "migracao-cuda", "**", "*.md"), recursive=True))
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading):
    """Âncora no estilo do GitHub: minúsculas, sem pontuação, espaços viram hífens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(path):
    with open(path, encoding="utf-8") as f:
        return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", f.read(), re.MULTILINE)}


def test_slug_matches_github_style():
    assert slug("Pendências que dependem de você") == "pendências-que-dependem-de-você"
    assert slug("Fase 1: correções de host no OpenCL (versão 0.1.11 do fork)") == \
        "fase-1-correções-de-host-no-opencl-versão-0111-do-fork"


def test_docs_exist():
    names = {os.path.basename(p) for p in DOCS}
    assert {"analise-e-plano.md", "roteiro-de-execucao.md"} <= names


@pytest.mark.parametrize("doc", DOCS, ids=os.path.basename)
def test_relative_links_and_anchors_resolve(doc):
    with open(doc, encoding="utf-8") as f:
        text = f.read()
    broken = []
    for target in LINK.findall(text):
        if re.match(r"[a-z]+://|mailto:", target):
            continue  # links externos não são verificados aqui (sem rede no CI)
        path, _, anchor = target.partition("#")
        dest = os.path.normpath(os.path.join(os.path.dirname(doc), path)) if path else doc
        missing_anchor = bool(anchor) and dest.endswith(".md") and os.path.exists(dest) and anchor not in anchors(dest)
        if not os.path.exists(dest) or missing_anchor:
            broken.append(target)
    assert not broken, f"links quebrados em {os.path.basename(doc)}: {broken}"


@pytest.mark.skipif(shutil.which("git") is None or not os.path.isdir(os.path.join(REPO, ".git")),
                    reason="sem git")
@pytest.mark.parametrize("doc", DOCS, ids=os.path.basename)
def test_cited_commits_exist(doc):
    with open(doc, encoding="utf-8") as f:
        hashes = set(re.findall(r"`([0-9a-f]{7,40})`", f.read()))
    missing = [h for h in hashes
               if subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"{h}^{{commit}}"],
                                 check=False, capture_output=True).returncode != 0]
    assert not missing, f"commits citados que não existem: {missing}"
