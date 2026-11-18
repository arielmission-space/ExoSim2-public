"""Tests for the ExoSim MCP lexical retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from exosim.mcp import retrieval
from exosim.mcp.retrieval import search_knowledge_sources, tokenize

if TYPE_CHECKING:
    from pathlib import Path


def test_tokenize_splits_identifiers() -> None:
    tokens = tokenize("LoadOptions and wl_grid drive PSF sampling")
    assert {"loadoptions", "load", "options", "wl", "grid", "psf"} <= set(tokens)
    assert "and" not in tokens  # stopword


def test_tokenize_drops_single_characters_and_stopwords() -> None:
    assert tokenize("a of x the") == []


@pytest.fixture
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny knowledge base with a single retrievable file."""
    root = tmp_path / "repo"
    (root / "docs" / "source").mkdir(parents=True)
    (root / "AGENTS.md").write_text("Grounding rules.\n", encoding="utf-8")
    (root / "docs" / "source" / "quickstart.rst").write_text(
        "Quickstart\n==========\n\n"
        "Run the focal plane recipe first.\n\n"
        "Radiometric model\n-----------------\n\n"
        "The radiometric model computes photon noise and the signal.\n",
        encoding="utf-8",
    )
    retrieval._CACHE.clear()
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(root))
    return root


def test_ranks_the_relevant_section_first(kb: Path) -> None:
    sources = search_knowledge_sources("radiometric photon noise", max_results=3)

    assert sources
    assert sources[0].path == "docs/source/quickstart.rst"
    assert "radiometric model computes photon noise" in sources[0].text.lower()
    assert sources[0].score == pytest.approx(1.0)


def test_rst_chunking_follows_sections(kb: Path) -> None:
    sources = search_knowledge_sources("radiometric model", max_results=5)
    hit = next(s for s in sources if "photon noise" in s.text.lower())
    # the chunk starts at the "Radiometric model" heading, not the top of file
    assert hit.text.lower().startswith("radiometric model")


def test_unmatched_query_returns_nothing(kb: Path) -> None:
    assert search_knowledge_sources("quxzzy frobnicated blorptastic") == []


def test_index_cache_invalidates_on_change(kb: Path) -> None:
    assert search_knowledge_sources("telescope aperture diameter") == []

    (kb / "docs" / "source" / "telescope.rst").write_text(
        "Telescope\n=========\n\nThe telescope aperture diameter sets the collecting area.\n",
        encoding="utf-8",
    )
    sources = search_knowledge_sources("telescope aperture diameter")

    assert sources
    assert sources[0].path == "docs/source/telescope.rst"


def test_process_docs_are_excluded_from_the_real_kb() -> None:
    """The MCP build/process notes are not usable ExoSim knowledge and their
    vocabulary (editor names, roadmap jargon) causes false matches."""
    from exosim.mcp.retrieval import knowledge_base_manifest

    matched = set(knowledge_base_manifest()["matched_files"])
    assert "docs/ai-agents/mcp-hardening-roadmap.md" not in matched
    assert "docs/ai-agents/mcp-release-plan.md" not in matched
    assert "docs/ai-agents/mcp-recipe-metadata.md" not in matched
    # the user-facing agent guides stay in
    assert "docs/ai-agents/parameter-setup-guide.md" in matched
