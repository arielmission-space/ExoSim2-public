"""Tests for the optional hybrid (BM25 + embedding) retrieval backend.

A deterministic fake encoder stands in for sentence-transformers so the tests
need no model download and no extra dependency.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from exosim.mcp import embeddings, retrieval
from exosim.mcp.retrieval import active_backend, search_knowledge_sources

if TYPE_CHECKING:
    from pathlib import Path

# A tiny "concept" space: each axis is a topic keyword. Paraphrases that share a
# concept (shot noise / photon noise -> "noise") end up close.
_CONCEPTS = ["noise", "photon", "telescope", "aperture", "detector", "wavelength"]
_SYNONYMS = {"shot": "noise", "mirror": "telescope", "diameter": "aperture"}


class FakeEncoder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(len(_CONCEPTS), dtype="float32")
            for word in re.findall(r"[a-z]+", text.lower()):
                word = _SYNONYMS.get(word, word)
                if word in _CONCEPTS:
                    vec[_CONCEPTS.index(word)] += 1.0
            rows.append(vec)
        return np.asarray(rows, dtype="float32")


@pytest.fixture
def hybrid_kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "docs" / "source").mkdir(parents=True)
    (root / "AGENTS.md").write_text("Rules.\n", encoding="utf-8")
    (root / "docs" / "source" / "noise.rst").write_text(
        "Noise\n=====\n\n"
        "The photon noise is the square root of the detected photon count.\n",
        encoding="utf-8",
    )
    (root / "docs" / "source" / "optics.rst").write_text(
        "Optics\n======\n\nThe telescope aperture diameter sets the collecting area.\n",
        encoding="utf-8",
    )

    retrieval._CACHE.clear()
    retrieval._EMBED_CACHE.clear()
    monkeypatch.setattr(retrieval, "_encoder", FakeEncoder())
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(root))
    monkeypatch.setenv("EXOSIM_MCP_RETRIEVAL", "hybrid")
    return root


def test_encoder_cosine_scores_are_bounded() -> None:
    index = embeddings.EmbeddingIndex(
        ["photon noise", "telescope aperture"], FakeEncoder()
    )
    scores = index.cosine_scores("shot noise")
    assert scores.shape == (2,)
    assert scores[0] > scores[1]  # "shot" -> "noise" concept
    assert -1.0 <= float(scores.min()) <= float(scores.max()) <= 1.0


def test_hybrid_backend_is_active_with_an_encoder(hybrid_kb: Path) -> None:
    assert active_backend() == "hybrid"


def test_hybrid_finds_a_paraphrase_with_no_shared_tokens(hybrid_kb: Path) -> None:
    # "shot noise" shares no token with "photon noise" but is semantically close
    sources = search_knowledge_sources("shot noise level", max_results=3)
    assert sources
    assert sources[0].path == "docs/source/noise.rst"


def test_hybrid_still_rejects_a_nonsense_query(hybrid_kb: Path) -> None:
    assert search_knowledge_sources("quxzzy frobnicated blorptastic") == []


def test_backend_falls_back_to_bm25_without_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrieval, "_encoder", None)
    monkeypatch.setattr(retrieval, "_load_embeddings", lambda: None)
    monkeypatch.setenv("EXOSIM_MCP_RETRIEVAL", "hybrid")
    assert active_backend() == "bm25"
