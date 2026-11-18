"""Optional semantic retrieval backend for the ExoSim MCP knowledge base.

BM25 (in :mod:`exosim.mcp.retrieval`) is excellent for identifiers and exact
phrases but misses paraphrase ("photon noise" vs "shot noise"). When the
``exosim[mcp-embeddings]`` extra is installed and ``EXOSIM_MCP_RETRIEVAL=hybrid``
is set, this module adds a sentence-embedding score that is combined with BM25.

The embedding index is held in memory, keyed by the same file-mtime signature as
the BM25 index, so it is rebuilt only when a source file changes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# A candidate must clear this cosine similarity to be admitted on semantics
# alone; below it, only a lexical (BM25) match counts. Keeps a nonsense query
# from matching everything a little.
MIN_COSINE = 0.35


class Encoder(Protocol):
    """Anything with a sentence-transformers-style ``encode``."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


class EmbeddingsUnavailableError(RuntimeError):
    """Raised when the embeddings extra is requested but not installed."""


def load_encoder() -> Encoder:
    """Load the default sentence-embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise EmbeddingsUnavailableError(
            "hybrid retrieval needs the embeddings extra: "
            "pip install 'exosim[mcp-embeddings]'"
        ) from exc
    return SentenceTransformer(MODEL_NAME)


def _unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingIndex:
    """Unit-normalised chunk embeddings with a cosine search."""

    def __init__(self, texts: list[str], encoder: Encoder) -> None:
        self._encoder = encoder
        self._vectors = _unit(encoder.encode(texts))

    def cosine_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` to every indexed chunk."""
        vec = _unit(self._encoder.encode([query]))[0]
        return self._vectors @ vec
