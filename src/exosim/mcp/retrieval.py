"""Local retrieval for grounded ExoSim MCP answers.

Retrieval is deliberately dependency-light: a BM25 ranker over structure-aware
chunks of a curated set of files. It is *lexical* — good for identifiers,
parameter names and exact phrases — and reports an explicit
``insufficient_evidence`` status when a query matches nothing. A future
embedding backend can be added behind an optional extra without changing this
contract.
"""

from __future__ import annotations

import itertools
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exosim.mcp.schemas import (
    KnowledgeBaseSourcePattern,
    KnowledgeSource,
    resolve_repo_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

DEFAULT_SOURCE_PATTERNS = (
    KnowledgeBaseSourcePattern(
        "AGENTS.md",
        "Repository-level rules for AI assistants and scientific grounding.",
    ),
    KnowledgeBaseSourcePattern(
        "README.md",
        "High-level project installation and contribution context.",
    ),
    KnowledgeBaseSourcePattern(
        "CHANGELOG.rst",
        "Released changes, feature history and behavioural notes.",
    ),
    KnowledgeBaseSourcePattern(
        "docs/ai-agents/*.md",
        "Curated agent handover notes, parameter guidance and playbooks.",
    ),
    KnowledgeBaseSourcePattern(
        "docs/source/*.rst",
        "Top-level user documentation (installation, quickstart, FAQs).",
    ),
    KnowledgeBaseSourcePattern(
        "docs/source/user/**/*.rst",
        "User-facing documentation for ExoSim pipelines and outputs.",
    ),
    KnowledgeBaseSourcePattern(
        "docs/source/contributing/**/*.rst",
        "Developer documentation for tasks, recipes and runtime configuration.",
    ),
    KnowledgeBaseSourcePattern("examples/**/*.xml", "Documented XML examples."),
    KnowledgeBaseSourcePattern(
        "src/exosim/recipes/*.py",
        "Recipe source code for pipeline behaviour.",
    ),
    KnowledgeBaseSourcePattern(
        "src/exosim/tasks/load/*.py",
        "Configuration loader source code.",
    ),
    KnowledgeBaseSourcePattern(
        "src/exosim/tasks/radiometric/**/*.py",
        "Radiometric task source code and implemented physics helpers.",
    ),
)

# Files that a pattern above would pull in but which are about *building* the
# MCP server / project process, not about *using* ExoSim. They add no usable
# knowledge and their meta-vocabulary (editor names, roadmap jargon) produces
# false lexical matches for unrelated queries.
_EXCLUDED_SOURCES = frozenset(
    {
        "docs/ai-agents/mcp-hardening-roadmap.md",
        "docs/ai-agents/mcp-recipe-metadata.md",
        "docs/ai-agents/mcp-release-plan.md",
    }
)

# A passage must contain at least this fraction of the distinct query terms
# (and at least one) to count as evidence.
MINIMUM_RETRIEVAL_SCORE = 0.15

_MAX_CHUNK_LINES = 120
_MAX_CHUNK_CHARS = 3000
_FALLBACK_WINDOW = 60
_BM25_K1 = 1.5
_BM25_B = 0.75

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_RST_UNDERLINE_RE = re.compile(r"^([=~^\"'`#*+.-])\1{2,}\s*$")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s")
_PY_DEF_RE = re.compile(r"^(async\s+def\s|def\s|class\s)")

# fmt: off
_STOPWORDS = frozenset({
    "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "do", "does", "this", "these", "those", "can", "could",
    "should", "would", "will",
})
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, with identifiers also split into sub-tokens.

    ``LoadOptions`` yields ``loadoptions``, ``load`` and ``options``;
    ``wl_grid`` yields ``wl`` and ``grid``. Stopwords and single characters are
    dropped.
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        tokens.append(word.lower())
        parts = _CAMEL_RE.findall(word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


@dataclass
class _Chunk:
    path: str
    start_line: int
    end_line: int
    text: str
    tf: Counter[str] = field(default_factory=Counter)
    length: int = 0

    @property
    def terms(self) -> set[str]:
        return set(self.tf)


@dataclass
class _Index:
    chunks: list[_Chunk]
    df: Counter[str]
    n_docs: int
    avgdl: float

    def bm25(self, query_terms: list[str], chunk: _Chunk) -> float:
        score = 0.0
        for term in query_terms:
            freq = chunk.tf.get(term, 0)
            if not freq:
                continue
            df = self.df.get(term, 0) or 1
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            denom = freq + _BM25_K1 * (
                1 - _BM25_B + _BM25_B * chunk.length / (self.avgdl or 1)
            )
            score += idf * (freq * (_BM25_K1 + 1)) / denom
        return score


_CACHE: dict[str, tuple[frozenset[tuple[str, int]], _Index]] = {}


def _iter_source_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for source_pattern in DEFAULT_SOURCE_PATTERNS:
        for path in sorted(root.glob(source_pattern.pattern)):
            if path in seen or not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue  # skip .ipynb_checkpoints, .git, ...
            if path.relative_to(root).as_posix() in _EXCLUDED_SOURCES:
                continue
            seen.add(path)
            yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def _chunk_boundaries(path: Path, lines: list[str]) -> list[int]:
    suffix = path.suffix.lower()
    starts: list[int] = [0]
    if suffix == ".py":
        starts += [i for i, line in enumerate(lines) if _PY_DEF_RE.match(line)]
    elif suffix in {".rst", ".txt"}:
        starts += [
            i - 1
            for i in range(1, len(lines))
            if _RST_UNDERLINE_RE.match(lines[i]) and lines[i - 1].strip()
        ]
    elif suffix in {".md", ".markdown"}:
        starts += [i for i, line in enumerate(lines) if _MD_HEADING_RE.match(line)]
    else:
        starts += list(range(_FALLBACK_WINDOW, len(lines), _FALLBACK_WINDOW))
    ordered = sorted({s for s in starts if 0 <= s < len(lines)})
    return ordered or [0]


def _split_long(start: int, block: list[str]) -> Iterable[tuple[int, list[str]]]:
    if len(block) <= _MAX_CHUNK_LINES:
        yield start, block
        return
    for offset in range(0, len(block), _MAX_CHUNK_LINES):
        yield start + offset, block[offset : offset + _MAX_CHUNK_LINES]


def _chunk_file(rel_path: str, path: Path) -> Iterable[_Chunk]:
    lines = _read_text(path).splitlines()
    if not lines:
        return
    boundaries = _chunk_boundaries(path, lines)
    boundaries.append(len(lines))
    for begin, end in itertools.pairwise(boundaries):
        if end <= begin:
            continue
        for sub_start, block in _split_long(begin, lines[begin:end]):
            text = "\n".join(block).strip()
            if not text:
                continue
            tokens = tokenize(text[:_MAX_CHUNK_CHARS])
            if not tokens:
                continue
            yield _Chunk(
                path=rel_path,
                start_line=sub_start + 1,
                end_line=sub_start + len(block),
                text=text[:_MAX_CHUNK_CHARS],
                tf=Counter(tokens),
                length=len(tokens),
            )


def _build_index(root: Path) -> _Index:
    chunks: list[_Chunk] = []
    df: Counter[str] = Counter()
    for path in _iter_source_files(root):
        rel = path.relative_to(root).as_posix()
        for chunk in _chunk_file(rel, path):
            chunks.append(chunk)
            df.update(chunk.terms)
    total = sum(chunk.length for chunk in chunks)
    return _Index(
        chunks=chunks,
        df=df,
        n_docs=len(chunks),
        avgdl=(total / len(chunks)) if chunks else 0.0,
    )


def _signature(root: Path) -> frozenset[tuple[str, int]]:
    entries: set[tuple[str, int]] = set()
    for path in _iter_source_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.add((path.as_posix(), stat.st_mtime_ns))
    return frozenset(entries)


def _get_index(root: Path, signature: frozenset[tuple[str, int]]) -> _Index:
    key = root.as_posix()
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    index = _build_index(root)
    _CACHE[key] = (signature, index)
    return index


ENV_RETRIEVAL = "EXOSIM_MCP_RETRIEVAL"
_EMBED_CACHE: dict[str, tuple[frozenset[tuple[str, int]], object]] = {}
_encoder: object | None = None


def active_backend() -> str:
    """Return the retrieval backend actually in use (``bm25`` or ``hybrid``)."""
    if os.environ.get(ENV_RETRIEVAL, "bm25").strip().lower() != "hybrid":
        return "bm25"
    return "hybrid" if _load_embeddings() is not None else "bm25"


def _load_embeddings() -> object | None:
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from exosim.mcp.embeddings import load_encoder

        _encoder = load_encoder()
    except Exception:
        return None
    return _encoder


def _embedding_index(root: Path, index: _Index, signature: frozenset) -> object | None:
    encoder = _load_embeddings()
    if encoder is None or not index.chunks:
        return None
    key = root.as_posix()
    cached = _EMBED_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    from exosim.mcp.embeddings import EmbeddingIndex

    emb = EmbeddingIndex([chunk.text for chunk in index.chunks], encoder)
    _EMBED_CACHE[key] = (signature, emb)
    return emb


def search_knowledge_sources(
    query: str,
    *,
    repo_root: str | Path | None = None,
    max_results: int = 5,
    minimum_score: float = MINIMUM_RETRIEVAL_SCORE,
) -> list[KnowledgeSource]:
    """Rank curated ExoSim passages for a query.

    The default backend is BM25: a passage is kept only if it contains at least
    ``minimum_score`` of the distinct query terms (and at least one). With
    ``EXOSIM_MCP_RETRIEVAL=hybrid`` (and the ``mcp-embeddings`` extra) a
    sentence-embedding cosine score is added — a passage may then also qualify
    on semantic similarity alone. Scores are normalised so the top hit is
    ``1.0``.
    """
    query_terms = tokenize(query)
    if not query_terms:
        return []
    distinct_query = set(query_terms)

    root = resolve_repo_path(".", repo_root)
    signature = _signature(root)
    index = _get_index(root, signature)
    if not index.chunks:
        return []

    bm25_raw: dict[int, float] = {}
    for i, chunk in enumerate(index.chunks):
        covered = len(distinct_query & chunk.terms)
        if covered and covered / len(distinct_query) >= minimum_score:
            bm25_raw[i] = index.bm25(query_terms, chunk)
    bm25_max = max(bm25_raw.values(), default=0.0) or 1.0

    scored: dict[int, float] = {i: raw / bm25_max for i, raw in bm25_raw.items()}

    if active_backend() == "hybrid":
        emb = _embedding_index(root, index, signature)
        if emb is not None:
            from exosim.mcp.embeddings import MIN_COSINE

            cosine = emb.cosine_scores(query)
            for i, value in enumerate(cosine):
                value = float(value)
                if value >= MIN_COSINE or i in bm25_raw:
                    scored[i] = max(scored.get(i, 0.0), value)

    if not scored:
        return []
    order = sorted(scored, key=lambda i: scored[i], reverse=True)[:max_results]
    return [
        KnowledgeSource(
            path=index.chunks[i].path,
            start_line=index.chunks[i].start_line,
            end_line=index.chunks[i].end_line,
            score=round(scored[i], 4),
            text=index.chunks[i].text,
        )
        for i in order
    ]


def knowledge_base_manifest(repo_root: str | Path | None = None) -> dict[str, object]:
    """Describe the approved local knowledge base.

    Returns
    -------
    dict
        Source patterns, matched file count and grounding policy metadata.
    """
    root = resolve_repo_path(".", repo_root)
    matched_files = [
        path.relative_to(root).as_posix() for path in _iter_source_files(root)
    ]
    return {
        "source_patterns": [
            source_pattern.to_dict() for source_pattern in DEFAULT_SOURCE_PATTERNS
        ],
        "matched_file_count": len(matched_files),
        "matched_files": matched_files,
        "retrieval_backend": active_backend(),
        "minimum_retrieval_score": MINIMUM_RETRIEVAL_SCORE,
        "grounding_policy": (
            "Explanatory answers should cite at least one retrieved source. "
            "If no passage reaches the minimum score, report insufficient evidence."
        ),
    }
