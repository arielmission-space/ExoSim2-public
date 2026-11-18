"""Retrieval eval suite: guardrail against ranking regressions.

Runs every case in :mod:`retrieval_cases` through the real retrieval stack
against the repository knowledge base and checks recall. This is deliberately an
aggregate check (recall@k over the whole set) so that a single reworded doc does
not break CI, while a real regression in the ranker or a lost doc page does.
"""

from __future__ import annotations

import pytest

from exosim.mcp.retrieval import search_knowledge_sources

from .retrieval_cases import CASES

TOP_K = 5
MIN_RECALL = 0.80


def _hits_for(question: str) -> list[str]:
    results = search_knowledge_sources(question, max_results=TOP_K)
    return [hit.path for hit in results]


@pytest.mark.slow
def test_retrieval_recall_at_k() -> None:
    misses: list[tuple[str, str, list[str]]] = []
    for case in CASES:
        paths = _hits_for(case.question)
        if not any(case.expected in path for path in paths):
            misses.append((case.question, case.expected, paths))

    recall = 1 - len(misses) / len(CASES)
    report = "\n".join(
        f"  - {q!r}\n    want *{exp}*, got:\n" + "\n".join(f"      {p}" for p in got)
        for q, exp, got in misses
    )
    assert recall >= MIN_RECALL, (
        f"retrieval recall@{TOP_K} = {recall:.0%} "
        f"(< {MIN_RECALL:.0%}); {len(misses)} misses:\n{report}"
    )


@pytest.mark.slow
def test_every_case_expected_path_exists() -> None:
    """A typo in an expected fragment would make a case unfalsifiable."""
    from exosim.mcp.retrieval import knowledge_base_manifest

    known = set(knowledge_base_manifest()["matched_files"])
    for case in CASES:
        assert any(case.expected in path for path in known), (
            f"no knowledge-base file matches expected fragment {case.expected!r} "
            f"for question {case.question!r}"
        )
