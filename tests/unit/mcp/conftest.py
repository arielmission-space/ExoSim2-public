"""Shared fixtures for the ExoSim MCP tests.

The MCP tool functions read their sandbox from the environment (the server sets
it once at startup); tests point it at the real checkout with a throwaway
working directory, and individual tests override ``EXOSIM_MCP_REPO_ROOT`` when
they need a synthetic repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _mcp_sandbox_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("EXOSIM_MCP_WORK_DIR", str(tmp_path / "mcp-work"))
    # Pipeline runs execute in-process by default in tests (no subprocess spawn);
    # test_jobs.py opts back into the real process backend where it matters.
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "inline")
    monkeypatch.delenv("EXOSIM_MCP_ALLOW_DIRS", raising=False)
    monkeypatch.delenv("EXOSIM_MCP_NO_EXEC", raising=False)
    monkeypatch.delenv("EXOSIM_MCP_MAX_THREADS", raising=False)
    monkeypatch.delenv("EXOSIM_MCP_MAX_RUNTIME", raising=False)
    monkeypatch.delenv("EXOSIM_MCP_RETRIEVAL", raising=False)
