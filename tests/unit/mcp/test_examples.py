"""Smoke test: the MCP example scripts run against the current tool API."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


@pytest.mark.parametrize("script", ["mcp_example.py", "mcp_demo_conversation.py"])
def test_mcp_example_script_runs(script: str, capsys: pytest.CaptureFixture) -> None:
    path = EXAMPLES / script
    saved_argv = sys.argv[:]
    sys.argv = [str(path)]
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = saved_argv
    out = capsys.readouterr().out
    assert "focalplane" in out
