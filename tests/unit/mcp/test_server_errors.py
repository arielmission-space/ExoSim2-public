"""Server-level tests: the text of a failing tool call must reach the client."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from exosim.mcp import server

from .stdio_helper import REPO_ROOT, result_text, run_stdio_session

if TYPE_CHECKING:
    from pathlib import Path

try:
    server._load_server_class()
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

pytestmark = pytest.mark.skipif(
    not _HAS_MCP, reason="requires the 'mcp' optional dependency"
)


def test_sandbox_violation_text_reaches_the_client(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "secret.h5"
    _, (response,), _ = run_stdio_session(
        [("inspect_output", {"path": str(outside)})],
        work_dir=tmp_path / "work",
    )
    assert response["result"]["isError"] is True
    text = result_text(response).lower()
    assert "outside the directories this mcp server may read" in text
    assert str(outside).lower() in text


def test_overwrite_refusal_text_reaches_the_client(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "out.h5").write_bytes(b"")
    config = REPO_ROOT / "examples" / "main_example.xml"
    _, (response,), _ = run_stdio_session(
        [
            (
                "run_pipeline",
                {
                    "pipeline": "focalplane",
                    "configuration": str(config),
                    "output": "out.h5",
                },
            )
        ],
        work_dir=work,
    )
    assert response["result"]["isError"] is True
    text = result_text(response)
    assert "overwrite=True" in text


def test_stdout_carries_only_json_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ExoSim logs must go to stderr, leaving stdout to the JSON-RPC protocol.

    At DEBUG level loading a configuration logs a lot. Without
    ``_route_logging_to_stderr`` in ``main`` none of it reaches stderr (the check
    that fails on the old code); with it, stdout stays pure JSON-RPC.
    """
    monkeypatch.setenv("EXOSIM_LOG_LEVEL", "DEBUG")
    lines, (response,), stderr = run_stdio_session(
        [("validate_config", {"configuration": "examples/main_example.xml"})],
        work_dir=tmp_path / "work",
    )
    assert response["result"]["isError"] is False
    assert lines
    for line in lines:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0", line
    assert "Input option file found" in stderr


def test_user_facing_runtime_errors_reach_the_client(tmp_path: Path) -> None:
    """ContentMissingError (a RuntimeError) is a clean message, not a crash."""
    empty = tmp_path / "empty-root"
    empty.mkdir()
    _, (response,), _ = run_stdio_session(
        [("draft_config", {"pipeline": "focalplane"})],
        work_dir=tmp_path / "work",
        repo_root=empty,
    )
    assert response["result"]["isError"] is True
    text = result_text(response)
    assert text != "Error executing tool draft_config"
    assert "Reference configuration" in text


def test_expose_errors_converts_package_runtime_errors_only() -> None:
    from exosim.mcp.assets import ContentMissingError
    from exosim.mcp.embeddings import EmbeddingsUnavailableError

    tool_error = server._tool_error_class()

    def raiser(exc: Exception):
        def tool() -> None:
            raise exc

        return server._expose_errors(tool)

    for exc in (ContentMissingError("no kb"), EmbeddingsUnavailableError("no extra")):
        with pytest.raises(tool_error, match=str(exc)):
            raiser(exc)()

    # A bare RuntimeError is a likely bug: it must not be relabelled as expected.
    with pytest.raises(RuntimeError) as info:
        raiser(RuntimeError("boom"))()
    assert not isinstance(info.value, tool_error)
