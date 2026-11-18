"""Tests for the ExoSim MCP server entry point."""

from __future__ import annotations

import asyncio
import builtins
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from exosim.mcp import server

if TYPE_CHECKING:
    from pathlib import Path

try:
    server._load_server_class()
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

_REAL_IMPORT = builtins.__import__

requires_mcp = pytest.mark.skipif(
    not _HAS_MCP, reason="requires the 'mcp' optional dependency"
)


def _tool_names(srv: object) -> set[str]:
    return {tool.name for tool in asyncio.run(srv.list_tools())}


def test_parse_args_defaults() -> None:
    args = server._parse_args([])
    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.no_exec is False


def test_parse_args_rejects_unknown_transport() -> None:
    with pytest.raises(SystemExit):
        server._parse_args(["--transport", "websocket"])


@requires_mcp
def test_build_server_records_bind_address() -> None:
    srv = server.build_server(host="0.0.0.0", port=9001)
    assert srv._exosim_bind == ("0.0.0.0", 9001)


@requires_mcp
def test_build_server_registers_read_only_tools_without_exec() -> None:
    srv = server.build_server(allow_exec=False)
    names = _tool_names(srv)
    assert "search_knowledge_base" in names
    assert {"run_pipeline", "get_job_status", "cancel_job", "list_jobs"}.isdisjoint(
        names
    )


@pytest.mark.parametrize("transport", ["stdio", "sse", "http"])
def test_main_runs_the_selected_transport(transport: str) -> None:
    captured: dict[str, object] = {}

    class FakeServer:
        settings = type("S", (), {"host": "", "port": 0})()

        def run(self, transport: str, **kwargs: object) -> None:
            captured["transport"] = transport
            captured["kwargs"] = kwargs

    with (
        patch.object(server, "build_server", return_value=FakeServer()),
        patch.object(server.Sandbox, "ensure_work_dir", lambda self: self.work_dir),
    ):
        server.main(["--transport", transport])

    assert captured["transport"] == server._TRANSPORTS[transport]
    # The bind address is a run() keyword on the network transports only.
    expected = {} if transport == "stdio" else {"host": "127.0.0.1", "port": 8000}
    assert captured["kwargs"] == expected


@requires_mcp
def test_every_registered_tool_has_a_description() -> None:
    srv = server.build_server()
    tools = asyncio.run(srv.list_tools())
    assert tools
    missing = [t.name for t in tools if not (t.description or "").strip()]
    assert missing == []


def test_route_logging_to_stderr(capfd: pytest.CaptureFixture[str]) -> None:
    from exosim.log import structlog_config

    server._route_logging_to_stderr()
    try:
        structlog_config.ExoSimLogger("exosim").info("hello-from-exosim")
        out, err = capfd.readouterr()
        assert "hello-from-exosim" not in out
        assert "hello-from-exosim" in err
    finally:
        structlog_config.set_log_stream(None)
        structlog_config.configure_structlog()


def test_main_installs_signal_hooks_on_main_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tools run in worker threads, so main() must register the handlers."""
    from exosim.mcp import jobs, server

    calls: list[str] = []
    monkeypatch.setattr(jobs.JOBS, "install_hooks", lambda: calls.append("hooks"))
    monkeypatch.setattr(server, "build_server", lambda **_: _FakeServer(calls))
    monkeypatch.setenv("EXOSIM_MCP_WORK_DIR", str(tmp_path / "w"))
    server.main(["--work-dir", str(tmp_path / "w")])
    assert calls == ["hooks", "run"]


class _FakeServer:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def run(self, **_: object) -> None:
        self._calls.append("run")


def _fail_mcp_import(name, *args, **kwargs):
    if name == "mcp.server":
        raise ImportError("boom from the SDK")
    return _REAL_IMPORT(name, *args, **kwargs)


@pytest.mark.parametrize(
    ("installed", "blames_version"),
    [("1.9.0", True), ("2.0.1", False), (None, False)],
)
def test_import_failure_message(installed, blames_version) -> None:
    """S6: only claim a version mismatch when it really is outside mcp>=2,<3."""
    from importlib.metadata import PackageNotFoundError

    def fake_version(_name: str) -> str:
        if installed is None:
            raise PackageNotFoundError
        return installed

    with (
        patch("builtins.__import__", side_effect=_fail_mcp_import),
        patch.object(server, "_pkg_version", fake_version),
        pytest.raises(ImportError) as caught,
    ):
        server._load_server_class()
    message = str(caught.value)
    assert "boom from the SDK" in message  # original text always kept
    assert ("is required" in message) is blames_version
    if installed is None:
        assert "not installed" in message


def test_unusable_default_work_dir_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """S3: an uncreatable default work dir falls back to ~/.exosim/mcp-runs."""
    from exosim.mcp.sandbox import ENV_WORK_DIR, Sandbox

    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so nothing can be created below it")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(ENV_WORK_DIR, raising=False)
    monkeypatch.setenv(
        "EXOSIM_MCP_REPO_ROOT", str(blocker)
    )  # default = blocker/mcp-runs
    sandbox = Sandbox.from_env()

    moved = server._prepare_work_dir(sandbox)

    assert moved.work_dir == (home / ".exosim" / "mcp-runs").resolve()
    assert moved.work_dir.is_dir()
    err = capsys.readouterr().err
    assert err.count("warning") == 1
    assert "Traceback" not in err


def test_unusable_explicit_work_dir_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from exosim.mcp.sandbox import ENV_WORK_DIR, Sandbox

    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    monkeypatch.setenv(ENV_WORK_DIR, str(blocker / "runs"))
    with pytest.raises(SystemExit) as caught:
        server._prepare_work_dir(Sandbox.from_env())
    assert "cannot use the work directory" in str(caught.value.code)
    assert "--work-dir" in str(caught.value.code)


def test_no_exec_does_not_create_the_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exosim.mcp.sandbox import ENV_WORK_DIR, Sandbox

    work = tmp_path / "lazy" / "runs"
    monkeypatch.setenv(ENV_WORK_DIR, str(work))
    sandbox = Sandbox.from_env(allow_exec=False)
    assert server._prepare_work_dir(sandbox).work_dir == work.resolve()
    assert not work.exists()


def test_no_exec_with_unusable_default_falls_back_without_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exosim.mcp.sandbox import ENV_WORK_DIR, Sandbox

    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(ENV_WORK_DIR, raising=False)
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(blocker))
    moved = server._prepare_work_dir(Sandbox.from_env(allow_exec=False))
    assert moved.work_dir == (home / ".exosim" / "mcp-runs").resolve()
    assert not moved.work_dir.exists()  # created lazily by the tools that need it
