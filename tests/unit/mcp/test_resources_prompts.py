"""Tests for MCP resources and prompts."""

from __future__ import annotations

import asyncio

import pytest

from exosim.mcp import prompts, resources


def test_knowledge_base_index_lists_uris() -> None:
    index = resources.knowledge_base_index()
    assert index["count"] > 0
    uris = {entry["uri"] for entry in index["resources"]}
    assert "exosim://kb/AGENTS.md" in uris


def test_read_knowledge_file_returns_text() -> None:
    text = resources.read_knowledge_file("AGENTS.md")
    assert isinstance(text, str)
    assert text.strip()


def test_read_knowledge_file_rejects_paths_outside_the_manifest() -> None:
    with pytest.raises(FileNotFoundError):
        resources.read_knowledge_file("pyproject.toml")
    with pytest.raises(FileNotFoundError):
        resources.read_knowledge_file("../../../etc/passwd")


@pytest.mark.parametrize("prompt", prompts.ALL_PROMPTS, ids=lambda p: p.__name__)
def test_prompts_mention_the_tools_and_grounding(prompt) -> None:
    text = prompt()
    assert "search_knowledge_base" in text or "inspect_output" in text
    assert "insufficient_evidence" in text


def _server():
    pytest.importorskip("mcp.server", reason="requires the 'mcp' optional dependency")
    from exosim.mcp.server import build_server

    return build_server()


def test_server_registers_resources_and_prompts() -> None:
    srv = _server()
    templates = asyncio.run(srv.list_resource_templates())
    prompt_names = {p.name for p in asyncio.run(srv.list_prompts())}

    assert any(t.uri_template == "exosim://kb/{+path}" for t in templates)
    assert {
        "set_up_channel",
        "diagnose_failed_run",
        "explain_output",
        "new_simulation",
    } <= prompt_names


def test_server_serves_knowledge_base_resources() -> None:
    srv = _server()
    uris = {str(r.uri) for r in asyncio.run(srv.list_resources())}
    assert "exosim://kb" in uris

    index = list(asyncio.run(srv.read_resource("exosim://kb")))
    assert "exosim://kb/AGENTS.md" in str(index[0].content)

    contents = list(asyncio.run(srv.read_resource("exosim://kb/AGENTS.md")))
    assert contents
    assert str(contents[0].content).strip()


@pytest.mark.parametrize("prompt", prompts.ALL_PROMPTS, ids=lambda p: p.__name__)
def test_server_prompts_render_through_the_sdk(prompt) -> None:
    srv = _server()
    result = asyncio.run(srv.get_prompt(prompt.__name__, {}))
    text = "".join(m.content.text for m in result.messages)
    assert "insufficient_evidence" in text


def test_server_serves_nested_knowledge_base_files() -> None:
    """``{path}`` matches ONE URI segment on mcp 2.x: nested files need ``{+path}``."""
    srv = _server()
    nested = "exosim://kb/docs/source/user/mcp.rst"
    contents = list(asyncio.run(srv.read_resource(nested)))
    assert contents
    assert str(contents[0].content).strip()


def test_server_rejects_traversal_through_the_resource_template() -> None:
    srv = _server()
    for uri in ("exosim://kb/../pyproject.toml", "exosim://kb//etc/passwd"):
        with pytest.raises(Exception):  # noqa: B017, PT011
            list(asyncio.run(srv.read_resource(uri)))
