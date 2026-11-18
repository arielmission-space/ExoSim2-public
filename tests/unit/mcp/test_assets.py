"""Tests for the MCP content root: precedence, packaged assets, empty warnings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from exosim.mcp import assets
from exosim.mcp.assets import (
    ContentMissingError,
    collect_assets,
    content_report,
    is_checkout,
    resolve_content_root,
)
from exosim.mcp.authoring import draft_configuration
from exosim.mcp.sandbox import Sandbox

from .conftest import REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path


def _fake_checkout(path: Path) -> Path:
    (path / "src" / "exosim").mkdir(parents=True)
    (path / "docs").mkdir()
    (path / "pyproject.toml").write_text("", encoding="utf-8")
    return path


@pytest.fixture
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "pkg" / "_data"
    data.mkdir(parents=True)
    monkeypatch.setattr(assets, "packaged_root", lambda: data)
    monkeypatch.setattr(assets, "_package_checkout", lambda: None)
    return data


class TestPrecedence:
    def test_explicit_beats_env_and_checkout(self, tmp_path: Path) -> None:
        cwd = _fake_checkout(tmp_path / "cwd")
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        got = resolve_content_root(
            explicit, env={"EXOSIM_MCP_REPO_ROOT": str(cwd)}, cwd=cwd
        )
        assert (got.path, got.source) == (explicit.resolve(), "override")

    def test_env_override(self, tmp_path: Path) -> None:
        got = resolve_content_root(env={"EXOSIM_MCP_REPO_ROOT": str(tmp_path)})
        assert (got.path, got.source) == (tmp_path.resolve(), "override")

    def test_cwd_checkout(self, tmp_path: Path, packaged: Path) -> None:
        cwd = _fake_checkout(tmp_path / "co")
        got = resolve_content_root(env={}, cwd=cwd)
        assert (got.path, got.source) == (cwd.resolve(), "checkout")

    def test_package_checkout_when_cwd_is_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged: Path
    ) -> None:
        co = _fake_checkout(tmp_path / "editable")
        monkeypatch.setattr(assets, "_package_checkout", lambda: co)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        got = resolve_content_root(env={}, cwd=elsewhere)
        assert (got.path, got.source) == (co, "checkout")

    def test_packaged_fallback(self, tmp_path: Path, packaged: Path) -> None:
        got = resolve_content_root(env={}, cwd=tmp_path)
        assert (got.path, got.source) == (packaged, "packaged")

    def test_none_when_nothing_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(assets, "packaged_root", lambda: None)
        monkeypatch.setattr(assets, "_package_checkout", lambda: None)
        got = resolve_content_root(env={}, cwd=tmp_path)
        assert (got.path, got.source) == (tmp_path.resolve(), "none")

    def test_repo_is_a_checkout(self) -> None:
        assert is_checkout(REPO_ROOT)


class TestSandboxIntegration:
    def test_packaged_work_dir_is_not_in_site_packages(
        self,
        tmp_path: Path,
        packaged: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EXOSIM_MCP_REPO_ROOT")
        monkeypatch.delenv("EXOSIM_MCP_WORK_DIR")
        monkeypatch.chdir(tmp_path)
        sandbox = Sandbox.from_env()
        assert sandbox.content_source == "packaged"
        assert sandbox.repo_root == packaged
        assert sandbox.work_dir == (tmp_path / "mcp-runs").resolve()

    def test_to_dict_reports_content(self) -> None:
        content = Sandbox.from_env().to_dict()["content"]
        assert content["source"] == "override"
        assert content["knowledge_base_files"] > 0
        assert content["example_configs"] > 0
        assert content["warnings"] == []


class TestEmptyContent:
    def test_report_warns_when_empty(self, tmp_path: Path) -> None:
        report = content_report(tmp_path, "packaged")
        assert report["knowledge_base_files"] == 0
        assert report["example_configs"] == 0
        assert len(report["warnings"]) == 2
        assert "--repo-root" in report["warnings"][0]

    def test_startup_logs_warning_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        assets._WARNED.discard(f"override:{tmp_path.resolve()}")
        with caplog.at_level(logging.WARNING, logger="exosim.mcp"):
            Sandbox.from_env(repo_root=tmp_path)
            Sandbox.from_env(repo_root=tmp_path)
        messages = [r.getMessage() for r in caplog.records]
        assert sum("Knowledge base is empty" in m for m in messages) == 1

    def test_draft_config_raises_actionable_error(self, tmp_path: Path) -> None:
        with pytest.raises(ContentMissingError, match="--repo-root"):
            draft_configuration("focalplane", repo_root=tmp_path)


class TestCollectAssets:
    def test_collects_expected_and_skips_large_and_internal(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "dest"
        copied = collect_assets(REPO_ROOT, dest)
        assert "AGENTS.md" in copied
        assert "examples/main_example.xml" in copied
        assert "examples/data/payload/QE.ecsv" in copied
        assert "docs/ai-agents/mcp-release-plan.md" not in copied
        assert "examples/customsed.csv" not in copied
        assert not any(name.endswith(".ipynb") for name in copied)
        total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        assert total < 6_000_000

    def test_collected_tree_is_a_working_content_root(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"
        collect_assets(REPO_ROOT, dest)
        report = content_report(dest, "packaged")
        assert report["warnings"] == []
        from exosim.mcp.retrieval import search_knowledge_sources

        hits = search_knowledge_sources(
            "how do I configure the focal plane", repo_root=dest
        )
        assert hits
        result = draft_configuration("focalplane", repo_root=dest)
        assert result["valid"], result
