"""Tests for the ExoSim MCP filesystem sandbox."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from exosim.mcp.sandbox import (
    ENV_ALLOW_DIRS,
    ENV_NO_EXEC,
    ENV_WORK_DIR,
    PathNotAllowedError,
    Sandbox,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests drive the sandbox explicitly; drop the conftest env defaults."""
    for name in (
        "EXOSIM_MCP_REPO_ROOT",
        ENV_WORK_DIR,
        ENV_ALLOW_DIRS,
        ENV_NO_EXEC,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.xml").write_text("<root />", encoding="utf-8")
    return tmp_path


def test_defaults_place_work_dir_inside_repo(repo: Path) -> None:
    box = Sandbox.from_env(repo_root=repo)
    assert box.repo_root == repo.resolve()
    assert box.work_dir == (repo / "mcp-runs").resolve()
    assert box.allow_exec is True


def test_read_allows_repo_and_work_dir(repo: Path) -> None:
    box = Sandbox.from_env(repo_root=repo)
    assert box.resolve_read("src/config.xml") == (repo / "src/config.xml").resolve()
    assert box.resolve_read(str(box.work_dir / "out.h5")) == box.work_dir / "out.h5"


def test_read_rejects_paths_outside_the_sandbox(repo: Path, tmp_path: Path) -> None:
    box = Sandbox.from_env(repo_root=repo)
    outside = tmp_path.parent / "somewhere-else"
    with pytest.raises(PathNotAllowedError):
        box.resolve_read(str(outside / "secret"))
    with pytest.raises(PathNotAllowedError):
        box.resolve_read("~/.ssh/id_rsa")


def test_read_rejects_dotdot_traversal(repo: Path) -> None:
    box = Sandbox.from_env(repo_root=repo)
    with pytest.raises(PathNotAllowedError):
        box.resolve_read("src/../../escape.txt")


def test_write_confined_to_work_dir(repo: Path) -> None:
    box = Sandbox.from_env(repo_root=repo)
    assert box.resolve_write("run/out.h5") == (box.work_dir / "run/out.h5").resolve()
    # writing into the checkout itself is refused
    with pytest.raises(PathNotAllowedError):
        box.resolve_write(str(repo / "src" / "out.h5"))


def test_extra_allow_dirs_extend_both_read_and_write(
    repo: Path, tmp_path: Path
) -> None:
    extra = tmp_path.parent / "project-configs"
    with patch.dict(os.environ, {ENV_ALLOW_DIRS: str(extra)}):
        box = Sandbox.from_env(repo_root=repo)
    assert box.resolve_read(str(extra / "my.xml")) == (extra / "my.xml").resolve()
    assert box.resolve_write(str(extra / "out.h5")) == (extra / "out.h5").resolve()


def test_explicit_work_dir_and_env(repo: Path, tmp_path: Path) -> None:
    with patch.dict(os.environ, {ENV_WORK_DIR: str(tmp_path / "env-work")}):
        box = Sandbox.from_env(repo_root=repo)
    assert box.work_dir == (tmp_path / "env-work").resolve()


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_no_exec_env_disables_execution(repo: Path, value: str) -> None:
    with patch.dict(os.environ, {ENV_NO_EXEC: value}):
        box = Sandbox.from_env(repo_root=repo)
    assert box.allow_exec is False


def test_run_pipeline_refused_when_execution_disabled(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exosim.mcp.tools import run_pipeline

    (repo / "src" / "config.xml").write_text("<root />", encoding="utf-8")
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(repo))
    monkeypatch.setenv(ENV_NO_EXEC, "1")
    with pytest.raises(PermissionError):
        run_pipeline("focalplane", "src/config.xml", "out.h5")


def test_run_pipeline_rejects_output_outside_work_dir(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exosim.mcp.tools import run_pipeline

    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(repo))
    fake_recipes = SimpleNamespace(CreateFocalPlane=lambda **_: None)
    with (
        patch.dict("sys.modules", {"exosim.recipes": fake_recipes}),
        pytest.raises(PathNotAllowedError),
    ):
        run_pipeline(
            "focalplane",
            "src/config.xml",
            str(repo / "src" / "overwrite_me.py"),
        )
