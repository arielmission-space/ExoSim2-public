"""Relative read paths resolve against the run work dir first, then the content root."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py
import pytest

from exosim.mcp.sandbox import PathNotAllowedError, Sandbox

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A content root and a *separate* work dir (as with packaged content)."""
    content = tmp_path / "content"
    work = tmp_path / "work"
    (content / "examples").mkdir(parents=True)
    work.mkdir()
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(content))
    monkeypatch.setenv("EXOSIM_MCP_WORK_DIR", str(work))
    cwd = tmp_path / "elsewhere"  # not readable: only content + work are
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return content, work


def _write_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("channels").create_group("Photometer")
        group.create_dataset("focal_plane", data=[[1.0, 2.0], [3.0, 4.0]])


def test_work_dir_wins_over_content_root(layout: tuple[Path, Path]) -> None:
    content, work = layout
    (content / "data.txt").write_text("content")
    (work / "data.txt").write_text("work")
    box = Sandbox.from_env()
    assert box.resolve_read("data.txt", prefer_work_dir=True) == work / "data.txt"
    # default behaviour unchanged: content root
    assert box.resolve_read("data.txt") == content / "data.txt"


def test_falls_back_to_content_root(layout: tuple[Path, Path]) -> None:
    content, _ = layout
    (content / "examples" / "a.xml").write_text("<r/>")
    box = Sandbox.from_env()
    got = box.resolve_read("examples/a.xml", prefer_work_dir=True)
    assert got == content / "examples" / "a.xml"


def test_missing_everywhere_resolves_to_content_root(
    layout: tuple[Path, Path],
) -> None:
    content, _ = layout
    box = Sandbox.from_env()
    assert box.resolve_read("nope.h5", prefer_work_dir=True) == content / "nope.h5"


def test_traversal_from_work_dir_is_still_rejected(
    layout: tuple[Path, Path], tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    box = Sandbox.from_env()
    with pytest.raises(PathNotAllowedError):
        box.resolve_read("../secret.txt", prefer_work_dir=True)
    with pytest.raises(PathNotAllowedError):
        box.resolve_read(str(secret), prefer_work_dir=True)


def test_symlink_out_of_sandbox_rejected(
    layout: tuple[Path, Path], tmp_path: Path
) -> None:
    _, work = layout
    outside = tmp_path / "elsewhere" / "private.h5"
    outside.write_text("x")
    link = work / "link.h5"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlinks unavailable")
    box = Sandbox.from_env()
    # explicit path through the symlink: rejected
    with pytest.raises(PathNotAllowedError):
        box.resolve_read(str(link), prefer_work_dir=True)
    # bare name: the escaping work-dir candidate is never returned, and the error
    # says why instead of reporting a plain "not found"
    with pytest.raises(PathNotAllowedError, match="resolves outside") as info:
        box.resolve_read("link.h5", prefer_work_dir=True)
    assert str(outside) not in str(info.value)


def test_explain_and_inspect_output_by_bare_name(layout: tuple[Path, Path]) -> None:
    """The documented ``explain_output("focal.h5")`` works after a run."""
    from exosim.mcp.tools import explain_output, inspect_output

    _, work = layout
    _write_h5(work / "focal.h5")

    explained = explain_output("focal.h5")
    assert explained["pipeline"] == "focalplane"
    assert explained["path"].endswith("focal.h5")
    assert inspect_output("focal.h5")["format"] == "hdf5"


def test_inspect_output_still_reads_content_root(layout: tuple[Path, Path]) -> None:
    from exosim.mcp.tools import inspect_output

    content, _ = layout
    _write_h5(content / "examples" / "demo.h5")
    assert inspect_output("examples/demo.h5")["format"] == "hdf5"


def test_inspect_output_cwd_not_readable(layout: tuple[Path, Path]) -> None:
    from exosim.mcp.tools import inspect_output

    _write_h5(layout[0].parent / "elsewhere" / "x.h5")
    # the cwd is neither work dir nor content root nor allowed: never read
    with pytest.raises(FileNotFoundError):
        inspect_output("x.h5")


def test_plot_output_by_bare_name(layout: tuple[Path, Path]) -> None:
    from exosim.mcp.plotting import generate_plots

    _, work = layout
    _write_h5(work / "focal.h5")
    result = generate_plots("focal.h5", kind="nonsense")
    # file found (error is about the kind, not about a missing file)
    assert "not found" not in result.get("error", "")
    assert "Cannot plot" in result["error"]


def test_compare_configs_finds_work_dir_draft(layout: tuple[Path, Path]) -> None:
    from exosim.mcp.comparison import compare_configurations as compare_configs

    content, work = layout
    (content / "examples" / "a.xml").write_text("<root><x>1</x></root>")
    (work / "draft.xml").write_text("<root><x>2</x></root>")
    result = compare_configs("examples/a.xml", "draft.xml")
    assert result["config2"] == str(work / "draft.xml")


def test_run_pipeline_missing_input_file_fails_before_submit(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from exosim.mcp import jobs
    from exosim.mcp.tools import run_pipeline

    content, _ = layout
    (content / "examples" / "cfg.xml").write_text("<root/>")
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    submitted: list[object] = []
    monkeypatch.setattr(jobs.JOBS, "submit", lambda *a, **k: submitted.append(1) or "x")
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        run_pipeline("subexposures", "examples/cfg.xml", "o.h5", input_file="nope.h5")
    assert not submitted


def test_run_pipeline_input_file_resolved_from_work_dir(
    layout: tuple[Path, Path],
) -> None:
    from exosim.mcp.pipelines import _resolve_spec

    content, work = layout
    (content / "examples" / "cfg.xml").write_text("<root/>")
    _write_h5(work / "focal.h5")
    _, spec = _resolve_spec(
        "subexposures",
        "examples/cfg.xml",
        "sub.h5",
        input_file="focal.h5",
        repo_root=None,
        work_dir=None,
        n_threads=None,
        chunk_size=None,
        overwrite=False,
        plot=False,
    )
    assert spec["input"] == str(work / "focal.h5")


def test_run_pipeline_input_outside_sandbox_rejected(
    layout: tuple[Path, Path], tmp_path: Path
) -> None:
    from exosim.mcp.tools import run_pipeline

    content, _ = layout
    (content / "examples" / "cfg.xml").write_text("<root/>")
    with pytest.raises(PathNotAllowedError):
        run_pipeline(
            "subexposures", "examples/cfg.xml", "o.h5", input_file="../secret.h5"
        )


def test_run_pipeline_configuration_resolved_from_work_dir(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft written under drafts/ can be run by its relative path."""
    from exosim.mcp import jobs
    from exosim.mcp.pipelines import dispatch_pipeline

    _, work = layout
    (work / "drafts").mkdir()
    (work / "drafts" / "focalplane_draft.xml").write_text("<root/>")
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    seen: list[dict] = []

    def fake_submit(spec: dict, **kwargs: object) -> str:
        seen.append(spec)
        return "job1"

    monkeypatch.setattr(jobs.JOBS, "submit", fake_submit)
    monkeypatch.setattr(jobs.JOBS, "wait", lambda *a, **k: {"status": "running"})
    result = dispatch_pipeline(
        "focalplane", "drafts/focalplane_draft.xml", "o.h5", wait_seconds=0
    )
    assert result["status"] == "running"
    assert seen[0]["configuration"] == str(work / "drafts" / "focalplane_draft.xml")


def _submit_spy(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    from exosim.mcp import jobs

    seen: list[float | None] = []

    def fake_submit(spec: dict, *, log_dir: object, max_runtime: float | None) -> str:
        seen.append(max_runtime)
        return "job1"

    monkeypatch.setattr(jobs.JOBS, "submit", fake_submit)
    monkeypatch.setattr(jobs.JOBS, "wait", lambda *a, **k: {"status": "running"})
    return seen


@pytest.mark.parametrize(
    ("requested", "cap", "expected"),
    [(None, "100", 100.0), (50, "100", 50.0), (1e6, "100", 100.0)],
)
def test_run_pipeline_runtime_is_clamped_to_server_cap(
    layout: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    requested: float | None,
    cap: str,
    expected: float,
) -> None:
    from exosim.mcp.pipelines import dispatch_pipeline

    content, _ = layout
    (content / "examples" / "cfg.xml").write_text("<root/>")
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    monkeypatch.setenv("EXOSIM_MCP_MAX_RUNTIME", cap)
    seen = _submit_spy(monkeypatch)
    dispatch_pipeline(
        "focalplane",
        "examples/cfg.xml",
        "o.h5",
        wait_seconds=0,
        max_runtime_seconds=requested,
    )
    assert seen == [expected]


@pytest.mark.parametrize("bad", [0, -5])
def test_run_pipeline_rejects_non_positive_runtime(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, bad: float
) -> None:
    from exosim.mcp.pipelines import dispatch_pipeline

    content, _ = layout
    (content / "examples" / "cfg.xml").write_text("<root/>")
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    seen = _submit_spy(monkeypatch)
    with pytest.raises(ValueError, match="EXOSIM_MCP_MAX_RUNTIME"):
        dispatch_pipeline(
            "focalplane",
            "examples/cfg.xml",
            "o.h5",
            wait_seconds=0,
            max_runtime_seconds=bad,
        )
    assert not seen
