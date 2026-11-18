"""Regression tests for the second independent review of the MCP server."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from exosim.mcp.jobs import JobManager
from exosim.mcp.pipelines import MAX_WAIT_SECONDS, dispatch_pipeline
from exosim.mcp.sandbox import DEFAULT_MAX_RUNTIME, Sandbox


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    content = tmp_path / "content"
    work = tmp_path / "work"
    (content / "examples").mkdir(parents=True)
    (content / "examples" / "cfg.xml").write_text("<root/>")
    work.mkdir()
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(content))
    monkeypatch.setenv("EXOSIM_MCP_WORK_DIR", str(work))
    return content, work


def test_same_output_as_running_job_is_refused(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: a second run on a running job's output is refused, even with overwrite."""
    from exosim.mcp import jobs

    _, work = layout
    manager = JobManager()
    monkeypatch.setattr(jobs, "JOBS", manager)
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    busy = manager.submit(
        {
            "pipeline": "selftest",
            "recipe": "exosim.mcp.jobs._SelfTestRecipe",
            "recipe_kwargs": {"sleep": 60},
            "output": str((work / "o.h5").resolve()),
        },
        log_dir=work / "logs",
        max_runtime=None,
    )
    try:
        for overwrite in (False, True):
            result = dispatch_pipeline(
                "focalplane",
                "examples/cfg.xml",
                "o.h5",
                wait_seconds=0,
                overwrite=overwrite,
            )
            assert result["status"] == "refused"
            assert result["conflicting_job"] == busy
            assert "already being written" in result["error"]
    finally:
        manager.shutdown()


def test_output_in_new_subdirectory_is_created(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5: the parent directory of the output exists before the run starts."""
    from exosim.mcp import jobs

    _, work = layout
    seen: list[bool] = []

    def fake_submit(spec: dict, *, log_dir: object, max_runtime: float | None) -> str:
        seen.append(Path(spec["output"]).parent.is_dir())
        return "job1"

    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    monkeypatch.setattr(jobs.JOBS, "submit", fake_submit)
    monkeypatch.setattr(jobs.JOBS, "wait", lambda *a, **k: {"status": "running"})
    dispatch_pipeline("focalplane", "examples/cfg.xml", "newsub/deeper/x.h5")
    assert seen == [True]
    assert (work / "newsub" / "deeper").is_dir()


def test_wait_seconds_is_clamped(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """N7: a call never blocks beyond the documented maximum."""
    from exosim.mcp import jobs

    waits: list[float] = []
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    monkeypatch.setattr(jobs.JOBS, "submit", lambda *a, **k: "job1")
    monkeypatch.setattr(
        jobs.JOBS,
        "wait",
        lambda job_id, timeout: waits.append(timeout) or {"status": "running"},
    )
    for requested in (1e6, -5, 7):
        dispatch_pipeline(
            "focalplane", "examples/cfg.xml", "o.h5", wait_seconds=requested
        )
    assert waits == [MAX_WAIT_SECONDS, 0.0, 7.0]
    assert MAX_WAIT_SECONDS <= 50


@pytest.mark.parametrize("value", ["0", "-5", "abc", "nan", "inf"])
def test_non_positive_max_runtime_falls_back_with_warning(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """S7: 0 / negative / invalid EXOSIM_MCP_MAX_RUNTIME means the default."""
    monkeypatch.setenv("EXOSIM_MCP_MAX_RUNTIME", value)
    with pytest.warns(RuntimeWarning, match="EXOSIM_MCP_MAX_RUNTIME"):
        box = Sandbox.from_env()
    assert box.max_runtime == DEFAULT_MAX_RUNTIME


def test_valid_and_unset_max_runtime_do_not_warn(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monkeypatch.delenv("EXOSIM_MCP_MAX_RUNTIME", raising=False)
        assert Sandbox.from_env().max_runtime == DEFAULT_MAX_RUNTIME
        monkeypatch.setenv("EXOSIM_MCP_MAX_RUNTIME", "120")
        assert Sandbox.from_env().max_runtime == 120.0


def test_log_colours_can_be_disabled_before_import():
    """EXOSIM_LOG_COLORS=false is honoured from the first import, so the
    RUN CONFIGURATION banner logged at import time in a job has no ANSI codes."""
    import os
    import subprocess
    import sys

    code = "from exosim.log import structlog_config as s; print(s._log_colors)"
    env = dict(os.environ, EXOSIM_LOG_COLORS="false")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False"
