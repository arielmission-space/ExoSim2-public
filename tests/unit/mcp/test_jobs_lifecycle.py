"""Lifecycle tests for the MCP job manager: watchdog, cap, shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from exosim.mcp.jobs import (
    DEFAULT_MAX_JOBS,
    JobLimitError,
    JobManager,
    max_jobs_from_env,
)

if TYPE_CHECKING:
    from pathlib import Path


def _spec(**recipe_kwargs: object) -> dict[str, object]:
    return {
        "pipeline": "selftest",
        "recipe": "exosim.mcp.jobs._SelfTestRecipe",
        "recipe_kwargs": recipe_kwargs,
        "output": "out.h5",
    }


def _wait_until_terminal(manager: JobManager, job_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = manager.status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never finished")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); treat it as dead
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    return not _pid_alive(pid)


def test_deadline_enforced_without_polling(tmp_path: Path) -> None:
    manager = JobManager()
    job_id = manager.submit(_spec(sleep=60), log_dir=tmp_path / "logs", max_runtime=1.0)
    job = manager._jobs[job_id]
    pid = job._process.pid

    # Never call status()/list_jobs(): the watchdog alone must end the job.
    deadline = time.time() + 20
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.2)

    assert job.status == "timeout"
    assert _wait_dead(pid)
    status = manager.status(job_id)
    assert status["status"] == "timeout"
    assert "max_runtime" in status["reason"]
    assert manager.list_jobs()[0]["reason"] == status["reason"]


def test_concurrency_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXOSIM_MCP_MAX_JOBS", "1")
    manager = JobManager()
    first = manager.submit(_spec(sleep=60), log_dir=tmp_path / "logs", max_runtime=None)
    try:
        other = _spec()
        other["output"] = "other.h5"  # a different output: only the cap applies
        with pytest.raises(JobLimitError) as excinfo:
            manager.submit(other, log_dir=tmp_path / "logs", max_runtime=None)
        message = str(excinfo.value)
        assert "EXOSIM_MCP_MAX_JOBS" in message
        assert first in message
        manager.cancel(first)
        second = manager.submit(_spec(), log_dir=tmp_path / "logs", max_runtime=None)
        _wait_until_terminal(manager, second)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("value", ["", "0", "-3", "abc"])
def test_max_jobs_invalid_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EXOSIM_MCP_MAX_JOBS", value)
    assert max_jobs_from_env() == DEFAULT_MAX_JOBS
    monkeypatch.setenv("EXOSIM_MCP_MAX_JOBS", "5")
    assert max_jobs_from_env() == 5


def test_finished_jobs_stay_readable_and_are_reaped(tmp_path: Path) -> None:
    manager = JobManager()
    job_id = manager.submit(
        _spec(fail=True), log_dir=tmp_path / "logs", max_runtime=None
    )
    _wait_until_terminal(manager, job_id)
    job = manager._jobs[job_id]
    assert job._process is None  # handle released: no zombie
    again = manager.status(job_id)
    assert again["status"] == "failed"
    assert "self-test failure" in again["log_tail"]
    assert job.returncode == 1


def test_dispatch_reports_cap_as_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exosim.mcp import jobs
    from exosim.mcp.pipelines import dispatch_pipeline

    manager = JobManager()
    monkeypatch.setattr(jobs, "JOBS", manager)
    monkeypatch.setenv("EXOSIM_MCP_MAX_JOBS", "1")
    monkeypatch.setenv("EXOSIM_MCP_EXEC_BACKEND", "process")
    busy = manager.submit(_spec(sleep=60), log_dir=tmp_path / "logs", max_runtime=None)
    try:
        result = dispatch_pipeline(
            "radiometric",
            "examples/radiometric/main_radiometric.xml",
            "o.h5",
            wait_seconds=0,
        )
    finally:
        manager.shutdown()
    assert result["status"] == "refused"
    assert "EXOSIM_MCP_MAX_JOBS" in result["error"]
    assert busy in result["running_jobs"]


_HELPER = """
import sys, threading, time
from pathlib import Path
from exosim.mcp.jobs import JobManager

if __name__ == "__main__":
    work = Path(sys.argv[1]); mode = sys.argv[2]
    manager = JobManager()
    spec = {"pipeline": "selftest", "recipe": "exosim.mcp.jobs._SelfTestRecipe",
            "recipe_kwargs": {"sleep": 120, "marker": str(work / "marker")},
            "output": "o.h5"}
    # mcp 2.x runs sync tools in worker threads: the first submit is off-thread,
    # so only the main-thread install (server.main()) can register the handlers.
    worker = threading.Thread(
        target=lambda: manager.submit(spec, log_dir=work / "logs", max_runtime=None)
    )
    worker.start(); worker.join()
    if mode == "sigterm":
        manager.install_hooks()
    job = next(iter(manager._jobs.values()))
    (work / "pid").write_text(str(job._process.pid))
    while not (work / "marker").exists():
        time.sleep(0.1)
    (work / "ready").write_text("1")
    if mode in ("sigterm", "unhooked"):
        time.sleep(60)
"""


def _run_helper(tmp_path: Path, mode: str) -> tuple[subprocess.Popen, int]:
    script = tmp_path / "helper.py"
    script.write_text(_HELPER, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), mode], env=os.environ.copy()
    )
    deadline = time.time() + 60
    while not (tmp_path / "ready").exists() and time.time() < deadline:
        time.sleep(0.2)
    assert (tmp_path / "ready").exists(), "helper never started its job"
    return proc, int((tmp_path / "pid").read_text())


def test_children_terminated_on_normal_exit(tmp_path: Path) -> None:
    proc, child = _run_helper(tmp_path, "exit")
    started = time.time()
    proc.wait(timeout=30)
    # interpreter exit must not block until the 120 s job ends
    assert time.time() - started < 20
    assert _wait_dead(child)


def test_children_terminated_on_sigterm(tmp_path: Path) -> None:
    proc, child = _run_helper(tmp_path, "sigterm")
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    assert _wait_dead(child)
