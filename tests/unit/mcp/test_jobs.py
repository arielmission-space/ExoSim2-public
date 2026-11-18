"""Tests for the ExoSim MCP background job manager."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from exosim.mcp.jobs import JobManager

if TYPE_CHECKING:
    from pathlib import Path


def _spec(**recipe_kwargs: object) -> dict[str, object]:
    return {
        "pipeline": "selftest",
        "recipe": "exosim.mcp.jobs._SelfTestRecipe",
        "recipe_kwargs": recipe_kwargs,
        "output": "out.h5",
    }


def _wait_until_terminal(
    manager: JobManager, job_id: str, timeout: float = 30.0
) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = manager.status(job_id)
        if status["status"] != "running":
            return status
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never finished: {manager.status(job_id)}")


def test_job_succeeds_and_reports_runtime(tmp_path: Path) -> None:
    manager = JobManager()
    job_id = manager.submit(_spec(), log_dir=tmp_path / "logs", max_runtime=None)

    status = _wait_until_terminal(manager, job_id)

    assert status["status"] == "succeeded"
    assert status["job_id"] == job_id
    assert status["runtime_seconds"] >= 0
    assert "log_tail" not in status  # only attached on failure


def test_job_failure_captures_log_tail(tmp_path: Path) -> None:
    manager = JobManager()
    job_id = manager.submit(
        _spec(fail=True), log_dir=tmp_path / "logs", max_runtime=None
    )

    status = _wait_until_terminal(manager, job_id)

    assert status["status"] == "failed"
    assert "self-test failure" in status["log_tail"]


def test_job_times_out(tmp_path: Path) -> None:
    manager = JobManager()
    job_id = manager.submit(_spec(sleep=30), log_dir=tmp_path / "logs", max_runtime=1.0)

    status = _wait_until_terminal(manager, job_id, timeout=20.0)

    assert status["status"] == "timeout"
    assert status["finished_at"] is not None


def test_job_can_be_cancelled(tmp_path: Path) -> None:
    manager = JobManager()
    marker = tmp_path / "marker"
    job_id = manager.submit(
        _spec(sleep=30, marker=str(marker)),
        log_dir=tmp_path / "logs",
        max_runtime=None,
    )
    # wait for the child to actually start
    deadline = time.time() + 15
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)

    status = manager.cancel(job_id)

    assert status["status"] == "cancelled"
    assert manager.status(job_id)["status"] == "cancelled"


def test_list_jobs_orders_by_submission(tmp_path: Path) -> None:
    manager = JobManager()
    first = manager.submit(_spec(), log_dir=tmp_path / "logs", max_runtime=None)
    other = _spec()
    other["output"] = "other.h5"  # same output as a running job would be refused
    second = manager.submit(other, log_dir=tmp_path / "logs", max_runtime=None)
    _wait_until_terminal(manager, first)
    _wait_until_terminal(manager, second)

    jobs = manager.list_jobs()

    assert [job["job_id"] for job in jobs] == [first, second]


def test_unknown_job_id_raises() -> None:
    manager = JobManager()
    with pytest.raises(KeyError):
        manager.status("does-not-exist")


def test_unknown_job_error_message_is_not_double_quoted() -> None:
    from exosim.mcp.tools import cancel_job, get_job_status

    for tool in (get_job_status, cancel_job):
        result = tool("nope")
        assert result["status"] == "unknown"
        assert result["error"] == "Unknown job id: nope"


def test_job_log_captures_all_output_without_colours(tmp_path: Path) -> None:
    """fd 1/2 and subprocess output land in the log, with no ANSI codes."""
    manager = JobManager()
    marker = tmp_path / "marker"
    job_id = manager.submit(
        _spec(chatty=True, marker=str(marker)),
        log_dir=tmp_path / "logs",
        max_runtime=None,
    )
    status = _wait_until_terminal(manager, job_id)
    assert status["status"] == "succeeded", status

    text = (tmp_path / "logs" / f"{job_id}.log").read_text(encoding="utf-8")
    for expected in (
        "structlog line",
        "raw fd 1 line",
        "raw fd 2 line",
        "subprocess line",
        "self-test recipe done",
    ):
        assert expected in text
    assert "\x1b[" not in text
    assert marker.read_text(encoding="utf-8") == "bytecode-off"


def test_submit_refuses_a_second_job_on_the_same_output(tmp_path: Path) -> None:
    from exosim.mcp.jobs import JobConflictError

    manager = JobManager()
    first = manager.submit(
        _spec(sleep=30), log_dir=tmp_path / "logs", max_runtime=None, max_jobs=5
    )
    try:
        with pytest.raises(JobConflictError) as caught:
            manager.submit(
                _spec(sleep=30), log_dir=tmp_path / "logs", max_runtime=None, max_jobs=5
            )
        assert caught.value.job_id == first
        # a different output is fine
        other = _spec(sleep=0)
        other["output"] = "other.h5"
        second = manager.submit(
            other, log_dir=tmp_path / "logs", max_runtime=None, max_jobs=5
        )
        _wait_until_terminal(manager, second)
    finally:
        manager.cancel(first)
    # once the first job is over, the output is free again
    third = manager.submit(
        _spec(sleep=0), log_dir=tmp_path / "logs", max_runtime=None, max_jobs=5
    )
    _wait_until_terminal(manager, third)
