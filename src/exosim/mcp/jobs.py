"""Background job manager for MCP pipeline runs.

An ExoSim simulation can take minutes to hours. Running it in-process would block
the MCP event loop, could not be cancelled, and would leak global
``RunConfig`` state into the server. Instead each run happens in a separate
process that this module supervises: it can be waited on with a timeout,
cancelled, and its log inspected.

The manager is a process-local singleton (:data:`JOBS`); jobs do not survive a
server restart.

Lifecycle guarantees
--------------------
* Children never outlive the server. Children are deliberately *not* daemonic
  (a daemonic process may not start worker processes of its own, which ExoSim
  can do). Instead (a) an ``atexit`` hook registered here terminates every
  running child before :mod:`multiprocessing`'s own exit hook would block
  joining it (atexit hooks run last-in first-out and ours is registered later),
  (b) SIGTERM / SIGINT handlers terminate the children, escalating from
  ``terminate`` to ``kill`` after a short grace period, then chain to the
  previous handler, and (c) each child watches its parent pid and exits by
  itself if the server is killed without any chance to clean up (SIGKILL).
* ``max_runtime`` is enforced by a watchdog thread, with no polling needed.
* At most ``EXOSIM_MCP_MAX_JOBS`` (default 2) jobs run at once.
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ExoSim imports numba/h5py at import time; only "spawn" is safe here.
_MP = mp.get_context("spawn")

LOG_TAIL_CHARS = 4000

ENV_MAX_JOBS = "EXOSIM_MCP_MAX_JOBS"
DEFAULT_MAX_JOBS = 2
_KILL_GRACE_SECONDS = 3.0
_WATCHDOG_INTERVAL = 0.25
_PARENT_POLL_SECONDS = 1.0

_STATUS_RUNNING = "running"
_STATUS_SUCCEEDED = "succeeded"
_STATUS_FAILED = "failed"
_STATUS_CANCELLED = "cancelled"
_STATUS_TIMEOUT = "timeout"

_TERMINAL = {_STATUS_SUCCEEDED, _STATUS_FAILED, _STATUS_CANCELLED, _STATUS_TIMEOUT}


class JobLimitError(RuntimeError):
    """Raised by :meth:`JobManager.submit` when too many jobs are running."""

    def __init__(self, limit: int, running: list[str]) -> None:
        self.limit = limit
        self.running = running
        super().__init__(
            f"Too many concurrent jobs: {len(running)} running "
            f"(limit {limit}, set by {ENV_MAX_JOBS}). Running job ids: "
            f"{', '.join(running)}. Wait for one to finish with get_job_status, "
            f"cancel one with cancel_job, or raise the limit via {ENV_MAX_JOBS} "
            "when starting the server."
        )


class JobConflictError(RuntimeError):
    """Raised by :meth:`JobManager.submit` when a running job owns the output."""

    def __init__(self, output: str, job_id: str) -> None:
        self.output = output
        self.job_id = job_id
        super().__init__(
            f"Output {output} is already being written by running job {job_id}. "
            "Wait for it with get_job_status, cancel it with cancel_job, or pick "
            "another output path (overwrite=True does not apply to a running job)."
        )


def max_jobs_from_env() -> int:
    """Concurrent-job cap from ``EXOSIM_MCP_MAX_JOBS`` (0 / invalid -> default)."""
    try:
        value = int(os.environ.get(ENV_MAX_JOBS, "").strip())
    except ValueError:
        return DEFAULT_MAX_JOBS
    return value if value > 0 else DEFAULT_MAX_JOBS


def _watch_parent(parent_pid: int) -> None:
    """Exit this child if the server dies without cleaning up (e.g. SIGKILL)."""
    while True:
        time.sleep(_PARENT_POLL_SECONDS)
        if os.getppid() != parent_pid:
            os._exit(1)


def _worker(spec: dict[str, Any], log_path: str, parent_pid: int | None = None) -> None:
    """Subprocess entry point: run one recipe, logging stdout/stderr to a file.

    Both the Python streams and the raw file descriptors 1 and 2 go to the log
    (so the ``RUN CONFIGURATION`` banner and the output of any subprocess the
    recipe starts land there, not on the server's streams), without ANSI colours,
    and no ``__pycache__`` is written into the content root.
    """
    import contextlib
    import sys
    import traceback

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if parent_pid is not None:
        threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()

    with open(log_path, "w", encoding="utf-8", buffering=1) as handle:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                from exosim.log import structlog_config
                from exosim.mcp.pipelines import run_recipe

                structlog_config.set_log_colors(False)
                structlog_config.configure_structlog(
                    log_level=os.environ.get("EXOSIM_LOG_LEVEL", "INFO"),
                    development=os.environ.get("EXOSIM_DEV_MODE", "true").lower()
                    == "true",
                    json_logs=os.environ.get("EXOSIM_JSON_LOGS", "false").lower()
                    == "true",
                )
                run_recipe(spec)
            except BaseException:
                traceback.print_exc(file=handle)
                sys.exit(1)


@dataclass
class Job:
    """State of a single pipeline run."""

    id: str
    pipeline: str
    output: str
    log_path: str
    status: str = _STATUS_RUNNING
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    deadline: float | None = None
    returncode: int | None = None
    reason: str | None = None
    _process: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable status, with the log tail once the job has finished."""
        data: dict[str, Any] = {
            "job_id": self.id,
            "pipeline": self.pipeline,
            "status": self.status,
            "output": self.output,
            "log_file": self.log_path,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_seconds": (
                round(
                    (self.finished_at or time.time())
                    - (self.started_at or self.submitted_at),
                    2,
                )
            ),
        }
        if self.reason:
            data["reason"] = self.reason
        if self.status in _TERMINAL and self.status != _STATUS_SUCCEEDED:
            data["log_tail"] = _read_tail(self.log_path)
        return data


def _read_tail(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-LOG_TAIL_CHARS:]


class JobManager:
    """Supervises pipeline subprocesses for one MCP server."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._watchdog: threading.Thread | None = None
        self._atexit_registered = False
        self._signals_installed = False

    # -- lifecycle --------------------------------------------------------
    def submit(
        self,
        spec: dict[str, Any],
        *,
        log_dir: str | Path,
        max_runtime: float | None,
        max_jobs: int | None = None,
    ) -> str:
        """Start a pipeline subprocess and return its job id.

        Raises
        ------
        JobConflictError
            If a running job already writes the same output path.
        JobLimitError
            If ``max_jobs`` (default: ``EXOSIM_MCP_MAX_JOBS``) jobs already run.
        """
        limit = max_jobs or max_jobs_from_env()
        with self._lock:
            running = self._running_ids()
            for running_id in running:
                if self._jobs[running_id].output == spec["output"]:
                    raise JobConflictError(spec["output"], running_id)
            if len(running) >= limit:
                raise JobLimitError(limit, running)

            job_id = uuid.uuid4().hex[:12]
            log_path = str(Path(log_dir) / f"{job_id}.log")
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            self.install_hooks()
            process = _MP.Process(
                target=_worker,
                args=(spec, log_path, os.getpid()),
                name=f"exosim-mcp-{job_id}",
            )
            # the spawned child copies the environment at start: without colours
            # from its first import (the RUN CONFIGURATION banner is logged then)
            previous = os.environ.get("EXOSIM_LOG_COLORS")
            os.environ["EXOSIM_LOG_COLORS"] = "false"
            try:
                process.start()
            finally:
                if previous is None:
                    os.environ.pop("EXOSIM_LOG_COLORS", None)
                else:
                    os.environ["EXOSIM_LOG_COLORS"] = previous
            now = time.time()
            self._jobs[job_id] = Job(
                id=job_id,
                pipeline=spec["pipeline"],
                output=spec["output"],
                log_path=log_path,
                started_at=now,
                deadline=(now + max_runtime) if max_runtime else None,
                _process=process,
            )
            self._ensure_watchdog()
            return job_id

    def wait(self, job_id: str, timeout: float | None) -> dict[str, Any]:
        """Block up to ``timeout`` seconds for the job, then report its status."""
        job = self._get(job_id)
        process = job._process
        if process is not None and job.status == _STATUS_RUNNING:
            # the watchdog may reap (close) the process while we wait
            with contextlib.suppress(ValueError, AttributeError):
                process.join(timeout)
        return self.status(job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        """Return the current status, reaping the process if it has exited."""
        with self._lock:
            job = self._get(job_id)
            self._refresh(job)
            return job.to_dict()

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Terminate a running job."""
        with self._lock:
            job = self._get(job_id)
            if job.status == _STATUS_RUNNING and job._process is not None:
                _terminate(job._process)
                self._finish(job, _STATUS_CANCELLED, reason="cancelled by request")
            return job.to_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return every known job, oldest first."""
        with self._lock:
            for job in self._jobs.values():
                self._refresh(job)
            return [
                job.to_dict()
                for job in sorted(self._jobs.values(), key=lambda j: j.submitted_at)
            ]

    def running_ids(self) -> list[str]:
        """Ids of the jobs currently running."""
        with self._lock:
            return self._running_ids()

    def shutdown(self) -> None:
        """Terminate every running child (idempotent; used by atexit/signals)."""
        with self._lock:
            procs = [
                (job, job._process)
                for job in self._jobs.values()
                if job.status == _STATUS_RUNNING and job._process is not None
            ]
            for _, process in procs:
                try:
                    if process.is_alive():
                        process.terminate()
                except (ValueError, AttributeError, OSError):
                    pass
            end = time.time() + _KILL_GRACE_SECONDS
            for _, process in procs:
                try:
                    process.join(max(0.0, end - time.time()))
                    if process.is_alive():
                        process.kill()
                        process.join(2)
                except (ValueError, AttributeError, OSError):
                    pass
            for job, _ in procs:
                self._finish(job, _STATUS_CANCELLED, reason="server shut down")

    # -- internals ------------------------------------------------------
    def _get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"Unknown job id: {job_id}") from None

    def _running_ids(self) -> list[str]:
        for job in self._jobs.values():
            self._refresh(job)
        return [j.id for j in self._jobs.values() if j.status == _STATUS_RUNNING]

    def _finish(self, job: Job, status: str, *, reason: str | None = None) -> None:
        """Record a terminal state, reap the process and release its handle."""
        process = job._process
        job.status = status
        job.finished_at = time.time()
        if reason:
            job.reason = reason
        if process is not None:
            try:
                process.join(0.1)
                job.returncode = process.exitcode
                if not process.is_alive():
                    process.close()
            except (ValueError, AttributeError, OSError):
                pass
            job._process = None

    def _refresh(self, job: Job) -> None:
        if job.status != _STATUS_RUNNING or job._process is None:
            return
        process = job._process
        try:
            alive = process.is_alive()
        except ValueError:  # already closed
            return
        if alive:
            if job.deadline is not None and time.time() > job.deadline:
                _terminate(process)
                self._finish(
                    job,
                    _STATUS_TIMEOUT,
                    reason="exceeded max_runtime and was terminated by the watchdog",
                )
            return
        code = process.exitcode
        self._finish(
            job,
            _STATUS_SUCCEEDED if code == 0 else _STATUS_FAILED,
            reason=None if code == 0 else f"exit code {code}",
        )

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(
            target=self._watch, name="exosim-mcp-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watch(self) -> None:
        """Enforce deadlines and reap finished children without any polling."""
        while True:
            time.sleep(_WATCHDOG_INTERVAL)
            with self._lock:
                running = [
                    j for j in self._jobs.values() if j.status == _STATUS_RUNNING
                ]
                for job in running:
                    self._refresh(job)
                if not any(j.status == _STATUS_RUNNING for j in running):
                    self._watchdog = None
                    return

    def install_hooks(self) -> None:
        """Register the atexit and SIGTERM/SIGINT hooks (idempotent).

        Signal handlers can only be set from the main thread, and mcp 2.x runs
        sync tools in worker threads, so ``server.main()`` calls this on the main
        thread before serving. A call from another thread registers only atexit
        and leaves the signal handlers for a later main-thread call.
        """
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self.shutdown)
        if self._signals_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        self._signals_installed = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)
                signal.signal(sig, self._make_handler(previous))
            except (ValueError, OSError):
                continue

    def _make_handler(self, previous: Any) -> Any:
        def handler(signum: int, frame: Any) -> None:
            self.shutdown()
            if callable(previous):
                previous(signum, frame)
            elif signum == signal.SIGINT:
                raise KeyboardInterrupt
            else:  # SIG_DFL / SIG_IGN: re-deliver with the default action
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        return handler


def _terminate(process: Any) -> None:
    """Ask the child to stop, then kill it after a short grace period."""
    process.terminate()
    process.join(_KILL_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(5)


class _SelfTestRecipe:
    """Trivial 'recipe' used only by the job-manager tests.

    Referenced as ``exosim.mcp.jobs._SelfTestRecipe`` in a run spec so the real
    subprocess path can be exercised without a heavy ExoSim simulation.
    """

    def __init__(
        self,
        *,
        sleep: float = 0.0,
        fail: bool = False,
        marker: str | None = None,
        chatty: bool = False,
    ) -> None:
        if marker:
            Path(marker).write_text(
                "bytecode-off" if sys.dont_write_bytecode else "bytecode-on",
                encoding="utf-8",
            )
        if chatty:  # exercise the log capture: structlog, raw fd, subprocess
            import subprocess

            from exosim.log.logger import get_logger

            get_logger("exosim.selftest").info("structlog line")
            os.write(1, b"raw fd 1 line\n")
            os.write(2, b"raw fd 2 line\n")
            subprocess.run(
                [sys.executable, "-c", "print('subprocess line')"], check=True
            )
        print("self-test recipe running")
        time.sleep(sleep)
        if fail:
            raise RuntimeError("self-test failure")
        print("self-test recipe done")


JOBS = JobManager()
