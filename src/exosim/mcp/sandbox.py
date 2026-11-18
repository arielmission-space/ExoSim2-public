"""Filesystem sandbox for the ExoSim MCP server.

An MCP client is an autonomous program driven by a language model: it can ask
the server to read or write *any* path it likes. Every filesystem access made by
a tool must therefore be confined to an explicit allowlist of directories.

The sandbox has two kinds of root:

``repo_root``
    The *content root*: the ExoSim checkout, or the documentation/examples
    packaged inside the wheel (see :mod:`exosim.mcp.assets` for the precedence).
    Tools may **read** configuration files, examples and source here, but never
    write.
``work_dir``
    A dedicated directory (default ``<repo_root>/mcp-runs``, or
    ``./mcp-runs`` when the content is packaged) where simulation outputs and
    logs are written and may be read back.

Extra read/write roots can be added with ``EXOSIM_MCP_ALLOW_DIRS`` for users who
keep their configurations outside the checkout.

Environment variables (all optional, overridden by explicit arguments):

=========================== ==================================================
``EXOSIM_MCP_REPO_ROOT``     content root override (default: checkout, else packaged)
``EXOSIM_MCP_WORK_DIR``      writable output directory (default: ``<repo>/mcp-runs``;
                             ``~/.exosim/mcp-runs`` if that cannot be created)
``EXOSIM_MCP_ALLOW_DIRS``    extra allowed roots, ``os.pathsep``-separated
``EXOSIM_MCP_NO_EXEC``       if truthy, ``run_pipeline`` is disabled
``EXOSIM_MCP_MAX_THREADS``   hard cap on the worker threads a run may request
``EXOSIM_MCP_MAX_RUNTIME``   hard wall-clock limit per run, seconds (default 3600;
                             zero, negative or invalid -> default, with a warning)
``EXOSIM_MCP_MAX_JOBS``      concurrent runs allowed (default 2; invalid or 0 -> default)
``EXOSIM_MCP_EXEC_BACKEND``  ``process`` (default) or ``inline`` (debugging/tests)
=========================== ==================================================
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from exosim.mcp.assets import (
    ENV_REPO_ROOT,  # noqa: F401  (re-exported for the server CLI)
    content_report,
    resolve_content_root,
    warn_if_content_missing,
)

ENV_WORK_DIR = "EXOSIM_MCP_WORK_DIR"
ENV_ALLOW_DIRS = "EXOSIM_MCP_ALLOW_DIRS"
ENV_NO_EXEC = "EXOSIM_MCP_NO_EXEC"
ENV_MAX_THREADS = "EXOSIM_MCP_MAX_THREADS"
ENV_MAX_RUNTIME = "EXOSIM_MCP_MAX_RUNTIME"
ENV_EXEC_BACKEND = "EXOSIM_MCP_EXEC_BACKEND"

DEFAULT_WORK_DIRNAME = "mcp-runs"
FALLBACK_WORK_DIR = Path("~/.exosim") / DEFAULT_WORK_DIRNAME  # per-user, see server
DEFAULT_MAX_RUNTIME = 3600.0
_TRUTHY = {"1", "true", "yes", "on"}


class PathNotAllowedError(ValueError):
    """Raised when a tool is asked to touch a path outside the sandbox."""


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _runtime_env() -> float:
    """``EXOSIM_MCP_MAX_RUNTIME`` in seconds; zero, negative or invalid -> default."""
    raw = os.environ.get(ENV_MAX_RUNTIME, "").strip()
    if not raw:
        return DEFAULT_MAX_RUNTIME
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        warnings.warn(
            f"{ENV_MAX_RUNTIME}={raw!r} is not a positive number of seconds; "
            f"using the default ({DEFAULT_MAX_RUNTIME:g} s).",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_MAX_RUNTIME
    return value


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


@dataclass(frozen=True)
class Sandbox:
    """The directories an MCP tool is allowed to touch, plus run limits."""

    repo_root: Path
    work_dir: Path
    extra_roots: tuple[Path, ...] = field(default_factory=tuple)
    allow_exec: bool = True
    max_threads: int | None = None
    max_runtime: float | None = DEFAULT_MAX_RUNTIME
    exec_backend: str = "process"
    content_source: str = "override"

    @classmethod
    def from_env(
        cls,
        *,
        repo_root: str | Path | None = None,
        work_dir: str | Path | None = None,
        allow_exec: bool | None = None,
    ) -> Sandbox:
        """Build a sandbox from explicit arguments, then the environment."""
        content = resolve_content_root(repo_root)
        root, source = _resolve(content.path), content.source
        # Never write runs into site-packages: packaged content uses ./mcp-runs.
        work_base = Path.cwd() if source == "packaged" else root
        work = _resolve(
            work_dir or os.environ.get(ENV_WORK_DIR) or work_base / DEFAULT_WORK_DIRNAME
        )
        warn_if_content_missing(root, source)
        extra = tuple(
            _resolve(item)
            for item in os.environ.get(ENV_ALLOW_DIRS, "").split(os.pathsep)
            if item.strip()
        )
        if allow_exec is None:
            allow_exec = os.environ.get(ENV_NO_EXEC, "").strip().lower() not in _TRUTHY
        backend = os.environ.get(ENV_EXEC_BACKEND, "process").strip().lower()
        return cls(
            repo_root=root,
            work_dir=work,
            extra_roots=extra,
            allow_exec=allow_exec,
            max_threads=_int_env(ENV_MAX_THREADS),
            max_runtime=_runtime_env(),
            exec_backend=backend if backend in {"process", "inline"} else "process",
            content_source=source,
        )

    def log_dir(self) -> Path:
        """Directory for per-job log files."""
        return self.work_dir / "logs"

    def clamp_threads(self, requested: int | None) -> int | None:
        """Clamp a requested worker-thread count to ``max_threads``."""
        if requested is None or self.max_threads is None:
            return requested
        return max(1, min(requested, self.max_threads))

    # -- roots ----------------------------------------------------------------
    @property
    def readable_roots(self) -> tuple[Path, ...]:
        """Directories a tool may read from."""
        return (self.repo_root, self.work_dir, *self.extra_roots)

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        """Directories a tool may write to (never the checkout itself)."""
        return (self.work_dir, *self.extra_roots)

    # -- resolution ---------------------------------------------------------
    def _to_abs(self, raw: str | Path, base: Path) -> Path:
        candidate = Path(raw).expanduser()
        return _resolve(candidate if candidate.is_absolute() else base / candidate)

    def _readable(self, target: Path) -> bool:
        return any(_within(target, root) for root in self.readable_roots)

    def resolve_read(self, raw: str | Path, *, prefer_work_dir: bool = False) -> Path:
        """Resolve a path a tool wants to read; confine it to a readable root.

        Relative paths are taken against the content root (``repo_root``). With
        ``prefer_work_dir`` they are tried against the working directory first
        (so ``focal.h5`` produced by a run is found) and fall back to the content
        root (so ``examples/...`` still works). Whichever candidate is used must
        lie in a readable root after symlinks and ``..`` are resolved.
        """
        escaped = False
        if prefer_work_dir and not Path(raw).expanduser().is_absolute():
            candidate = self._to_abs(raw, self.work_dir)
            if self._readable(candidate):
                if candidate.exists():
                    return candidate
            else:
                # e.g. a symlink in the work dir pointing out of the sandbox
                escaped = os.path.lexists(self.work_dir / Path(raw).expanduser())
        target = self._to_abs(raw, self.repo_root)
        if escaped and self._readable(target) and not target.exists():
            raise PathNotAllowedError(
                f"'{raw}' exists in the working directory but resolves outside the "
                "directories this MCP server may read "
                f"({', '.join(str(r) for r in self.readable_roots)})"
            )
        if not self._readable(target):
            raise PathNotAllowedError(
                f"{target} is outside the directories this MCP server may read "
                f"({', '.join(str(r) for r in self.readable_roots)})"
            )
        return target

    def resolve_write(self, raw: str | Path) -> Path:
        """Resolve a path a tool wants to write; confine it to the work dir."""
        target = self._to_abs(raw, self.work_dir)
        if not any(_within(target, root) for root in self.writable_roots):
            raise PathNotAllowedError(
                f"{target} is outside the writable working directory "
                f"({', '.join(str(r) for r in self.writable_roots)}); "
                "pass an output path under the working directory"
            )
        return target

    def work_dir_usable(self) -> bool:
        """True if the working directory exists or can be created (not created here)."""
        probe = self.work_dir
        while not probe.exists():
            if probe.parent == probe:
                return False
            probe = probe.parent
        return probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)

    def ensure_work_dir(self) -> Path:
        """Create the working directory if needed and return it."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir

    def to_dict(self) -> dict[str, object]:
        """Serialisable description, for the ``exosim_project_context`` tool."""
        return {
            "repo_root": str(self.repo_root),
            "content": content_report(self.repo_root, self.content_source),
            "work_dir": str(self.work_dir),
            "extra_roots": [str(r) for r in self.extra_roots],
            "execution_enabled": self.allow_exec,
            "max_threads": self.max_threads,
            "max_runtime_seconds": self.max_runtime,
        }
