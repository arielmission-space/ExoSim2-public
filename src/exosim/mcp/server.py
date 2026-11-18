"""MCP server entry point for ExoSim."""

from __future__ import annotations

import argparse
import functools
import os
import sys
from importlib.metadata import (
    PackageNotFoundError,
    version as _pkg_version,
)
from typing import TYPE_CHECKING, Any

from exosim.mcp import prompts, resources, tools
from exosim.mcp.assets import ContentMissingError
from exosim.mcp.embeddings import EmbeddingsUnavailableError
from exosim.mcp.sandbox import (
    ENV_NO_EXEC,
    ENV_REPO_ROOT,
    ENV_WORK_DIR,
    FALLBACK_WORK_DIR,
    Sandbox,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_TRANSPORTS = {"stdio": "stdio", "sse": "sse", "http": "streamable-http"}

# Failures a tool raises on purpose (sandbox refusals, missing files, bad
# arguments, an unavailable optional extra). The SDK would reduce any other
# exception to "Error executing tool X". ``RuntimeError`` is deliberately NOT
# listed as a whole: it is also what a genuine bug often raises, and those must keep
# the SDK's generic error (the traceback goes to the server's stderr). Only the
# RuntimeError subclasses this package defines for user-facing conditions are.
_EXPECTED_ERRORS = (
    OSError,
    ValueError,
    ImportError,
    KeyError,
    LookupError,
    ContentMissingError,
    EmbeddingsUnavailableError,
)


def _tool_error_class() -> type[Exception]:
    """Return the SDK's anticipated-tool-failure exception."""
    from mcp.server.mcpserver.exceptions import ToolError

    return ToolError


def _expose_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool so its expected errors reach the client with their message.

    The function itself is unchanged (unit tests call it directly and expect the
    original exceptions); only the registered wrapper converts them to the SDK's
    ``ToolError``, which the client receives as an error result carrying the text.
    """
    tool_error = _tool_error_class()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except _EXPECTED_ERRORS as exc:
            # KeyError.__str__ adds quotes around the message.
            message = (
                str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            )
            raise tool_error(message or type(exc).__name__) from exc

    return wrapper


def _major(version: str) -> int | None:
    """Major component of a version string, ``None`` if it cannot be parsed."""
    head = version.split(".", 1)[0].strip()
    return int(head) if head.isdigit() else None


def _load_server_class() -> type:
    """Return the ``mcp`` SDK server class (``MCPServer``, ``mcp`` 2.x only).

    Raises
    ------
    ImportError
        If the SDK is missing or is a 1.x release.
    """
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        try:
            installed = _pkg_version("mcp")
        except PackageNotFoundError:
            installed = None
        if installed is None:
            detail = "the 'mcp' SDK (mcp>=2,<3) is not installed"
        elif _major(installed) not in (None, 2):
            detail = f"mcp {installed} is installed but mcp>=2,<3 is required"
        else:
            # inside the supported range (or unparsable): the import failed for
            # another reason, so do not blame the version.
            detail = f"mcp {installed} is installed but could not be imported"
        raise ImportError(
            f"The ExoSim MCP server needs the optional dependency group: "
            f'pip install "exosim[mcp]" ({detail}). '
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return MCPServer


def _route_logging_to_stderr() -> None:
    """Send ExoSim's console logs to stderr.

    ExoSim's structlog set-up writes to ``sys.stdout``, which is the JSON-RPC
    channel of the stdio transport: one stray log line corrupts the stream.
    """
    from exosim.log import structlog_config

    structlog_config.set_log_stream(sys.stderr)
    structlog_config.configure_structlog(
        log_level=os.environ.get("EXOSIM_LOG_LEVEL", "INFO"),
        log_file=os.environ.get("EXOSIM_LOG_FILE"),
        development=os.environ.get("EXOSIM_DEV_MODE", "true").lower() == "true",
        json_logs=os.environ.get("EXOSIM_JSON_LOGS", "false").lower() == "true",
    )


def _prepare_work_dir(sandbox: Sandbox) -> Sandbox:
    """Make sure the working directory is usable, falling back when it is the default.

    * Execution enabled: the directory is created now; with ``--no-exec`` it is
      only checked (tools create it lazily, e.g. ``draft_config``).
    * An unusable *default* directory (say a read-only ``/mcp-runs`` when started
      from ``/``) falls back to ``~/.exosim/mcp-runs`` with one warning line.
    * An unusable *explicit* directory (``--work-dir`` / ``EXOSIM_MCP_WORK_DIR``)
      is a configuration error: one clear line on stderr and a non-zero exit.
    """
    explicit = bool(os.environ.get(ENV_WORK_DIR))
    try:
        if sandbox.allow_exec:
            sandbox.ensure_work_dir()
            return sandbox
        if sandbox.work_dir_usable() or explicit:
            return sandbox
        reason: Exception | str = "its parent directory is not writable"
    except OSError as exc:
        reason = exc
    if explicit:
        raise SystemExit(
            f"exosim-mcp: error: cannot use the work directory {sandbox.work_dir} "
            f"({reason}); choose a writable one with --work-dir or "
            f"{ENV_WORK_DIR}."
        )
    fallback = FALLBACK_WORK_DIR.expanduser()
    os.environ[ENV_WORK_DIR] = str(fallback)
    moved = Sandbox.from_env(allow_exec=sandbox.allow_exec)
    try:
        if moved.allow_exec:
            moved.ensure_work_dir()
    except OSError as exc:
        raise SystemExit(
            f"exosim-mcp: error: cannot create a work directory: {sandbox.work_dir} "
            f"({reason}) and the fallback {fallback} ({exc}) are both unusable; "
            f"set --work-dir or {ENV_WORK_DIR}."
        ) from None
    print(
        f"exosim-mcp: warning: the default work directory {sandbox.work_dir} is not "
        f"usable ({reason}); using {fallback} instead. Set --work-dir or "
        f"{ENV_WORK_DIR} to choose another.",
        file=sys.stderr,
        flush=True,
    )
    return moved


def build_server(
    *,
    allow_exec: bool | None = None,
    host: str | None = None,
    port: int | None = None,
):
    """Build the ExoSim MCP server.

    Parameters
    ----------
    allow_exec : bool, optional
        Whether to register the execution tools. Defaults to the value derived
        from the environment (``EXOSIM_MCP_NO_EXEC``).
    host, port : optional
        Bind address for the network transports (``sse`` / ``http``).

    Returns
    -------
    MCPServer
        Configured MCP server instance. ``server._exosim_bind`` carries the
        ``(host, port)`` for the network transports, which ``main`` passes as
        ``run()`` keywords.

    Raises
    ------
    ImportError
        If the optional MCP dependency (``mcp>=2,<3``) is not installed.
    """
    server_class = _load_server_class()

    sandbox = Sandbox.from_env(allow_exec=allow_exec)

    server = server_class("exosim")
    server._exosim_bind = (host, port)

    # Core information tools
    server.tool()(_expose_errors(tools.capabilities))
    server.tool()(_expose_errors(tools.exosim_project_context))
    server.tool(name="knowledge_base_manifest")(
        _expose_errors(tools.knowledge_base_manifest_tool)
    )
    server.tool()(_expose_errors(tools.search_knowledge_base))
    # Configuration tools
    server.tool()(_expose_errors(tools.validate_config))
    server.tool()(_expose_errors(tools.compare_configs))
    server.tool()(_expose_errors(tools.config_requirements))
    server.tool()(_expose_errors(tools.draft_config))
    # Pipeline tools
    server.tool()(_expose_errors(tools.list_pipelines))
    server.tool()(_expose_errors(tools.describe_pipeline))
    server.tool()(_expose_errors(tools.list_examples))
    if sandbox.allow_exec:
        server.tool()(_expose_errors(tools.run_pipeline))
        server.tool()(_expose_errors(tools.get_job_status))
        server.tool()(_expose_errors(tools.cancel_job))
        server.tool()(_expose_errors(tools.list_jobs))
    # Output tools
    server.tool()(_expose_errors(tools.inspect_output))
    server.tool()(_expose_errors(tools.explain_output))
    if sandbox.allow_exec:
        server.tool()(_expose_errors(tools.plot_output))
    # Guidance and assistance
    server.tool()(_expose_errors(tools.customisation_guide))
    server.tool()(_expose_errors(tools.suggest_next_steps))
    server.tool()(_expose_errors(tools.diagnose_error))

    # Resources: the knowledge-base files, browsable by URI
    server.resource("exosim://kb")(resources.knowledge_base_index)
    server.resource("exosim://kb/{+path}")(resources.read_knowledge_file)

    # Prompts: ready-made workflows
    for prompt in prompts.ALL_PROMPTS:
        server.prompt()(prompt)

    return server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="exosim-mcp",
        description="Run the ExoSim Model Context Protocol server.",
    )
    parser.add_argument(
        "--transport",
        choices=sorted(_TRANSPORTS),
        default="stdio",
        help="Transport: stdio (default, one local client), sse or http "
        "(networked, for shared or remote clients).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for sse / http (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for sse / http (default: 8000).",
    )
    parser.add_argument(
        "--repo-root",
        metavar="PATH",
        help="Content root to serve (default: EXOSIM_MCP_REPO_ROOT, else the "
        "current directory if it is an ExoSim checkout, else the installed "
        "checkout, else the content packaged in the wheel).",
    )
    parser.add_argument(
        "--work-dir",
        metavar="PATH",
        help="Writable directory for simulation outputs (default: "
        "EXOSIM_MCP_WORK_DIR, else <content root>/mcp-runs for a checkout or "
        "./mcp-runs for packaged content; ~/.exosim/mcp-runs if that cannot "
        "be created). An explicit directory that cannot be used is an error.",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Disable the execution tools (run_pipeline, get_job_status, "
        "cancel_job, list_jobs, plot_output); serve Q&A, drafting and "
        "validation only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the ExoSim MCP server."""
    args = _parse_args(argv)
    if args.repo_root:
        os.environ[ENV_REPO_ROOT] = args.repo_root
    if args.work_dir:
        os.environ[ENV_WORK_DIR] = args.work_dir
    if args.no_exec:
        os.environ[ENV_NO_EXEC] = "1"

    _route_logging_to_stderr()
    sandbox = _prepare_work_dir(Sandbox.from_env())
    where = (
        "stdio"
        if args.transport == "stdio"
        else f"{args.transport} on {args.host}:{args.port}"
    )
    # stdout is the JSON-RPC channel for the stdio transport: log to stderr.
    print(
        f"ExoSim MCP server\n"
        f"  transport:  {where}\n"
        f"  repo root:  {sandbox.repo_root}\n"
        f"  work dir:   {sandbox.work_dir}\n"
        f"  execution:  {'enabled' if sandbox.allow_exec else 'disabled (--no-exec)'}",
        file=sys.stderr,
        flush=True,
    )
    server = build_server(host=args.host, port=args.port)
    transport = _TRANSPORTS[args.transport]
    run_kwargs: dict[str, object] = {}
    if transport != "stdio":
        # The bind address is a run() keyword, not a constructor argument.
        run_kwargs = {"host": args.host, "port": args.port}
    # Signal handlers can only be installed from the main thread; tools run in
    # worker threads, so register them here, before serving.
    from exosim.mcp import jobs

    jobs.JOBS.install_hooks()
    server.run(transport=transport, **run_kwargs)


if __name__ == "__main__":
    main()
