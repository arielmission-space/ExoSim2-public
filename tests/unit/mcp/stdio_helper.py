"""Minimal raw JSON-RPC client for the stdio MCP server (used by server-level tests)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_stdio_session(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    work_dir: Path,
    repo_root: Path = REPO_ROOT,
    timeout: float = 180.0,
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Start the real server over stdio and issue ``tools/call`` requests.

    Returns
    -------
    tuple
        Raw stdout lines, the ``result`` of each call (in order), and stderr.
    """
    env = dict(os.environ)
    env["EXOSIM_MCP_EXEC_BACKEND"] = "inline"
    messages: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    for index, (name, arguments) in enumerate(calls, start=1):
        messages.append(
            {
                "jsonrpc": "2.0",
                "id": index,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "exosim.mcp.server",
            "--repo-root",
            str(repo_root),
            "--work-dir",
            str(work_dir),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(work_dir.parent),
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True
    )
    stderr_thread.start()
    # Close the pipe only after every answer arrived: EOF cancels in-flight calls.
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    lines: list[str] = []
    results: dict[int, dict[str, Any]] = {}
    try:
        for message in messages:
            proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        wanted = set(range(1, len(calls) + 1))
        while wanted:
            line = proc.stdout.readline()
            if not line:
                break
            if not line.strip():
                continue
            lines.append(line.rstrip("\n"))
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict) and message.get("id") in wanted:
                results[message["id"]] = message
                wanted.discard(message["id"])
        proc.stdin.close()
        # Anything still written before exit also counts as stdout output.
        lines.extend(
            row.rstrip("\n") for row in proc.stdout.read().splitlines() if row.strip()
        )
        proc.wait(timeout=30)
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        stderr_thread.join(timeout=5)
    ordered = [results.get(i, {}) for i in range(1, len(calls) + 1)]
    return lines, ordered, "".join(stderr_chunks)


def result_text(message: dict[str, Any]) -> str:
    """Concatenate the text content of a ``tools/call`` response."""
    result = message.get("result") or {}
    parts = [c.get("text", "") for c in result.get("content", [])]
    if "error" in message:
        parts.append(json.dumps(message["error"]))
    return "\n".join(parts)
