"""MCP resources: the curated knowledge-base files, browsable by URI.

Tools *do* things; resources are content a client can list and read. The MCP
server exposes every file in the knowledge-base manifest under
``exosim://kb/<repo-relative-path>`` so a client can pull a whole page rather
than only the passages a search returned.
"""

from __future__ import annotations

from exosim.mcp.retrieval import knowledge_base_manifest
from exosim.mcp.schemas import resolve_repo_path

KB_URI_PREFIX = "exosim://kb/"
_MAX_RESOURCE_BYTES = 200_000


def knowledge_base_index() -> dict[str, object]:
    """List the readable knowledge-base resources."""
    manifest = knowledge_base_manifest()
    return {
        "resources": [
            {"uri": f"{KB_URI_PREFIX}{path}", "path": path}
            for path in manifest["matched_files"]
        ],
        "count": manifest["matched_file_count"],
    }


def read_knowledge_file(path: str) -> str:
    """Return the text of one knowledge-base file.

    Parameters
    ----------
    path : str
        Repository-relative path, as listed by :func:`knowledge_base_index`.

    Raises
    ------
    FileNotFoundError
        If the path is not part of the approved knowledge base.
    """
    allowed = set(knowledge_base_manifest()["matched_files"])
    if path not in allowed:
        raise FileNotFoundError(
            f"{path} is not part of the ExoSim knowledge base; "
            "see knowledge_base_manifest for the approved files"
        )
    resolved = resolve_repo_path(path)
    text = resolved.read_text(encoding="utf-8", errors="replace")
    if len(text.encode("utf-8")) > _MAX_RESOURCE_BYTES:
        text = text[:_MAX_RESOURCE_BYTES] + "\n\n... [truncated]\n"
    return text
