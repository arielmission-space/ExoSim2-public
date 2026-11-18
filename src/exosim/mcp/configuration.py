"""Configuration validation helpers for the ExoSim MCP layer."""

from __future__ import annotations

import threading
import warnings
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

from astropy import units as u
from astropy.units import UnitsWarning

from exosim.mcp.schemas import resolve_repo_path
from exosim.tasks.load.load_options import LoadOptions
from exosim.utils import RunConfig

if TYPE_CHECKING:
    from pathlib import Path


# ``LoadOptions`` appends every file it reads to the process-wide
# ``RunConfig.config_file_list``; the includes of one call are found by diffing it.
# Sync tools run in worker threads (mcp 2.x), so any code that loads a configuration
# and reads that list must hold this lock for the whole load-and-diff.
CONFIG_LOAD_LOCK = threading.RLock()


def config_root(main_path: Path) -> Path:
    """The directory ``__ConfigPath__`` expands to for a configuration file.

    This is the file's own ``<ConfigPath>`` taken relative to the file (``.`` when
    absent, unreadable or not XML), which is how ExoSim itself reads it.
    """
    try:
        root = ET.parse(main_path).getroot()
    except (ET.ParseError, OSError):
        return main_path.parent
    elem = root.find("ConfigPath")
    relative = (elem.text or ".").strip() if elem is not None else "."
    return (main_path.parent / relative).resolve()


def validate_configuration_file(
    configuration: str | Path,
    *,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and summarise an ExoSim configuration file.

    Parameters
    ----------
    configuration : str or pathlib.Path
        XML, YAML or YML configuration file.
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.
    config_path : str or pathlib.Path, optional
        Directory the ``<config>`` includes and ``<datafile>`` references
        resolve against. Defaults to the file's own ``<ConfigPath>`` (relative to
        the file), or the file's directory when it has none.

    Returns
    -------
    dict
        Validation status, included files and a compact structure summary.
    """
    # A draft written by ``draft_config`` lives in the working directory, so a
    # relative path is tried there before the repository root.
    configuration_path = resolve_repo_path(
        configuration, repo_root, prefer_work_dir=True
    )
    # An explicit ``config_path`` wins; otherwise honour the file's own
    # ``<ConfigPath>`` relative to the file (e.g. ``..`` in the radiometric example).
    resolve_from = (
        str(config_path)
        if config_path is not None
        else str(config_root(configuration_path))
    )
    with CONFIG_LOAD_LOCK:
        offset = len(RunConfig.config_file_list)
        try:
            # Resolve relative ``<config>`` includes next to the configuration file,
            # which is how ExoSim configurations are normally written.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnitsWarning)
                options = LoadOptions()(
                    filename=str(configuration_path),
                    config_path=resolve_from,
                )
        except Exception as exc:
            return {
                "valid": False,
                "configuration": str(configuration_path),
                "error": f"{type(exc).__name__}: {exc}",
                "included_files": [],
            }

        # Positional, not a set difference: a file validated before is appended
        # again and must be reported again.
        included_files = list(RunConfig.config_file_list[offset:])
    return {
        "valid": True,
        "configuration": str(configuration_path),
        "included_files": included_files,
        "top_level_sections": list(options.keys()),
        "summary": _summarise(options),
    }


_EXAMPLE_FOR_PIPELINE = {
    "focalplane": "examples/main_example.xml",
    "subexposures": "examples/main_example.xml",
    "ndrs": "examples/main_example.xml",
    "radiometric": "examples/radiometric/main_radiometric.xml",
}


def config_requirements(
    pipeline: str,
    configuration: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Report the configuration sections a pipeline expects.

    The reference is the bundled example for that pipeline: its top-level
    sections are what a working configuration needs. If ``configuration`` is
    given, the missing and extra sections are reported too, each with a pointer
    to the documentation.

    Parameters
    ----------
    pipeline : str
        Pipeline name (``focalplane``, ``radiometric``, ...).
    configuration : str or pathlib.Path, optional
        A configuration to check against the reference.
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.

    Returns
    -------
    dict
        ``reference_sections`` (with a doc link each) and, if a configuration
        was given, ``present`` / ``missing`` / ``extra``.
    """
    from exosim.mcp.retrieval import search_knowledge_sources

    key = pipeline.lower()
    example = _EXAMPLE_FOR_PIPELINE.get(key)
    if example is None:
        return {"error": f"Unknown pipeline: {pipeline}", "pipeline": pipeline}

    reference = validate_configuration_file(example, repo_root=repo_root)
    if not reference.get("valid"):
        return {"error": "reference example failed to load", "detail": reference}
    boilerplate = {"ConfigPath", "fileName", "random_seed", "comment"}
    ref_sections = [
        s
        for s in reference["top_level_sections"]
        if not s.startswith("__") and s not in boilerplate
    ]

    def doc_for(section: str) -> str | None:
        hits = search_knowledge_sources(
            f"{section} configuration", repo_root=repo_root, max_results=5
        )
        if not hits:
            return None
        # Prefer a user-facing documentation page over a source file or an
        # internal agent note, so the pointer is something a person can read.
        for hit in hits:
            if hit.path.startswith("docs/source/"):
                return hit.path
        return hits[0].path

    summary = reference.get("summary", {})

    def keys_for(section: str) -> list[str]:
        node = summary.get(section)
        return sorted(node) if isinstance(node, dict) else []

    result: dict[str, Any] = {
        "pipeline": key,
        "reference_example": example,
        "reference_sections": [
            {"section": s, "keys": keys_for(s), "see": doc_for(s)} for s in ref_sections
        ],
    }

    if configuration is not None:
        checked = validate_configuration_file(configuration, repo_root=repo_root)
        if not checked.get("valid"):
            result["configuration_error"] = checked.get("error")
            return result
        present = [
            s
            for s in checked["top_level_sections"]
            if not s.startswith("__") and s not in boilerplate
        ]
        result["present"] = [s for s in ref_sections if s in present]
        result["missing"] = [s for s in ref_sections if s not in present]
        result["extra"] = [s for s in present if s not in ref_sections]

    return result


def _summarise(value: Any, *, depth: int = 0, max_depth: int = 3) -> Any:
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key): _summarise(child, depth=depth + 1, max_depth=max_depth)
            for key, child in value.items()
        }
    if isinstance(value, u.Quantity):
        return {
            "type": "Quantity",
            "unit": str(value.unit),
            "shape": getattr(value.value, "shape", None),
        }
    return type(value).__name__
