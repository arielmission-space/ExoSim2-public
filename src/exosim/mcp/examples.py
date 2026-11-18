"""MCP helper: catalogue the bundled example configurations.

An AI client asked to run or explain something should start from a real example
config, not an invented one. ``list_examples`` returns every ``examples/**.xml``
with its structure, its ``<config>`` includes, and which pipeline it suits.
"""

from __future__ import annotations

import re
from typing import Any

from exosim.mcp.schemas import resolve_repo_path

_CONFIG_RE = re.compile(r"<config>\s*([^<]+?)\s*</config>")
_TYPE_RE = re.compile(r"<type>\s*([^<]+?)\s*</type>")

# Curated one-liners for the well-known files; anything else gets a structural
# description built from its contents.
_DESCRIPTIONS = {
    "examples/main_example.xml": (
        "Entry point for the focal-plane / sub-exposure / NDR pipelines: grids, "
        "pointing, jitter, and sky + payload includes."
    ),
    "examples/radiometric/main_radiometric.xml": (
        "Entry point for the radiometric pipeline (target-list mode)."
    ),
    "examples/payload_example.xml": "Payload include: telescope and channels.",
    "examples/sky_example.xml": "Sky / source configuration include.",
    "examples/radiometric/sky_radiometric.xml": "Sky include for the radiometric example.",
    "examples/telescope.xml": "Telescope optics include.",
    "examples/photometer.xml": "Photometer channel include.",
    "examples/spectrometer.xml": "Spectrometer channel include.",
    "examples/tools_input_example.xml": "Input for the standalone map / readout tools.",
}

_PIPELINE_HINTS = {
    "examples/main_example.xml": ["focalplane", "subexposures", "ndrs"],
    "examples/radiometric/main_radiometric.xml": ["radiometric"],
}


def _structural_description(sections: list[str], type_tag: str | None) -> str:
    if type_tag:
        return f"{type_tag.capitalize()} channel configuration."
    if "payload" in sections:
        return "Payload configuration (telescope and channels)."
    if "sky" in sections or "source" in sections:
        return "Sky / source configuration."
    return "Configuration fragment: " + ", ".join(sections[:6])


def list_example_configs(repo_root: str | None = None) -> dict[str, Any]:
    """Return the bundled example configurations with a short description each."""
    root = resolve_repo_path(".", repo_root)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.glob("examples/**/*.xml")):
        rel = path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        sections = re.findall(r"^\s{4}<([A-Za-z_]+)", text, flags=re.MULTILINE)
        includes = [m.replace("__ConfigPath__/", "") for m in _CONFIG_RE.findall(text)]
        type_tag = (_TYPE_RE.search(text) or [None, None])[1]
        entries.append(
            {
                "path": rel,
                "description": _DESCRIPTIONS.get(
                    rel, _structural_description(sections, type_tag)
                ),
                "is_entrypoint": bool(includes),
                "includes": includes,
                "suits_pipelines": _PIPELINE_HINTS.get(rel, []),
            }
        )
    return {"examples": entries, "count": len(entries)}
