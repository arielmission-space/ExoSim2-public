"""Configuration authoring for the ExoSim MCP layer.

``config_requirements`` tells an AI client *what* a configuration needs;
``draft_configuration`` closes the loop and produces one. It starts from the
bundled reference example for a pipeline, inlines its ``<config>`` includes into
a single self-contained file, applies a set of dotted-path overrides, writes the
result to the sandbox working directory and validates it with the real ExoSim
loader.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from astropy.units import UnitsWarning

from exosim.mcp.assets import require_content_file
from exosim.mcp.configuration import (
    _EXAMPLE_FOR_PIPELINE,
    config_root as _config_root,
    validate_configuration_file,
)
from exosim.mcp.sandbox import Sandbox

_CONFIG_KEYWORD = "__ConfigPath__"


def draft_configuration(
    pipeline: str,
    overrides: dict[str, str] | None = None,
    *,
    repo_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Draft a working configuration for a pipeline and validate it.

    Parameters
    ----------
    pipeline : str
        Pipeline name (``focalplane``, ``radiometric``, ...).
    overrides : dict, optional
        ``"section.subsection.key": "value"`` entries applied to the reference
        example. A value may carry a unit (``"1.1 m"``); the number and the unit
        are written separately, matching ExoSim's XML. Use
        ``config_requirements`` to see the sections and keys available.
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.
    overwrite : bool
        Whether an existing ``drafts/<pipeline>_draft.xml`` may be replaced.

    Returns
    -------
    dict
        ``draft`` (the written file path), ``applied`` / ``not_applied``
        overrides, and the validation outcome (``valid``, ``top_level_sections``
        or ``error``).

    Raises
    ------
    FileExistsError
        If the draft file exists and ``overwrite`` is false.
    ValueError
        If an override sets a path outside the sandbox or points a task at a
        Python file while execution is disabled (see :func:`_check_override`).
    """
    key = pipeline.lower()
    example = _EXAMPLE_FOR_PIPELINE.get(key)
    if example is None:
        return {"error": f"Unknown pipeline: {pipeline}", "pipeline": pipeline}

    sandbox = Sandbox.from_env(repo_root=repo_root)
    example_path = require_content_file(
        sandbox.resolve_read(example), f"Reference configuration {example}"
    )
    config_root = _config_root(example_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        root = _inline_includes(example_path, config_root)

    # Keep data files resolvable from the sandbox working directory and from a
    # later ``exosim`` run: pin the include root to its absolute location.
    config_path_elem = root.find("ConfigPath")
    if config_path_elem is not None:
        config_path_elem.text = str(config_root)

    draft_path = sandbox.resolve_write(Path("drafts") / f"{key}_draft.xml")
    if draft_path.exists() and not overwrite:
        raise FileExistsError(
            f"Draft file already exists: {draft_path}. "
            "Set overwrite=True to replace it."
        )

    applied, not_applied = _apply_overrides(
        root, overrides or {}, sandbox=sandbox, config_root=config_root
    )

    draft_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(draft_path, encoding="unicode", xml_declaration=False)

    validation = validate_configuration_file(
        draft_path, repo_root=repo_root, config_path=config_root
    )

    result: dict[str, Any] = {
        "pipeline": key,
        "based_on": example,
        "draft": str(draft_path),
        "applied": applied,
        "not_applied": not_applied,
        "valid": validation.get("valid", False),
    }
    if validation.get("valid"):
        result["top_level_sections"] = validation.get("top_level_sections")
    else:
        result["error"] = validation.get("error")
    return result


def _inline_includes(path: Path, config_root: Path) -> ET.Element:
    """Return the parsed file with every ``<config>`` include spliced in."""
    element = ET.parse(path).getroot()
    _splice(element, path.parent, config_root)
    return element


def _splice(element: ET.Element, base_dir: Path, config_root: Path) -> None:
    for child in list(element):
        if child.tag == "config":
            ref = (child.text or "").strip().replace(_CONFIG_KEYWORD, str(config_root))
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                ref_path = base_dir / ref_path
            included = ET.parse(ref_path).getroot()
            _splice(included, ref_path.parent, config_root)
            position = list(element).index(child)
            element.remove(child)
            for offset, node in enumerate(list(included)):
                element.insert(position + offset, node)
        else:
            _splice(child, base_dir, config_root)


_PATH_KEYS = {
    "configpath",
    "datafile",
    "path",
    "filename",
    "file_name",
    "dp_map",
    "limb_darkening_coefficients",
    "url",
}
_PATH_SUFFIXES = ("_path", "_file", "_filepath", "_files", "_dir", "_filename")


def _is_path_key(tag: str) -> bool:
    tag = tag.lower()
    return tag in _PATH_KEYS or tag.endswith(_PATH_SUFFIXES)


def _is_executable_pointer(tag: str, value: str) -> bool:
    """True if the value makes ExoSim import and run user code.

    ``klass_factory.find_task`` executes a ``*_task`` / ``task_model`` entry only
    when it is a path to a ``.py`` file; any other string is looked up by class
    name among the subclasses under ``exosim.tasks`` (built-in code), so a plain
    class name such as ``LoadOpticalElement`` is safe.
    """
    return value.strip().lower().endswith(".py")


def _check_override(
    tag: str,
    value: str,
    *,
    sandbox: Sandbox,
    config_root: Path,
) -> None:
    """Refuse overrides that would escape the sandbox or run arbitrary code.

    * A value pointing at a ``.py`` file (``*_task``, ``task_model``) is code that
      ExoSim will execute: refused when execution is disabled, and otherwise only
      accepted from a readable location.
    * ``ConfigPath``, ``datafile`` and other path-like keys must resolve inside
      the directories the server may read.
    """
    text = value.strip()
    if _is_executable_pointer(tag, text):
        if not sandbox.allow_exec:
            raise ValueError(
                f"Override of '{tag}' points a task at a Python file ({text}); "
                "this is arbitrary code execution and is refused while "
                "execution is disabled (--no-exec)."
            )
    elif not _is_path_key(tag):
        return
    if "://" in text:
        scheme, _, rest = text.partition("://")
        if scheme.lower() != "file":
            raise ValueError(
                f"Override of '{tag}' is a remote URL ({text}); remote resources "
                "are not accepted in overrides, use a file in a readable directory."
            )
        text = rest
    expanded = text.replace(_CONFIG_KEYWORD, str(config_root))
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = config_root / candidate
    # Raises PathNotAllowedError (a ValueError) outside the readable roots.
    sandbox.resolve_read(candidate)


def _apply_overrides(
    root: ET.Element,
    overrides: dict[str, str],
    *,
    sandbox: Sandbox,
    config_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    applied: list[dict[str, str]] = []
    not_applied: list[dict[str, Any]] = []

    for path, raw in overrides.items():
        node, _walked, failure = _walk(root, [s for s in str(path).split(".") if s])
        if failure is not None:
            failure["path"] = path
            not_applied.append(failure)
            continue

        _check_override(node.tag, str(raw), sandbox=sandbox, config_root=config_root)
        value, unit = _split_unit(str(raw))
        node.text = value
        if unit is not None:
            node.set("unit", unit)
        applied.append({"path": path, "value": str(raw)})

    return applied, not_applied


def _walk(
    node: ET.Element, segments: list[str]
) -> tuple[ET.Element, list[str], dict[str, Any] | None]:
    """Descend ``node`` by ``segments``, disambiguating repeated tags by text.

    Repeated tags (e.g. two ``<channel>`` elements, or several ``<source>``)
    are selected by the next segment, which is matched against the element's
    text — so ``payload.channel.Photometer.wl_min`` picks the ``Photometer``
    channel.
    """
    walked: list[str] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        by_tag = [child for child in node if child.tag == segment]
        if len(by_tag) == 1:
            node, index = by_tag[0], index + 1
            walked.append(segment)
            continue
        if len(by_tag) > 1:
            names = [(child.text or "").strip() for child in by_tag]
            if index + 1 < len(segments):
                wanted = segments[index + 1]
                pick = next(
                    (c for c, n in zip(by_tag, names, strict=True) if n == wanted),
                    None,
                )
                if pick is not None:
                    node = pick
                    walked += [segment, segments[index + 1]]
                    index += 2
                    continue
            return (
                node,
                walked,
                {
                    "reason": f"'{segment}' is repeated; name the one you mean",
                    "available": names,
                },
            )
        by_text = next(
            (child for child in node if (child.text or "").strip() == segment), None
        )
        if by_text is not None:
            node, index = by_text, index + 1
            walked.append(segment)
            continue
        return (
            node,
            walked,
            {
                "reason": f"no '{segment}' under {'.'.join(walked) or '<root>'}",
                "available": sorted({child.tag for child in node}),
            },
        )
    return node, walked, None


def _split_unit(value: str) -> tuple[str, str | None]:
    parts = value.split()
    if len(parts) == 2:
        number, unit = parts
        try:
            float(number)
        except ValueError:
            return value, None
        return number, unit
    return value, None
