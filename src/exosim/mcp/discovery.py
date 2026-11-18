"""Dynamic discovery utilities for ExoSim MCP tools.

This module provides utilities to introspect the ExoSim codebase and
dynamically discover available recipes, their parameters, and metadata.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from exosim.mcp.schemas import PipelineDefinition

# Import failures met while discovering recipes. A missing optional module (for
# example ``distutils`` for ``batman`` on Python 3.12 without setuptools) must not
# vanish silently: ``capabilities()`` and ``list_pipelines`` report these.
_DISCOVERY_ERRORS: list[dict[str, str | None]] = []

# Recipes that are discovered but deliberately not exposed. ``SimulateObservation``
# chains the other recipes through hard-coded ``./test_common.h5`` / ``./test_se.h5``
# files in the *current directory* (which the MCP runner sets to the configuration's
# directory: the checkout or the read-only packaged examples, shared by concurrent
# jobs) and deletes them afterwards; it also has no ``plot`` parameter. It is not
# safe to run through the sandboxed server until the recipe takes a working
# directory. Run focalplane -> subexposures -> ndrs instead.
_HIDDEN_RECIPES = frozenset({"SimulateObservation"})

_HINTS = {
    "distutils": (
        "distutils was removed from the standard library in Python 3.12 and "
        "batman needs it: pip install 'setuptools>=64'"
    ),
}


def record_discovery_error(exc: BaseException, module: str) -> dict[str, str | None]:
    """Remember an import failure and return its structured description.

    Parameters
    ----------
    exc : BaseException
        The (typically ``ImportError``) exception that was caught.
    module : str
        The module whose import was attempted (e.g. ``exosim.recipes``).

    Returns
    -------
    dict
        ``module``, ``missing_module``, ``error`` and ``hint`` entries.
    """
    missing = getattr(exc, "name", None)
    root = (missing or "").split(".")[0]
    entry: dict[str, str | None] = {
        "module": module,
        "missing_module": missing,
        "error": f"{type(exc).__name__}: {exc}",
        "hint": _HINTS.get(root)
        or (f"Install the missing module: pip install {root}" if root else None),
    }
    if entry not in _DISCOVERY_ERRORS:
        _DISCOVERY_ERRORS.append(entry)
    return entry


def discovery_errors() -> list[dict[str, str | None]]:
    """Return the import errors recorded so far (a copy)."""
    return [dict(entry) for entry in _DISCOVERY_ERRORS]


def clear_discovery_errors() -> None:
    """Forget the recorded errors (used by tests)."""
    _DISCOVERY_ERRORS.clear()


def discover_recipes(
    module_name: str = "exosim.recipes",
) -> dict[str, PipelineDefinition]:
    """Discover all available ExoSim recipes dynamically.

    Parameters
    ----------
    module_name : str
        Name of the module to inspect for recipes.

    Returns
    -------
    dict[str, PipelineDefinition]
        Dictionary mapping pipeline names to their definitions.
    """
    try:
        recipes_module = importlib.import_module(module_name)
    except ImportError as exc:
        record_discovery_error(exc, module_name)
        return {}

    pipelines = {}

    for name, obj in inspect.getmembers(recipes_module, inspect.isclass):
        # Check if this is a recipe class (defined within the recipes module)
        # Accept both direct module (e.g., "exosim.recipes") and submodules
        # (e.g., "exosim.recipes.create_focal_plane")
        if not obj.__module__.startswith(module_name):
            continue

        # Skip base classes and utility classes
        if name.startswith("_") or not hasattr(obj, "__init__"):
            continue
        if name in _HIDDEN_RECIPES:
            continue

        # Extract metadata from the class
        pipeline_def = _extract_pipeline_metadata(name, obj)
        if pipeline_def:
            pipelines[pipeline_def.name] = pipeline_def

    return pipelines


def _extract_pipeline_metadata(
    class_name: str, recipe_class: type
) -> PipelineDefinition | None:
    """Extract pipeline metadata from a recipe class.

    Parameters
    ----------
    class_name : str
        Name of the recipe class.
    recipe_class : type
        The recipe class to extract metadata from.

    Returns
    -------
    PipelineDefinition or None
        Pipeline definition if extraction succeeds, None otherwise.
    """
    # Get the __init__ signature to determine required inputs
    try:
        sig = inspect.signature(recipe_class.__init__)
        params = sig.parameters
    except Exception:
        return None

    # Derive CLI name from class name (e.g., CreateFocalPlane -> focalplane)
    # Remove common prefixes like "Create" and "Simulate"
    cli_name = class_name
    for prefix in ["Create", "Simulate", "Generate", "Build"]:
        if cli_name.startswith(prefix):
            cli_name = cli_name[len(prefix) :]
            break
    # Remove common suffixes like "Model", "Recipe", "Pipeline"
    for suffix in ["Model", "Recipe", "Pipeline"]:
        if cli_name.endswith(suffix):
            cli_name = cli_name[: -len(suffix)]
            break
    cli_name = _camel_to_snake(cli_name).replace("_", "")

    # Check if there's explicit MCP metadata in the class (optional, for overrides)
    mcp_metadata = getattr(recipe_class, "__mcp_metadata__", None)

    if mcp_metadata:
        # Use explicit metadata if available (optional override)
        return PipelineDefinition(
            name=mcp_metadata.get("name", cli_name),
            recipe=f"exosim.recipes.{class_name}",
            cli_command=mcp_metadata.get("cli_command", f"exosim {cli_name}"),
            required_inputs=tuple(
                mcp_metadata.get("required_inputs", ["configuration"])
            ),
            output_description=mcp_metadata.get(
                "output_description", "Pipeline output."
            ),
        )

    # Extract information automatically from code
    required_inputs = []
    param_names = [p for p in params if p != "self"]

    # Detect configuration parameter
    if "options_file" in param_names or "configuration" in param_names:
        required_inputs.append("configuration")

    # Detect input file parameter (for downstream pipelines)
    if "input_file" in param_names or "input" in param_names:
        required_inputs.append("input")

    # Detect output parameter
    if "output_file" in param_names or "output" in param_names:
        required_inputs.append("output")

    # Extract description from docstring intelligently
    output_desc = _extract_output_description(recipe_class)

    return PipelineDefinition(
        name=cli_name,
        recipe=f"exosim.recipes.{class_name}",
        cli_command=f"exosim {cli_name}",
        required_inputs=tuple(required_inputs)
        if required_inputs
        else ("configuration", "output"),
        output_description=output_desc,
    )


def _extract_output_description(recipe_class: type) -> str:
    """Extract output description from recipe docstring.

    Parameters
    ----------
    recipe_class : type
        The recipe class to extract from.

    Returns
    -------
    str
        Description of the pipeline output.
    """
    doc = inspect.getdoc(recipe_class)
    if not doc:
        return "Pipeline output."

    lines = doc.split("\n")

    # Look for class-level description (first paragraph)
    first_paragraph = []
    for line in lines:
        line = line.strip()
        if not line:
            if first_paragraph:
                break
            continue
        first_paragraph.append(line)

    # Check if first paragraph mentions output or product
    description = " ".join(first_paragraph)
    if "output" in description.lower() or "produce" in description.lower():
        # Extract the relevant sentence
        sentences = description.split(".")
        for sentence in sentences:
            if "output" in sentence.lower() or "produce" in sentence.lower():
                return sentence.strip() + "."

    # Fallback: use class name to generate description
    class_name = recipe_class.__name__
    if "FocalPlane" in class_name:
        return "HDF5 focal plane output with detector layout and PSF."
    if "Radiometric" in class_name:
        return "HDF5 output with radiometric model and ECSV tables."
    if "SubExposure" in class_name:
        return "HDF5 sub-exposure output with pointing jitter."
    if "NDR" in class_name:
        return "HDF5 non-destructive reads (NDRs) output."
    if "Observation" in class_name:
        return "Complete observation simulation output."

    return "Pipeline output file (HDF5)."


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case.

    Parameters
    ----------
    name : str
        CamelCase string.

    Returns
    -------
    str
        snake_case string.
    """
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def get_recipe_instance_params(recipe_class: type) -> dict[str, Any]:
    """Get the parameters required to instantiate a recipe.

    Parameters
    ----------
    recipe_class : type
        The recipe class to inspect.

    Returns
    -------
    dict[str, Any]
        Dictionary of parameter names and their default values.
    """
    try:
        sig = inspect.signature(recipe_class.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            params[name] = {
                "required": param.default is inspect.Parameter.empty,
                "default": None
                if param.default is inspect.Parameter.empty
                else param.default,
                "annotation": (
                    str(param.annotation)
                    if param.annotation is not inspect.Parameter.empty
                    else None
                ),
            }
        return params
    except Exception:
        return {}
