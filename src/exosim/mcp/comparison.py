"""Configuration comparison utilities for the ExoSim MCP layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.table import Table

from exosim.mcp.configuration import CONFIG_LOAD_LOCK, config_root
from exosim.mcp.schemas import resolve_repo_path
from exosim.tasks.load.load_options import LoadOptions

if TYPE_CHECKING:
    from pathlib import Path


def compare_configurations(
    config1: str | Path,
    config2: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two ExoSim configuration files.

    Parameters
    ----------
    config1 : str or pathlib.Path
        First configuration file.
    config2 : str or pathlib.Path
        Second configuration file.
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.

    Returns
    -------
    dict
        Comparison result with differences categorized by type.
    """
    config1_path = resolve_repo_path(config1, repo_root, prefer_work_dir=True)
    config2_path = resolve_repo_path(config2, repo_root, prefer_work_dir=True)

    # Load configurations
    try:
        with CONFIG_LOAD_LOCK:
            options1 = LoadOptions()(
                filename=str(config1_path),
                config_path=str(config_root(config1_path)),
            )
    except Exception as exc:
        return {
            "valid": False,
            "error": f"Failed to load {config1_path}: {type(exc).__name__}: {exc}",
        }

    try:
        with CONFIG_LOAD_LOCK:
            options2 = LoadOptions()(
                filename=str(config2_path),
                config_path=str(config_root(config2_path)),
            )
    except Exception as exc:
        return {
            "valid": False,
            "error": f"Failed to load {config2_path}: {type(exc).__name__}: {exc}",
        }

    # Compare structures
    comparison = {
        "valid": True,
        "config1": str(config1_path),
        "config2": str(config2_path),
        "only_in_config1": [],
        "only_in_config2": [],
        "different_values": [],
        "identical_sections": [],
    }

    all_keys = set(options1.keys()) | set(options2.keys())

    for key in sorted(all_keys):
        if key not in options1:
            comparison["only_in_config2"].append(key)
        elif key not in options2:
            comparison["only_in_config1"].append(key)
        else:
            # Both have the key, compare values
            if _values_differ(options1[key], options2[key]):
                comparison["different_values"].append(
                    {
                        "section": key,
                        "type1": type(options1[key]).__name__,
                        "type2": type(options2[key]).__name__,
                        "summary": _summarize_difference(
                            key, options1[key], options2[key]
                        ),
                    }
                )
            else:
                comparison["identical_sections"].append(key)

    return comparison


def _values_differ(val1: Any, val2: Any, tolerance: float = 1e-9) -> bool:
    """Check if two values differ, with tolerance for numerical comparisons."""
    if not isinstance(val1, type(val2)):
        return True

    if isinstance(val1, dict):
        if set(val1.keys()) != set(val2.keys()):
            return True
        return any(_values_differ(val1[k], val2[k]) for k in val1)

    if isinstance(val1, list | tuple):
        if len(val1) != len(val2):
            return True
        return any(_values_differ(v1, v2) for v1, v2 in zip(val1, val2, strict=True))

    if isinstance(val1, Table):  # loaded data files
        if val1.colnames != val2.colnames or len(val1) != len(val2):
            return True
        return any(
            _values_differ(np.asarray(val1[c]), np.asarray(val2[c]), tolerance)
            for c in val1.colnames
        )

    if isinstance(val1, np.ndarray):  # arrays and astropy Quantity arrays
        if val1.shape != val2.shape or getattr(val1, "unit", None) != getattr(
            val2, "unit", None
        ):
            return True
        try:
            return not np.allclose(
                np.asarray(val1, dtype=float),
                np.asarray(val2, dtype=float),
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            )
        except (TypeError, ValueError):
            return not np.array_equal(np.asarray(val1), np.asarray(val2))

    # Numerical comparison with tolerance
    try:
        return abs(float(val1) - float(val2)) > tolerance
    except (TypeError, ValueError):
        # Not numerical, use direct comparison
        return val1 != val2


def _summarize_difference(key: str, val1: Any, val2: Any) -> str:
    """Create a human-readable summary of the difference."""
    if isinstance(val1, dict) and isinstance(val2, dict):
        keys_only_1 = set(val1.keys()) - set(val2.keys())
        keys_only_2 = set(val2.keys()) - set(val1.keys())
        common_keys = set(val1.keys()) & set(val2.keys())
        diff_keys = [k for k in common_keys if _values_differ(val1[k], val2[k])]

        parts = []
        if keys_only_1:
            parts.append(f"{len(keys_only_1)} keys only in config1")
        if keys_only_2:
            parts.append(f"{len(keys_only_2)} keys only in config2")
        if diff_keys:
            parts.append(f"{len(diff_keys)} different values")

        return ", ".join(parts) if parts else "nested structure differs"

    if isinstance(val1, list | tuple) and isinstance(val2, list | tuple):
        return f"length {len(val1)} vs {len(val2)}"

    # For simple values, show both
    str1 = str(val1)[:50]
    str2 = str(val2)[:50]
    return f"'{str1}' vs '{str2}'"
