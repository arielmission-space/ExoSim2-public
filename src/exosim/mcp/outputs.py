"""Output inspection helpers for the ExoSim MCP layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
from astropy.table import Table

from exosim.mcp.output_map import annotate_items, describe_items, known_nodes
from exosim.mcp.schemas import resolve_repo_path

if TYPE_CHECKING:
    from pathlib import Path


def inspect_output_file(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    max_items: int = 80,
) -> dict[str, Any]:
    """Inspect an ExoSim output file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file to inspect.
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.
    max_items : int
        Maximum number of HDF5 nodes to report.

    Returns
    -------
    dict
        Structural summary of the output file.
    """
    output_path = resolve_repo_path(path, repo_root, prefer_work_dir=True)
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    suffix = output_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _inspect_hdf5(output_path, max_items=max_items)
    if suffix in {".ecsv", ".csv"}:
        return _inspect_table(output_path)
    raise ValueError(f"Unsupported output format: {suffix}")


def explain_output_file(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    max_items: int = 200,
) -> dict[str, Any]:
    """Explain an ExoSim output: its structure plus the science of each node.

    Every recognised group / dataset is annotated with what the numbers mean,
    which pipeline stage produced them and the doc page that explains them,
    using the built-in :mod:`exosim.mcp.output_map`. Unrecognised nodes are
    listed separately so the caller knows what was left unexplained.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file to explain.
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.
    max_items : int
        Maximum number of HDF5 nodes to walk.

    Returns
    -------
    dict
        ``explained`` (nodes with a meaning), ``unexplained`` (node paths with
        no entry in the map), and ``pipeline`` (best guess at the producing
        pipeline).
    """
    summary = inspect_output_file(path, repo_root=repo_root, max_items=max_items)

    if summary.get("format") != "hdf5":
        columns = summary.get("columns", [])
        return {
            "path": summary["path"],
            "format": summary.get("format"),
            "note": (
                "Tabular output. Columns are radiometric quantities per "
                "wavelength bin; see the radiometric model documentation."
            ),
            "columns": columns,
            "see": "docs/source/user/radiometric_model/index.rst",
        }

    descriptions = describe_items(summary["items"])
    explained: list[dict[str, Any]] = []
    unexplained: list[str] = []
    stages: set[str] = set()
    for item in summary["items"]:
        described = descriptions.get(item["path"])
        if described is None:
            unexplained.append(item["path"])
            continue
        stages.add(described["stage"])
        explained.append(
            {
                "path": item["path"],
                "type": item["type"],
                "shape": item.get("shape"),
                **described,
            }
        )

    pipeline = _guess_pipeline(stages, summary["items"])
    return {
        "path": summary["path"],
        "format": "hdf5",
        "pipeline": pipeline,
        "explained": explained,
        "unexplained": unexplained,
        # Make the cap visible: how much of the file this answer covers.
        "explained_count": len(explained),
        "unexplained_count": len(unexplained),
        "walked": len(summary["items"]),
        "total_nodes": summary["total_nodes"],
        "truncated": summary["truncated"],
        "known_nodes": known_nodes(),
    }


_STAGE_PRIORITY = ("ndrs", "subexposures", "radiometric", "focalplane")


def _guess_pipeline(stages: set[str], items: list[dict[str, Any]]) -> str | None:
    leaves = {item["path"].rstrip("/").split("/")[-1] for item in items}
    if {"NDRs", "sub_ndrs", "ADC_ndrs"} & leaves:
        return "ndrs"
    if {"SubExposures", "sub_exposures", "Astrosignal"} & leaves:
        return "subexposures"
    if {"radiometric", "radiometric_apertures", "targets"} & leaves:
        return "radiometric"
    if {"focal_plane", "frg_focal_plane"} & leaves:
        return "focalplane"
    for stage in _STAGE_PRIORITY:
        if stage in stages:
            return stage
    return None


def _inspect_hdf5(path: Path, *, max_items: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    def describe(name: str, obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
        item: dict[str, Any] = {
            "path": name,
            "type": "group" if isinstance(obj, h5py.Group) else "dataset",
        }
        if isinstance(obj, h5py.Dataset):
            item["shape"] = obj.shape
            item["dtype"] = str(obj.dtype)
        if obj.attrs:
            item["attrs"] = {
                key: _normalise_attr(value) for key, value in obj.attrs.items()
            }
        return item

    with h5py.File(path, "r") as handle:
        total = 0

        def count(_name: str) -> None:
            nonlocal total
            total += 1

        handle.visit(count)
        root_attrs = {
            key: _normalise_attr(value) for key, value in handle.attrs.items()
        }
        # Breadth first, so a small cap still reaches every top-level group
        # instead of being spent inside the first deep subtree.
        frontier: list[tuple[str, h5py.Group]] = [("", handle)]
        while frontier and len(items) < max_items:
            next_frontier: list[tuple[str, h5py.Group]] = []
            for prefix, group in frontier:
                for key in group:
                    if len(items) >= max_items:
                        break
                    obj = group.get(key)
                    if obj is None:  # dangling link
                        continue
                    name = f"{prefix}/{key}" if prefix else key
                    items.append(describe(name, obj))
                    if isinstance(obj, h5py.Group):
                        next_frontier.append((name, obj))
            frontier = next_frontier

    return {
        "path": str(path),
        "format": "hdf5",
        "root_attrs": root_attrs,
        "items": annotate_items(items),
        "total_nodes": total,
        "truncated": total > len(items),
    }


def _inspect_table(path: Path) -> dict[str, Any]:
    try:
        table = Table.read(path)
    except Exception as exc:  # astropy raises many types for malformed tables
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return {
        "path": str(path),
        "format": path.suffix.lower().lstrip("."),
        "columns": list(table.colnames),
        "row_count": len(table),
        "metadata": dict(table.meta),
    }


def _normalise_attr(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
