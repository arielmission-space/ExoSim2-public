"""Render figures from an ExoSim output for the MCP layer.

``inspect_output`` / ``explain_output`` describe an output in words;
``generate_plots`` turns it into PNGs a vision-capable client can actually look
at, by driving the same plotters as the ``exosim-plot`` command. Figures are
written under the sandbox working directory.

This is an execution tool: it is only registered when the server allows
execution, because rendering a large focal-plane cube is not free.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import h5py

from exosim.mcp.sandbox import Sandbox

# output-file layout -> plotter kind
_KIND_MARKERS: dict[str, tuple[str, ...]] = {
    "ndrs": ("NDRs", "ADC_ndrs", "sub_ndrs"),
    "subexposures": ("SubExposures", "Astrosignal"),
    "focalplane": ("focal_plane", "frg_focal_plane"),
}
_VALID_KINDS = (*_KIND_MARKERS, "radiometric")
_VALID_SCALES = {"linear": "linear", "db": "dB"}

# The plotters drive matplotlib's pyplot global state (current figure/axes), and
# the MCP SDK runs sync tools in parallel worker threads: two renders at once
# draw on each other's figures. One render at a time.
_RENDER_LOCK = threading.Lock()


def generate_plots(
    path: str | Path,
    kind: str | None = None,
    *,
    time_step: int = 0,
    scale: str = "linear",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Render figures from an ExoSim output file.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 output to plot.
    kind : str, optional
        ``focalplane``, ``radiometric``, ``subexposures`` or ``ndrs``. Detected
        from the file layout when omitted.
    time_step : int
        Focal-plane time slice to render (``focalplane`` only).
    scale : str
        ``linear`` or ``dB`` (``focalplane`` only).
    repo_root : str or pathlib.Path, optional
        Repository root for relative paths.

    Returns
    -------
    dict
        ``kind``, ``output_dir`` and ``figures`` (the written PNG paths).
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.use("Agg", force=True)

    normalised_scale = _VALID_SCALES.get(str(scale).strip().lower())
    if normalised_scale is None:
        raise ValueError(
            f"Unknown scale {scale!r}: use 'linear' or 'dB' (focalplane plots only)."
        )
    scale = normalised_scale

    sandbox = Sandbox.from_env(repo_root=repo_root)
    output_path = sandbox.resolve_read(path, prefer_work_dir=True)
    if not output_path.exists():
        return {"error": f"Output file not found: {output_path}"}

    detected = kind.lower() if kind else _detect_kind(output_path)
    if detected not in _VALID_KINDS:
        return {
            "error": f"Cannot plot '{detected or 'unknown'}' outputs",
            "supported": list(_VALID_KINDS),
        }

    table_paths: list[str] = []
    if detected == "radiometric":
        table_paths = _radiometric_table_groups(output_path)
        if not table_paths:
            return {
                "kind": "radiometric",
                "plottable": False,
                "message": (
                    "no plottable radiometric table in this file (kind: "
                    "radiometric): expected a 'table' dataset under "
                    "'radiometric/' or 'radiometric_apertures/'. The full "
                    "radiometric table is written next to the HDF5 file as "
                    "radiometric_table.ecsv; use inspect_output on it."
                ),
                "top_level": _top_level_names(output_path),
            }

    # Two outputs may share a stem (a/x.h5, b/x.h5): key the directory on the
    # resolved path as well, so their figures never mix.
    digest = hashlib.sha1(str(output_path).encode("utf-8")).hexdigest()[:8]
    out_dir = sandbox.resolve_write(Path("plots") / f"{output_path.stem}-{digest}")
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    error: str | None = None
    with _RENDER_LOCK:
        before = _snapshot(out_dir)
        try:
            _render(
                detected,
                str(output_path),
                out_dir,
                time_step=time_step,
                scale=scale,
                table_paths=table_paths,
                warnings=warnings,
            )
        except Exception as exc:  # report, do not crash the server
            error = f"{type(exc).__name__}: {exc}"
        # Only what this call wrote: earlier calls' PNGs may be stale.
        figures = sorted(
            str(p) for p, stamp in _snapshot(out_dir).items() if before.get(p) != stamp
        )

    result: dict[str, Any] = {
        "kind": detected,
        "output_dir": str(out_dir),
        "figures": figures,
    }
    if error:
        result["error"] = error
    if warnings:
        result["warnings"] = warnings
    return result


def _snapshot(directory: Path) -> dict[Path, tuple[int, int]]:
    """Modification stamp of every PNG in ``directory``."""
    stamps: dict[Path, tuple[int, int]] = {}
    for png in directory.glob("*.png"):
        try:
            info = png.stat()
        except OSError:
            continue
        stamps[png] = (info.st_mtime_ns, info.st_size)
    return stamps


def _radiometric_table_groups(path: Path) -> list[str]:
    """The groups holding a radiometric ``table`` dataset, in plotting order.

    The plotter's default is ``radiometric``, but a target-list run stores one
    table per target (``radiometric/<target>/table``) and the aperture table
    under ``radiometric_apertures``.
    """
    try:
        with h5py.File(path, "r") as handle:
            top = handle.get("radiometric")
            if isinstance(top, h5py.Group):
                if "table" in top:
                    return ["radiometric"]
                targets = [
                    f"radiometric/{key}"
                    for key in top
                    if isinstance(top.get(key), h5py.Group) and "table" in top[key]
                ]
                if targets:
                    return targets
            apertures = handle.get("radiometric_apertures")
            if isinstance(apertures, h5py.Group) and "table" in apertures:
                return ["radiometric_apertures"]
    except OSError:
        return []
    return []


def _top_level_names(path: Path) -> list[str]:
    try:
        with h5py.File(path, "r") as handle:
            return sorted(handle.keys())
    except OSError:
        return []


def _detect_kind(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as handle:
            names: set[str] = set()
            handle.visit(lambda name: names.add(name.rsplit("/", 1)[-1]))
    except OSError:
        return None
    if "radiometric" in names or "radiometric_apertures" in names:
        return "radiometric"
    for candidate, markers in _KIND_MARKERS.items():
        if names.intersection(markers):
            return candidate
    return None


def _render(
    kind: str,
    input_file: str,
    out_dir: Path,
    *,
    time_step: int,
    scale: str,
    table_paths: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    if kind == "focalplane":
        from exosim.plots.focal_plane_plotter import FocalPlanePlotter

        plotter = FocalPlanePlotter(input=input_file)
        plotter.plot_focal_plane(time_step=time_step, scale=scale)
        plotter.save_fig(str(out_dir / f"focal_plane_{time_step}.png"))
        plotter.plot_efficiency()
        plotter.save_fig(str(out_dir / "efficiency.png"))
        return

    if kind == "radiometric":
        from exosim.plots.radiometric_plotter import RadiometricPlotter

        groups = table_paths or ["radiometric"]
        plotter = None
        for group in groups:
            plotter = RadiometricPlotter(input=input_file, table_path=group)
            plotter.plot_table()
            # one figure per target in a target-list output
            suffix = "" if len(groups) == 1 else "_" + group.rsplit("/", 1)[-1]
            plotter.save_fig(
                str(out_dir / f"radiometric{suffix}.png".replace(" ", "_"))
            )
        # The aperture figure needs per-channel focal planes in the same file,
        # which a table-only or target-list output does not have.
        try:
            plotter.plot_apertures()
            plotter.save_fig(str(out_dir / "apertures.png"))
        except Exception as exc:
            if warnings is not None:
                warnings.append(
                    f"apertures figure skipped ({type(exc).__name__}: {exc}); "
                    "it needs channels/<name>/focal_plane in the same file"
                )
        return

    if kind == "subexposures":
        from exosim.plots.sub_exposures_plotter import SubExposuresPlotter

        SubExposuresPlotter(input=input_file).plot(str(out_dir))
        return

    from exosim.plots.ndrs_plotter import NDRsPlotter

    NDRsPlotter(input=input_file).plot(str(out_dir))
