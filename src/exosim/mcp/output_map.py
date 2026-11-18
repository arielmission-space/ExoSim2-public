"""A built-in map from ExoSim output node names to their physical meaning.

``inspect_output`` reports the *structure* of an HDF5 / ECSV output. This module
adds the *science*: for every well-known group or dataset name ExoSim writes,
what the numbers mean, which pipeline stage produced them, and the doc page that
explains them. ``explain_output`` uses it so an AI client does not have to guess
from retrieval alone.

The map is keyed by the leaf node name (the last ``/``-separated component).
Names that depend on the configuration (channel, source, target and table
column names) cannot be keys, so :func:`describe_node` also matches structural
patterns on the whole path (children of ``channels``, ``sources``, ``*_to_group``
tables, serialised-signal fields, ``info``, ``metadata``), and
:func:`describe_items` recognises groups by their children.
"""

from __future__ import annotations

import re
from typing import Any

# leaf name -> (meaning, producing stage, doc page)
_NODE_MEANING: dict[str, tuple[str, str, str]] = {
    # --- focal plane pipeline -------------------------------------------------
    "focal_plane": (
        "The target star focal plane: a (time, y, x) cube of counts/s per "
        "pixel, oversampled if the detector defines an oversampling factor.",
        "focalplane",
        "docs/source/user/focal_plane/resulting_focal_plane.rst",
    ),
    "frg_focal_plane": (
        "The foreground (zodi, instrument emission, sky) focal plane, same "
        "shape as ``focal_plane``; kept separate so the astronomical signal "
        "can be applied to the star alone.",
        "focalplane",
        "docs/source/user/focal_plane/foregrounds.rst",
    ),
    "bkg_focal_plane": (
        "The field-of-view background stars focal plane (other sources than "
        "the target), same shape as ``focal_plane``.",
        "focalplane",
        "docs/source/user/focal_plane/sky_sources.rst",
    ),
    "frg_sub_focal_planes": (
        "Per-contribution foreground focal planes, one dataset per optical "
        "element, summed into ``frg_focal_plane``.",
        "focalplane",
        "docs/source/user/focal_plane/foregrounds.rst",
    ),
    "responsivity": (
        "Detector responsivity vs wavelength (counts per photon, including "
        "quantum efficiency), used to convert the incident SED to counts/s.",
        "focalplane",
        "docs/source/user/focal_plane/channel.rst",
    ),
    "efficiency": (
        "End-to-end optical efficiency (throughput) vs wavelength for the "
        "optical path or channel.",
        "focalplane",
        "docs/source/user/focal_plane/optical_paths.rst",
    ),
    "psf": (
        "The point spread function cube used to distribute a source's flux "
        "over the focal plane.",
        "focalplane",
        "docs/source/user/focal_plane/focal_plane_array.rst",
    ),
    "sky": (
        "The parsed sky model: sources and foregrounds as produced before "
        "propagation through the instrument.",
        "focalplane",
        "docs/source/user/focal_plane/sky_sources.rst",
    ),
    "sources": (
        "Per-source spectral energy distributions (``Sed``) on the simulation "
        "wavelength/time grid.",
        "focalplane",
        "docs/source/user/focal_plane/sky_sources.rst",
    ),
    "foregrounds": (
        "Radiance of each diffuse foreground contribution vs wavelength.",
        "focalplane",
        "docs/source/user/focal_plane/foregrounds.rst",
    ),
    "telescope": (
        "The common (telescope) optical path: transmission and emission of "
        "the elements shared by every channel.",
        "focalplane",
        "docs/source/user/focal_plane/optical_paths.rst",
    ),
    "final contributes": (
        "The combined optical-path contribution (transmission and emitted "
        "radiance) reaching the detector.",
        "focalplane",
        "docs/source/user/focal_plane/optical_paths.rst",
    ),
    "intrapixel_response_function": (
        "The intra-pixel response function (IRF) applied to the oversampled "
        "focal plane.",
        "focalplane",
        "docs/source/user/focal_plane/focal_plane_array.rst",
    ),
    "radiance": (
        "Radiance emitted by an optical element or foreground, vs wavelength.",
        "focalplane",
        "docs/source/user/focal_plane/optical_paths.rst",
    ),
    "irf": (
        "The intra-pixel response function kernel.",
        "focalplane",
        "docs/source/user/focal_plane/focal_plane_array.rst",
    ),
    # --- sub-exposures pipeline --------------------------------------------
    "SubExposures": (
        "The sub-exposure cube: the focal plane sampled at the reading "
        "cadence, in counts, with pointing jitter and foregrounds applied.",
        "subexposures",
        "docs/source/user/sub-exposures/finalising_sub_exposures.rst",
    ),
    "sub_exposures": (
        "The sub-exposure cube (counts) at the reading cadence, jitter and "
        "foregrounds applied.",
        "subexposures",
        "docs/source/user/sub-exposures/finalising_sub_exposures.rst",
    ),
    "sub_exposures_bkg": (
        "The background-stars contribution to the sub-exposures, added into "
        "the main cube.",
        "subexposures",
        "docs/source/user/sub-exposures/finalising_sub_exposures.rst",
    ),
    "Astrosignal": (
        "The time-varying astronomical signal (e.g. a transit light curve) "
        "per source and signal type, before it is imprinted on the star.",
        "subexposures",
        "docs/source/user/sub-exposures/astronomical_signal.rst",
    ),
    # --- NDRs pipeline ----------------------------------------------------
    "NDRs": (
        "Non-destructive reads: the detector ramp in counts, one frame per "
        "read, after accumulation of the sub-exposures.",
        "ndrs",
        "docs/source/user/ndrs/results.rst",
    ),
    "sub_ndrs": (
        "The individual (pre-merge) non-destructive reads before groups are "
        "averaged together.",
        "ndrs",
        "docs/source/user/ndrs/accumulate.rst",
    ),
    "ADC_ndrs": (
        "The NDRs after the analogue-to-digital conversion, in ADU.",
        "ndrs",
        "docs/source/user/ndrs/analogtodigtital.rst",
    ),
    "cosmic rays": (
        "The cosmic-ray hit map added to the NDRs.",
        "ndrs",
        "docs/source/user/ndrs/cosmic_rays.rst",
    ),
    "shot noise": (
        "The realised photon (shot) noise added to the reads.",
        "ndrs",
        "docs/source/user/ndrs/shot_noise.rst",
    ),
    "read noise": (
        "The realised detector read noise added to the reads.",
        "ndrs",
        "docs/source/user/ndrs/readnoise.rst",
    ),
    "kTC noise": (
        "The realised reset (kTC) noise.",
        "ndrs",
        "docs/source/user/ndrs/reset_bias.rst",
    ),
    "gain noise": (
        "The realised gain-drift contribution.",
        "ndrs",
        "docs/source/user/ndrs/gain_drift.rst",
    ),
    "dead_pixel_map": (
        "The dead / hot pixel map applied to the detector.",
        "ndrs",
        "docs/source/user/ndrs/dead_pixels.rst",
    ),
    # --- radiometric pipeline -------------------------------------------
    "radiometric": (
        "The radiometric model: per wavelength bin, the signal and noise "
        "budget (photon, dark, read, ...) and the resulting SNR.",
        "radiometric",
        "docs/source/user/radiometric_model/estimate_signal.rst",
    ),
    "radiometric_apertures": (
        "The extraction apertures used by the radiometric model: size and "
        "pixel count per wavelength bin.",
        "radiometric",
        "docs/source/user/radiometric_model/estimate_apertures.rst",
    ),
    "targets": (
        "Per-target results when the radiometric pipeline runs over a target list.",
        "radiometric",
        "docs/source/user/radiometric_model/operating_modes.rst",
    ),
    # --- shared ---------------------------------------------------------
    "channels": (
        "Per-channel results; each child group is one instrument channel "
        "(photometer or spectrometer).",
        "focalplane",
        "docs/source/user/focal_plane/channel.rst",
    ),
    "simulation_data": (
        "Auxiliary per-channel data kept for later stages (QE maps, timing, "
        "reading scheme).",
        "subexposures",
        "docs/source/user/sub-exposures/reading_scheme.rst",
    ),
    "configuration": (
        "A verbatim copy of the input configuration used for this run "
        "(present only when the recipe is asked to store it).",
        "focalplane",
        "docs/source/user/quickstart.rst",
    ),
    "info": (
        "Run metadata: ExoSim version, timing and provenance.",
        "focalplane",
        "docs/source/user/quickstart.rst",
    ),
}


# The fields of a serialised ``Signal`` / ``Sed`` (``Signal.write`` stores the
# object as a dictionary): the same names recur under every channel, source and
# optical element, so they are matched by leaf name below any group.
_SIGNAL_FIELDS: dict[str, str] = {
    "data": "The data array of this signal (time, spatial, spectral axes).",
    "time": "The time grid of this signal.",
    "spectral": "The spectral (wavelength) grid of this signal.",
    "spatial": "The spatial grid of this signal.",
    "data_units": "Unit string of the ``data`` array.",
    "time_units": "Unit string of the ``time`` grid.",
    "spectral_units": "Unit string of the ``spectral`` grid.",
    "spatial_units": "Unit string of the ``spatial`` grid.",
    "cached": "Whether the data are kept in a cache file instead of this dataset.",
    "datatype": "Python class this signal was serialised from.",
    "metadata": "Free-form metadata attached to this signal (name, provenance).",
}

_TABLE_SEE = "docs/source/user/radiometric_model/estimate_signal.rst"
_INFO_SEE = "docs/source/user/quickstart.rst"
_TABLE_SUFFIX = ".__table_column_meta__"
_INDEXED = re.compile(r"^(.+?)_?(\d+)$")


def _inherited_stage(parts: list[str]) -> tuple[str, str]:
    """Stage and doc page of the nearest recognised ancestor."""
    for ancestor in reversed(parts[:-1]):
        entry = _NODE_MEANING.get(ancestor)
        if entry is not None:
            return entry[1], entry[2]
    if parts and parts[0].startswith("radiometric"):
        return "radiometric", _TABLE_SEE
    return "focalplane", "docs/source/user/focal_plane/resulting_focal_plane.rst"


def _describe_structural(parts: list[str]) -> tuple[str, str, str] | None:
    """Meaning of a node whose *name* varies (channel, target, column, ...).

    Real names depend on the configuration (channel names, source names,
    table columns), so they cannot be keys of ``_NODE_MEANING``; the position
    in the tree and the fixed vocabulary around them identify the node.
    """
    leaf = parts[-1]
    parent = parts[-2] if len(parts) > 1 else ""
    stage, see = _inherited_stage(parts)

    if parts[0] == "info" and len(parts) > 1:
        return (
            f"Run metadata entry ``{leaf}`` (ExoSim version, timing, "
            "RunConfig settings and provenance).",
            "focalplane",
            _INFO_SEE,
        )
    if "metadata" in parts[:-1]:
        return (
            f"Metadata entry ``{leaf}`` of the enclosing signal (grid and "
            "parsing parameters it was built from).",
            stage,
            see,
        )
    if parent == "psf":
        return (
            f"Component ``{leaf}`` of the point spread function (PSF) used "
            "to populate the focal plane.",
            "focalplane",
            "docs/source/user/focal_plane/focal_plane_array.rst",
        )
    if leaf == "path" and parent:
        return (
            "The optical path of this channel: one group per optical element "
            "plus the combined ``final contributes``.",
            "focalplane",
            "docs/source/user/focal_plane/optical_paths.rst",
        )
    indexed = _INDEXED.match(leaf)
    if indexed is not None and indexed.group(1) in _NODE_MEANING:
        # a list written by ``write_array`` is stored as ``name0``, ``name_1``...
        meaning, base_stage, base_see = _NODE_MEANING[indexed.group(1)]
        return (f"Entry {indexed.group(2)} of: {meaning}", base_stage, base_see)
    if leaf.endswith(_TABLE_SUFFIX):
        return (
            f"YAML column metadata (units, descriptions) of the table "
            f"``{leaf[: -len(_TABLE_SUFFIX)]}``.",
            stage,
            see,
        )
    for index, part in enumerate(parts):
        if not part.endswith("_to_group"):
            continue
        table = part[: -len("_to_group")]
        depth = len(parts) - 1 - index
        if depth == 0:
            return (
                f"Column-by-column copy of the table ``{table}`` (one group per "
                "column, with its ``value`` and ``unit``).",
                stage,
                see,
            )
        if depth == 1:
            return (f"Column ``{leaf}`` of the table ``{table}``.", stage, see)
        if leaf == "value":
            return (f"Values of column ``{parent}`` of ``{table}``.", stage, see)
        if leaf == "unit":
            return (f"Unit of column ``{parent}`` of ``{table}``.", stage, see)
        return None
    if leaf == "table":
        return (
            "A data table: one row per spectral bin (or per aperture), one "
            "column per quantity.",
            stage,
            see,
        )
    if parent == "channels":
        return (
            f"Instrument channel ``{leaf}``: its focal planes, optical "
            "elements, responsivity and sources.",
            "focalplane",
            "docs/source/user/focal_plane/channel.rst",
        )
    if parent == "sources":
        return (
            f"Spectral energy distribution of the source ``{leaf}``.",
            "focalplane",
            "docs/source/user/focal_plane/sky_sources.rst",
        )
    if parent in {"radiometric", "targets"}:
        return (
            f"Results for the target ``{leaf}``.",
            "radiometric",
            "docs/source/user/radiometric_model/operating_modes.rst",
        )
    if leaf in _SIGNAL_FIELDS and parent:
        return (_SIGNAL_FIELDS[leaf], stage, see)
    return None


def describe_node(name: str) -> dict[str, Any] | None:
    """Return the meaning of one output node, or ``None`` if it is not known.

    A leaf name in the built-in map matches anywhere. Otherwise the position
    of the node in a full ``/``-separated path is used to recognise
    configuration-dependent names (channels, targets, table columns) and the
    fixed fields of a serialised signal.

    Parameters
    ----------
    name : str
        Either a leaf node name or a full ``/``-separated HDF5 path.

    Returns
    -------
    dict or None
        ``{"node", "meaning", "stage", "see"}`` when recognised.
    """
    parts = [p for p in name.split("/") if p]
    if not parts:
        return None
    leaf = parts[-1]
    entry = _NODE_MEANING.get(leaf) or _NODE_MEANING.get(name)
    if entry is None:
        entry = _describe_structural(parts)
    if entry is None:
        return None
    meaning, stage, see = entry
    return {"node": leaf, "meaning": meaning, "stage": stage, "see": see}


def _optical_element(path: str) -> dict[str, Any]:
    leaf = path.rstrip("/").split("/")[-1]
    return {
        "node": leaf,
        "meaning": (
            f"Optical element ``{leaf}``: its emitted radiance and "
            "transmission (efficiency) vs wavelength."
        ),
        "stage": "focalplane",
        "see": "docs/source/user/focal_plane/optical_paths.rst",
    }


def describe_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Describe every recognisable item, using the siblings for context.

    On top of :func:`describe_node` this recognises a group by its children: a
    group holding ``radiance`` / ``efficiency`` is an optical element, whatever
    it is called in the configuration.

    Returns
    -------
    dict
        Item path -> ``{"node", "meaning", "stage", "see"}``.
    """
    children: dict[str, set[str]] = {}
    for item in items:
        path = item.get("path", "")
        parent, _, leaf = path.rpartition("/")
        children.setdefault(parent, set()).add(leaf)

    described: dict[str, dict[str, Any]] = {}
    for item in items:
        path = item.get("path", "")
        found = describe_node(path)
        if (
            found is None
            and item.get("type") == "group"
            and {"radiance", "efficiency"} & children.get(path, set())
        ):
            found = _optical_element(path)
        if found is not None:
            described[path] = found
    return described


def annotate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a ``meaning`` block to every recognised item of an inspection."""
    described = describe_items(items)
    for item in items:
        found = described.get(item.get("path", ""))
        if found is not None:
            item["meaning"] = found["meaning"]
            item["stage"] = found["stage"]
            item["see"] = found["see"]
    return items


def known_nodes() -> list[str]:
    """The output node names this map knows about."""
    return sorted(_NODE_MEANING)
