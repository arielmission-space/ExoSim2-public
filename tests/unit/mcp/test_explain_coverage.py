"""explain_output must generalise over per-channel / per-source / table nodes.

The synthetic files are written with ExoSim's own writers (``Signal.write``,
``write_table``), so their layout is the one a real run produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from exosim.mcp.output_map import describe_node
from exosim.mcp.outputs import explain_output_file
from exosim.models.signal import Signal
from exosim.output.hdf5.hdf5 import HDF5Output

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _tmp_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))


def _signal() -> Signal:
    return Signal(
        spectral=np.linspace(0.1, 1, 10) * u.um,
        data=np.ones((3, 1, 10)),
        time=np.linspace(1, 5, 3) * u.hr,
    )


@pytest.fixture
def focal_plane_file(tmp_path: Path) -> Path:
    path = tmp_path / "focal.h5"
    signal = _signal()
    with HDF5Output(str(path)) as out:
        channels = out.create_group("channels")
        for name in ("Photometer", "Spectrometer"):
            ch = channels.create_group(name)
            signal.write(ch, "focal_plane")
            signal.write(ch, "frg_focal_plane")
            signal.write(ch, "responsivity")
            signal.write(ch.create_group("sources"), "HD 209458")
            element = ch.create_group("M1_mirror")
            signal.write(element, "radiance")
            signal.write(element, "efficiency")
    return path


@pytest.fixture
def radiometric_file(tmp_path: Path) -> Path:
    path = tmp_path / "radio.h5"
    table = QTable({"ch_name": ["a", "a"], "left_bin_edge": [1, 2] * u.um})
    with HDF5Output(str(path)) as out:
        radiometric = out.create_group("radiometric")
        for target in ("GJ 1214", "HD 209458"):
            radiometric.create_group(target).write_table("table", table)
        out.create_group("radiometric_apertures").write_table("table", table)
    return path


def _coverage(result: dict) -> float:
    walked = len(result["explained"]) + len(result["unexplained"])
    return len(result["explained"]) / walked


def test_focal_plane_nodes_are_explained(focal_plane_file: Path) -> None:
    result = explain_output_file(focal_plane_file, max_items=500)
    assert result["pipeline"] == "focalplane"
    assert _coverage(result) >= 0.95, result["unexplained"][:10]
    by_path = {item["path"]: item for item in result["explained"]}
    assert "spectral" in by_path["channels/Photometer/focal_plane/spectral"]["meaning"]
    assert (
        "optical element" in by_path["channels/Photometer/M1_mirror"]["meaning"].lower()
    )
    assert "channel" in by_path["channels/Spectrometer"]["meaning"].lower()


def test_small_cap_still_explains_what_it_walks(focal_plane_file: Path) -> None:
    result = explain_output_file(focal_plane_file, max_items=50)
    assert _coverage(result) >= 0.9, result["unexplained"]
    assert result["truncated"] is True
    assert result["total_nodes"] > 50
    assert result["walked"] == 50


def test_unexplained_count_makes_the_cap_visible(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "odd.h5"
    with h5py.File(path, "w") as handle:
        for i in range(6):
            handle.create_dataset(f"mystery_{i}", data=[1.0])
    result = explain_output_file(path, max_items=4)
    assert result["unexplained_count"] == len(result["unexplained"]) == 4
    assert result["explained_count"] == 0
    assert result["truncated"] is True
    assert result["total_nodes"] == 6


def test_radiometric_tables_are_explained(radiometric_file: Path) -> None:
    result = explain_output_file(radiometric_file, max_items=500)
    assert result["pipeline"] == "radiometric"
    assert _coverage(result) >= 0.95, result["unexplained"][:10]


def test_breadth_first_walk_reaches_every_top_level_group(
    focal_plane_file: Path,
) -> None:
    result = explain_output_file(focal_plane_file, max_items=12)
    paths = {item["path"] for item in result["explained"]}
    assert {"channels/Photometer", "channels/Spectrometer"} <= paths


@pytest.mark.parametrize(
    "name",
    [
        "info/exosim/author",
        "radiometric/GJ 1214/table",
        "radiometric/GJ 1214/table.__table_column_meta__",
        "radiometric/GJ 1214/table_to_group/qe/value",
        "channels/Photometer/focal_plane/time_units",
        "channels/Photometer/focal_plane/metadata/wl_min",
        "channels/Photometer/psf/psf_cube",
        "channels/Photometer/path",
        "telescope/final contributes/radiance_0",
    ],
)
def test_describe_node_patterns(name: str) -> None:
    assert describe_node(name) is not None


def test_describe_node_still_none_for_unknown() -> None:
    assert describe_node("something/unrelated") is None
