"""plot_output on radiometric files: table location and structured messages."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import h5py
import pytest

from exosim.mcp.plotting import generate_plots


def _write_h5(path: Path, layout: dict[str, tuple[int, ...]]) -> None:
    with h5py.File(path, "w") as handle:
        for name, shape in layout.items():
            handle.create_dataset(name, shape=shape)


class _FakeRadiometricPlotter:
    """Stands in for RadiometricPlotter; records the table path it was given."""

    created: ClassVar[list[tuple[str, str]]] = []
    fail_apertures: ClassVar[bool] = False

    def __init__(self, input: str, table_path: str = "radiometric") -> None:
        type(self).created.append((input, table_path))

    def plot_table(self, **_kw) -> None:
        pass

    def plot_apertures(self, *_a, **_kw) -> None:
        if type(self).fail_apertures:
            raise KeyError("focal_plane")

    def save_fig(self, name: str) -> None:
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        Path(name).write_bytes(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def fake_radiometric(monkeypatch):
    _FakeRadiometricPlotter.created = []
    _FakeRadiometricPlotter.fail_apertures = False
    monkeypatch.setattr(
        "exosim.plots.radiometric_plotter.RadiometricPlotter",
        _FakeRadiometricPlotter,
    )
    monkeypatch.setattr("exosim.plots.RadiometricPlotter", _FakeRadiometricPlotter)
    return _FakeRadiometricPlotter


def test_radiometric_table_in_apertures_group(
    tmp_path: Path, monkeypatch, fake_radiometric
) -> None:
    """A real radiometric run stores the table under radiometric_apertures/."""
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    out = tmp_path / "test_radio.h5"
    _write_h5(out, {"radiometric_apertures/table": (4,)})

    result = generate_plots(str(out))

    assert result["kind"] == "radiometric"
    assert "error" not in result
    assert fake_radiometric.created[0][1] == "radiometric_apertures"
    assert result["figures"]


def test_radiometric_aperture_failure_is_a_warning(
    tmp_path: Path, monkeypatch, fake_radiometric
) -> None:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    out = tmp_path / "test_radio.h5"
    _write_h5(out, {"radiometric/table": (4,)})
    fake_radiometric.fail_apertures = True

    result = generate_plots(str(out))

    assert "error" not in result
    assert fake_radiometric.created[0][1] == "radiometric"
    assert any("apertures" in w for w in result["warnings"])
    assert any(f.endswith("radiometric.png") for f in result["figures"])


def test_target_list_output_plots_one_figure_per_target(
    tmp_path: Path, monkeypatch, fake_radiometric
) -> None:
    """A target-list run stores radiometric/<target>/table (the real layout)."""
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    out = tmp_path / "targets.h5"
    _write_h5(
        out,
        {
            "radiometric/GJ 1214/table": (4,),
            "radiometric/HD 209458/table": (4,),
            "radiometric_apertures/table": (4,),
        },
    )

    result = generate_plots(str(out))

    assert [c[1] for c in fake_radiometric.created] == [
        "radiometric/GJ 1214",
        "radiometric/HD 209458",
    ]
    names = {Path(f).name for f in result["figures"]}
    assert {"radiometric_GJ_1214.png", "radiometric_HD_209458.png"} <= names


def test_radiometric_without_table_is_structured_message(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    out = tmp_path / "no_table.h5"
    _write_h5(out, {"radiometric_apertures/other": (4,)})

    result = generate_plots(str(out))

    assert result["kind"] == "radiometric"
    assert result["plottable"] is False
    assert "no plottable" in result["message"]
    assert "error" not in result
