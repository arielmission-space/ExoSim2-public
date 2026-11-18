"""Tests for the MCP plot_output tool."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import h5py
import pytest

from exosim.mcp.plotting import _detect_kind, generate_plots


def _write_h5(path: Path, layout: dict[str, tuple[int, ...]]) -> None:
    with h5py.File(path, "w") as handle:
        for name, shape in layout.items():
            handle.create_dataset(name, shape=shape)


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ({"channels/Photometer/focal_plane": (1, 4, 4)}, "focalplane"),
        ({"channels/Photometer/NDRs": (2, 4, 4)}, "ndrs"),
        ({"channels/Photometer/SubExposures": (2, 4, 4)}, "subexposures"),
        ({"radiometric/table": (4,)}, "radiometric"),
        ({"something/else": (1,)}, None),
    ],
)
def test_detect_kind(tmp_path: Path, layout, expected) -> None:
    out = tmp_path / "out.h5"
    _write_h5(out, layout)
    assert _detect_kind(out) == expected


def test_generate_plots_missing_file() -> None:
    result = generate_plots("does_not_exist.h5")
    assert "error" in result


def test_generate_plots_unsupported_kind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    out = tmp_path / "mystery.h5"
    _write_h5(out, {"unknown/group": (1,)})
    result = generate_plots(str(out))
    assert "error" in result
    assert "supported" in result


def test_generate_plots_dispatches_and_collects_figures(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    out = tmp_path / "focal.h5"
    _write_h5(out, {"channels/Photometer/focal_plane": (1, 4, 4)})

    class FakePlotter:
        def __init__(self, input: str) -> None:
            self.input = input
            self._target: Path | None = None

        def plot_focal_plane(self, **_kw) -> None:
            pass

        def plot_efficiency(self, **_kw) -> None:
            pass

        def save_fig(self, name: str) -> None:
            Path(name).parent.mkdir(parents=True, exist_ok=True)
            Path(name).write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        "exosim.plots.focal_plane_plotter.FocalPlanePlotter", FakePlotter
    )

    result = generate_plots(str(out), "focalplane")

    assert result["kind"] == "focalplane"
    assert Path(result["output_dir"]).name.startswith("focal-")
    assert len(result["figures"]) == 2
    assert all(p.endswith(".png") for p in result["figures"])


class _FakeFocalPlotter:
    """Writes the two focal-plane PNGs; counts how many renders overlap."""

    active = 0
    peak = 0
    guard = threading.Lock()
    delay = 0.0
    fail = False

    def __init__(self, input: str) -> None:
        pass

    def plot_focal_plane(self, **_kw) -> None:
        cls = type(self)
        with cls.guard:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
        time.sleep(cls.delay)
        with cls.guard:
            cls.active -= 1

    def plot_efficiency(self, **_kw) -> None:
        if type(self).fail:
            raise RuntimeError("boom")

    def save_fig(self, name: str) -> None:
        Path(name).write_bytes(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def fake_focal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "exosim.plots.focal_plane_plotter.FocalPlanePlotter", _FakeFocalPlotter
    )
    _FakeFocalPlotter.active = _FakeFocalPlotter.peak = 0
    _FakeFocalPlotter.delay = 0.0
    _FakeFocalPlotter.fail = False
    return _FakeFocalPlotter


def test_concurrent_plots_are_serialised(fake_focal, tmp_path: Path) -> None:
    """pyplot is global state: two renders must never overlap (regression)."""
    out = tmp_path / "focal.h5"
    _write_h5(out, {"channels/Photometer/focal_plane": (1, 4, 4)})
    fake_focal.delay = 0.05
    results: list[dict] = []
    threads = [
        threading.Thread(target=lambda: results.append(generate_plots(str(out))))
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert fake_focal.peak == 1
    assert len(results) == 6
    assert all("error" not in r for r in results)


def test_unknown_scale_is_rejected(fake_focal, tmp_path: Path) -> None:
    out = tmp_path / "focal.h5"
    _write_h5(out, {"channels/Photometer/focal_plane": (1, 4, 4)})
    with pytest.raises(ValueError, match="scale"):
        generate_plots(str(out), scale="log")
    assert "error" not in generate_plots(str(out), scale="db")


def test_failed_call_does_not_return_stale_figures(fake_focal, tmp_path: Path) -> None:
    out = tmp_path / "focal.h5"
    _write_h5(out, {"channels/Photometer/focal_plane": (1, 4, 4)})
    first = generate_plots(str(out), time_step=0)
    assert len(first["figures"]) == 2
    time.sleep(0.01)
    fake_focal.fail = True
    second = generate_plots(str(out), time_step=1)
    assert "error" in second
    # focal_plane_1.png was written by this call; efficiency.png is stale
    assert [Path(f).name for f in second["figures"]] == ["focal_plane_1.png"]


def test_same_stem_outputs_get_separate_plot_dirs(fake_focal, tmp_path: Path) -> None:
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        _write_h5(
            tmp_path / sub / "x.h5", {"channels/Photometer/focal_plane": (1, 4, 4)}
        )
    one = generate_plots(str(tmp_path / "a" / "x.h5"))
    two = generate_plots(str(tmp_path / "b" / "x.h5"))
    assert one["output_dir"] != two["output_dir"]
    assert not set(one["figures"]) & set(two["figures"])
