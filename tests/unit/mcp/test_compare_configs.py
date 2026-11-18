"""compare_configs honours each file's own <ConfigPath> (radiometric example)."""

from __future__ import annotations

from pathlib import Path

import pytest

from exosim.mcp.authoring import draft_configuration
from exosim.mcp.comparison import compare_configurations

REPO_ROOT = Path(__file__).resolve().parents[3]
RADIOMETRIC = "examples/radiometric/main_radiometric.xml"


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("EXOSIM_MCP_WORK_DIR", str(tmp_path / "work"))
    return tmp_path / "work"


def test_radiometric_example_compares_with_itself(work: Path) -> None:
    result = compare_configurations(RADIOMETRIC, RADIOMETRIC, repo_root=REPO_ROOT)
    assert result["valid"], result.get("error")
    assert not result["different_values"]
    assert not result["only_in_config1"]
    assert result["identical_sections"]


def test_radiometric_example_compares_with_a_draft(work: Path) -> None:
    drafted = draft_configuration("radiometric", repo_root=REPO_ROOT)
    assert drafted["valid"], drafted.get("error")
    result = compare_configurations(
        RADIOMETRIC, "drafts/radiometric_draft.xml", repo_root=REPO_ROOT
    )
    assert result["valid"], result.get("error")
    assert result["config2"].endswith("drafts/radiometric_draft.xml")


def test_values_differ_handles_arrays_and_tables() -> None:
    import numpy as np
    from astropy.table import Table

    from exosim.mcp.comparison import _values_differ

    assert not _values_differ(np.arange(3.0), np.arange(3.0))
    assert _values_differ(np.arange(3.0), np.arange(3.0) + 1)
    assert _values_differ(np.arange(3.0), np.arange(4.0))
    same = Table({"a": [1.0, 2.0]})
    assert not _values_differ(same, Table({"a": [1.0, 2.0]}))
    assert _values_differ(same, Table({"a": [1.0, 3.0]}))
    assert _values_differ(same, Table({"b": [1.0, 2.0]}))
