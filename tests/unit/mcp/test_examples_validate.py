"""Every example entry-point configuration must validate through the MCP tools."""

from __future__ import annotations

import pytest

from exosim.mcp import tools
from exosim.mcp.configuration import config_root, validate_configuration_file

from .conftest import REPO_ROOT

_ENTRY_POINTS = [
    entry["path"]
    for entry in tools.list_examples()["examples"]
    if entry["is_entrypoint"]
]


def test_entry_points_are_listed() -> None:
    assert "examples/main_example.xml" in _ENTRY_POINTS
    assert "examples/radiometric/main_radiometric.xml" in _ENTRY_POINTS


@pytest.mark.parametrize("example", _ENTRY_POINTS)
def test_example_entry_point_validates(example: str) -> None:
    result = tools.validate_config(example)
    assert result["valid"], result.get("error")


def test_explicit_config_path_wins() -> None:
    # The wrong include root must break the includes: the argument overrides the file.
    result = validate_configuration_file(
        "examples/radiometric/main_radiometric.xml",
        config_path="examples/radiometric",
    )
    assert not result["valid"]


def test_in_file_config_path_is_relative_to_the_file() -> None:
    root = config_root(REPO_ROOT / "examples" / "radiometric" / "main_radiometric.xml")
    assert root == (REPO_ROOT / "examples").resolve()


def test_config_requirements_radiometric() -> None:
    result = tools.config_requirements("radiometric")
    assert result["reference_example"] == "examples/radiometric/main_radiometric.xml"
    assert result["reference_sections"]
