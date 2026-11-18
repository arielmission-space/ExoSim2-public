"""Configuration tools find drafts in the working directory by relative path."""

from __future__ import annotations

from pathlib import Path

from exosim.mcp.configuration import config_requirements, validate_configuration_file
from exosim.mcp.sandbox import Sandbox
from exosim.mcp.tools import draft_config, validate_config


def test_draft_validates_by_bare_name_and_relative_path() -> None:
    result = draft_config("focalplane")
    draft = Path(result["draft"])
    work_dir = Sandbox.from_env().work_dir
    assert draft.is_relative_to(work_dir)

    relative = str(draft.relative_to(work_dir))
    assert relative.startswith("drafts")

    for outcome in (
        validate_configuration_file(relative),
        validate_config(relative),
    ):
        assert outcome["valid"] is True, outcome.get("error")
        assert Path(outcome["configuration"]).resolve() == draft.resolve()

    # A file directly in the working directory is found by its bare name.
    bare = work_dir / "bare_draft.xml"
    bare.write_text(draft.read_text())
    outcome = validate_configuration_file("bare_draft.xml")
    assert outcome["valid"] is True, outcome.get("error")


def test_config_requirements_accepts_a_work_dir_draft() -> None:
    draft = Path(draft_config("focalplane")["draft"])
    work_dir = Sandbox.from_env().work_dir
    result = config_requirements("focalplane", str(draft.relative_to(work_dir)))
    assert "configuration_error" not in result
    assert "present" in result


def test_repo_relative_paths_still_work() -> None:
    result = validate_configuration_file("examples/main_example.xml")
    assert result["valid"] is True
