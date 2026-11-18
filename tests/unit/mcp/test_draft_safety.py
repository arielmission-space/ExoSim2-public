"""draft_config must not write unsafe pointers or silently overwrite a draft."""

from __future__ import annotations

from pathlib import Path

import pytest

from exosim.mcp.tools import draft_config


def test_config_path_outside_the_sandbox_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside the directories"):
        draft_config("focalplane", {"ConfigPath": str(outside)})


def test_config_path_inside_the_checkout_is_accepted() -> None:
    from exosim.mcp.sandbox import Sandbox

    inside = str(Sandbox.from_env().repo_root / "examples")
    result = draft_config("focalplane", {"ConfigPath": inside})
    assert {"path": "ConfigPath", "value": inside} in result["applied"]


def test_python_task_pointer_is_refused_without_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXOSIM_MCP_NO_EXEC", "1")
    script = tmp_path / "evil.py"
    script.write_text("raise SystemExit\n")
    with pytest.raises(ValueError, match="arbitrary code execution"):
        draft_config("focalplane", {"jitter.jitter_task": str(script)})


def test_python_task_pointer_outside_the_sandbox_is_refused_with_exec(
    tmp_path: Path,
) -> None:
    script = tmp_path / "evil.py"
    script.write_text("raise SystemExit\n")
    with pytest.raises(ValueError, match="outside the directories"):
        draft_config("focalplane", {"jitter.jitter_task": str(script)})


def test_task_class_names_remain_allowed_without_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXOSIM_MCP_NO_EXEC", "1")
    result = draft_config(
        "focalplane", {"jitter.jitter_task": "EstimatePointingJitter"}
    )
    assert result["applied"]


def test_refused_override_writes_no_draft() -> None:
    from exosim.mcp.sandbox import Sandbox

    with pytest.raises(ValueError, match="outside the directories"):
        draft_config("focalplane", {"ConfigPath": "/definitely/not/allowed"})
    assert not (Sandbox.from_env().work_dir / "drafts").exists()


def test_existing_draft_is_not_overwritten_by_default() -> None:
    first = draft_config("focalplane")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        draft_config("focalplane", {"wl_grid.wl_min": "0.55 micron"})
    # The first draft is untouched.
    assert "0.55" not in Path(first["draft"]).read_text()

    again = draft_config(
        "focalplane", {"wl_grid.wl_min": "0.55 micron"}, overwrite=True
    )
    assert 'unit="micron">0.55<' in Path(again["draft"]).read_text()


def test_builtin_task_model_name_is_allowed_without_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plain class name only selects classes under ``exosim.tasks``."""
    from exosim.mcp.authoring import _check_override
    from exosim.mcp.sandbox import Sandbox

    monkeypatch.setenv("EXOSIM_MCP_NO_EXEC", "1")
    box = Sandbox.from_env()
    _check_override(
        "task_model", "LoadOpticalElement", sandbox=box, config_root=tmp_path
    )
    for pointer in ("custom.py", "/abs/path/custom.PY"):
        with pytest.raises(ValueError, match="arbitrary code execution"):
            _check_override("task_model", pointer, sandbox=box, config_root=tmp_path)


@pytest.mark.parametrize(
    "key",
    [
        "qe_map_filename",
        "pnl_filename",
        "fileName",
        "file_name",
        "dp_map",
        "limb_darkening_coefficients",
        "url",
        "targetlist_filepath",
    ],
)
def test_more_path_like_keys_are_confined(key: str, tmp_path: Path) -> None:
    """N5: these keys name files ExoSim opens, so they must stay in the sandbox."""
    from exosim.mcp.authoring import _check_override
    from exosim.mcp.sandbox import Sandbox

    box = Sandbox.from_env()
    with pytest.raises(ValueError, match="outside the directories"):
        _check_override(key, "/etc/passwd", sandbox=box, config_root=box.repo_root)
    # a file under the readable content root is fine
    _check_override(
        key, "__ConfigPath__/data/x.csv", sandbox=box, config_root=box.repo_root
    )


def test_remote_url_override_is_refused(tmp_path: Path) -> None:
    from exosim.mcp.authoring import _check_override
    from exosim.mcp.sandbox import Sandbox

    box = Sandbox.from_env()
    with pytest.raises(ValueError, match="remote URL"):
        _check_override(
            "url", "https://exodb.space/api", sandbox=box, config_root=box.repo_root
        )
