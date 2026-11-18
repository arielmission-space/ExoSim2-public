"""Tests for advanced ExoSim MCP tools."""

from pathlib import Path

from exosim.mcp.tools import compare_configs, diagnose_error, suggest_next_steps


def test_compare_configs_missing_files(tmp_path, monkeypatch):
    """Test compare_configs with non-existent files."""
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    config1 = tmp_path / "nonexistent1.xml"
    config2 = tmp_path / "nonexistent2.xml"

    result = compare_configs(str(config1), str(config2))

    assert result["valid"] is False
    assert "error" in result
    assert "Failed to load" in result["error"]


def test_suggest_next_steps_after_config_validation():
    """Test suggest_next_steps suggests focalplane after config validation."""
    result = suggest_next_steps("config_validated")

    assert result["context"] == "config_validated"
    assert len(result["suggestions"]) > 0
    assert result["suggestions"][0]["step"] == "run_focalplane"
    assert "focalplane" in result["suggestions"][0]["command"]


def test_suggest_next_steps_after_focalplane():
    """After focalplane: subexposures consume focal.h5; radiometric is independent."""
    result = suggest_next_steps("focalplane_complete")

    first = result["suggestions"][0]
    assert first["step"] == "run_subexposures"
    assert "-i focal.h5" in first["command"]
    radio = [s for s in result["suggestions"] if s["step"] == "run_radiometric"]
    assert radio
    assert "focal.h5" not in radio[0]["required_files"]
    assert " -i " not in radio[0]["command"]


def test_suggest_next_steps_after_radiometric():
    """Radiometric is a leaf branch: never suggest feeding it to subexposures."""
    result = suggest_next_steps("radiometric_complete")

    assert len(result["suggestions"]) > 0
    for item in result["suggestions"]:
        assert "subexposures" not in item["command"]
        assert " -i radio.h5" not in item["command"]


def test_suggest_next_steps_after_subexposures():
    """Test suggest_next_steps suggests ndrs after subexposures."""
    result = suggest_next_steps("subexposures_complete")

    assert len(result["suggestions"]) > 0
    assert result["suggestions"][0]["step"] == "run_ndrs"
    assert "ndrs" in result["suggestions"][0]["command"]


def test_suggest_next_steps_after_ndrs():
    """Test suggest_next_steps suggests analysis after ndrs."""
    result = suggest_next_steps("ndrs_complete")

    assert len(result["suggestions"]) > 0
    assert result["suggestions"][0]["step"] == "analyze_results"


def test_suggest_next_steps_generic():
    """Test suggest_next_steps with unknown context."""
    result = suggest_next_steps("unknown_state")

    assert len(result["suggestions"]) > 0
    # Should suggest starting with config validation
    assert "validate" in result["suggestions"][0]["step"].lower()


def test_diagnose_error_file_not_found():
    """Test diagnose_error recognizes file not found errors."""
    result = diagnose_error(
        "FileNotFoundError: [Errno 2] No such file or directory: 'missing.xml'",
        context="loading configuration",
    )

    assert result["category"] == "missing_file"
    assert len(result["suggestions"]) > 0
    assert any("path" in s.lower() for s in result["suggestions"])


def test_diagnose_error_missing_parameter():
    """Test diagnose_error recognizes missing parameter errors."""
    result = diagnose_error(
        "KeyError: 'radiometric_model' not found in configuration",
        context="radiometric pipeline",
    )

    assert result["category"] == "missing_parameter"
    assert len(result["suggestions"]) > 0
    assert any("configuration" in s.lower() for s in result["suggestions"])


def test_diagnose_error_invalid_value():
    """Test diagnose_error recognizes invalid value errors."""
    result = diagnose_error(
        "ValueError: wavelength must be positive",
        context="loading payload",
    )

    assert result["category"] == "invalid_parameter"
    assert len(result["suggestions"]) > 0
    assert any(
        "value" in s.lower() or "unit" in s.lower() for s in result["suggestions"]
    )


def test_diagnose_error_memory():
    """Test diagnose_error recognizes memory errors."""
    result = diagnose_error(
        "MemoryError: Unable to allocate array",
        context="focalplane simulation",
    )

    assert result["category"] == "memory_issue"
    assert len(result["suggestions"]) > 0
    assert any(
        "memory" in s.lower() or "chunk" in s.lower() for s in result["suggestions"]
    )


def test_diagnose_error_import():
    """Test diagnose_error recognizes import errors."""
    result = diagnose_error(
        "ModuleNotFoundError: No module named 'scipy'",
        context="starting exosim",
    )

    assert result["category"] == "missing_dependency"
    assert len(result["suggestions"]) > 0
    assert any("install" in s.lower() for s in result["suggestions"])


def test_diagnose_error_generic():
    """Test diagnose_error handles unknown errors."""
    result = diagnose_error(
        "SomeUnknownError: Something went wrong",
        context="unknown operation",
    )

    assert result["category"] == "general_error"
    assert len(result["suggestions"]) > 0
    assert any(
        "debug" in s.lower() or "trace" in s.lower() for s in result["suggestions"]
    )


def test_diagnose_error_parses_a_traceback():
    """diagnose_error extracts the exception type and failing frame."""
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/x/exosim/tasks/load/load_options.py", line 233, in _finalise\n'
        "    raise FileNotFoundError(xmlfile)\n"
        "FileNotFoundError: Referenced config file not found: sky.xml\n"
    )
    result = diagnose_error("run failed", traceback_text=tb)

    assert result["error_type"] == "FileNotFoundError"
    assert result["exception_message"].endswith("sky.xml")
    assert "load_options.py:233" in result["last_frame"]
    assert result["category"] == "missing_file"


def test_config_requirements_lists_reference_sections():
    from exosim.mcp.tools import config_requirements

    result = config_requirements("focalplane")
    sections = {s["section"] for s in result["reference_sections"]}

    assert {"wl_grid", "time_grid", "payload", "sky"} <= sections
    assert all("see" in s for s in result["reference_sections"])
    # the pointer should be a page a person can read, not a source file
    seen = [s["see"] for s in result["reference_sections"] if s["see"]]
    assert seen
    assert all(p.startswith("docs/source/") for p in seen)
    # each section also lists the keys the reference example sets
    wl_grid = next(s for s in result["reference_sections"] if s["section"] == "wl_grid")
    assert {"wl_min", "wl_max"} <= set(wl_grid["keys"])


def test_draft_config_produces_a_valid_configuration():
    from exosim.mcp.tools import draft_config

    result = draft_config("focalplane")

    assert result["valid"] is True
    assert "payload" in result["top_level_sections"]
    assert Path(result["draft"]).is_file()


def test_draft_config_applies_and_reports_overrides():
    from exosim.mcp.tools import draft_config

    result = draft_config(
        "focalplane",
        {"wl_grid.wl_min": "0.55 micron", "sky.nonsense.key": "3"},
    )

    assert result["valid"] is True
    assert {a["path"] for a in result["applied"]} == {"wl_grid.wl_min"}

    missed = result["not_applied"][0]
    assert missed["path"] == "sky.nonsense.key"
    assert missed["available"]  # the keys that *do* exist under sky

    assert 'unit="micron">0.55<' in Path(result["draft"]).read_text()


def test_draft_config_unknown_pipeline():
    from exosim.mcp.tools import draft_config

    assert "error" in draft_config("not-a-pipeline")


def test_config_requirements_without_a_config_reports_no_diff():
    from exosim.mcp.tools import config_requirements

    result = config_requirements("focalplane")
    assert "missing" not in result
    assert "present" not in result


def test_config_requirements_diffs_a_given_config():
    from exosim.mcp.tools import config_requirements

    result = config_requirements("focalplane", "examples/main_example.xml")

    assert result["missing"] == []
    assert "wl_grid" in result["present"]


def test_config_requirements_unknown_pipeline():
    from exosim.mcp.tools import config_requirements

    result = config_requirements("not-a-pipeline")
    assert "error" in result


def test_config_requirements_extra_excludes_framework_keys():
    from exosim.mcp.tools import config_requirements

    result = config_requirements("focalplane", "examples/main_example.xml")
    assert result["extra"] == []
    assert not {"ConfigPath", "fileName", "random_seed"} & set(result["extra"])
