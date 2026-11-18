"""Unit tests for ExoSim MCP tool helpers."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import h5py
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from exosim.mcp.pipelines import dispatch_pipeline
from exosim.mcp.tools import (
    customisation_guide,
    exosim_project_context,
    inspect_output,
    knowledge_base_manifest_tool,
    list_pipelines,
    search_knowledge_base,
    validate_config,
)


def test_search_knowledge_base_returns_sources() -> None:
    """Test retrieval returns grounded ExoSim source passages."""
    result = search_knowledge_base(
        "radiometric model",
        max_results=2,
    )

    assert result["status"] == "ok"
    assert result["sources"]
    assert {"path", "start_line", "end_line", "score", "text"} <= set(
        result["sources"][0]
    )
    assert "minimum_retrieval_score" in result


def test_search_knowledge_base_respects_char_budget() -> None:
    """max_chars caps each passage and flags truncation."""
    full = search_knowledge_base("radiometric model photon noise", max_results=3)
    assert full["status"] == "ok"

    budgeted = search_knowledge_base(
        "radiometric model photon noise", max_results=3, max_chars=900
    )
    assert budgeted["truncated"] is True
    for source in budgeted["sources"]:
        assert len(source["text"]) <= 320
        if source.get("truncated"):
            assert source["full_text_chars"] > len(source["text"])
            assert "path" in source
            assert "start_line" in source


def test_knowledge_base_manifest_lists_approved_sources() -> None:
    """Test manifest reports approved retrieval sources and matched files."""
    result = knowledge_base_manifest_tool()

    patterns = {source["pattern"] for source in result["source_patterns"]}
    assert "docs/source/user/**/*.rst" in patterns
    assert result["matched_file_count"] > 0
    assert result["minimum_retrieval_score"] > 0


def test_exosim_project_context_reads_curated_files() -> None:
    """Test project context returns curated agent guidance files."""
    result = exosim_project_context()

    assert "AGENTS.md" in result["files"]
    assert "docs/ai-agents/codebase-map.md" in result["files"]
    assert result["missing"] == []
    assert "capabilities" in result


def test_capabilities_reports_environment() -> None:
    from exosim.mcp.tools import capabilities

    result = capabilities()

    assert result["versions"]["exosim"]
    assert "focalplane" in result["pipelines"]
    assert "search_knowledge_base" in result["tools"]["always_available"]
    # conftest leaves execution enabled (inline backend)
    assert result["execution_enabled"] is True
    assert "run_pipeline" in result["tools"]["execution_only"]
    assert result["retrieval"]["backend"] in {"bm25", "hybrid"}


def test_capabilities_hides_execution_tools_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exosim.mcp.tools import capabilities

    monkeypatch.setenv("EXOSIM_MCP_NO_EXEC", "1")
    result = capabilities()

    assert result["execution_enabled"] is False
    assert result["tools"]["execution_only"] == []
    assert "run_pipeline" in result["tools"]["disabled"]


def test_search_knowledge_base_reports_insufficient_evidence() -> None:
    """Test retrieval reports insufficient evidence for unmatched queries."""
    result = search_knowledge_base("quxzzy frobnicated zputanium blorptastic")

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
    assert "minimum_retrieval_score" in result


def test_customisation_guide_returns_grounded_sources() -> None:
    """Test customisation guide uses retrieval over developer docs."""
    result = customisation_guide(
        "write a custom task",
    )

    assert result["status"] == "ok"
    assert result["sources"]


def test_list_pipelines_contains_radiometric() -> None:
    """Test pipeline listing exposes the radiometric workflow."""
    result = list_pipelines()
    names = {pipeline["name"] for pipeline in result["pipelines"]}

    assert "radiometric" in names


def test_describe_pipeline_reports_the_recipe_parameters() -> None:
    """describe_pipeline exposes the recipe's __init__ parameters."""
    from exosim.mcp.tools import describe_pipeline

    result = describe_pipeline("subexposures")
    params = {p["name"]: p for p in result["parameters"]}

    assert result["summary"]
    assert params["input_file"]["required"] is True
    assert params["input_file"]["mcp_argument"] == "input_file"
    assert params["options_file"]["mcp_argument"] == "configuration"
    assert params["options_file"]["description"]  # parsed from the docstring


def test_describe_pipeline_rejects_unknown_names() -> None:
    from exosim.mcp.tools import describe_pipeline

    assert "error" in describe_pipeline("does-not-exist")


def test_validate_config_reports_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test configuration validation reports loader errors."""
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    config = tmp_path / "broken.xml"
    config.write_text("<root>", encoding="utf-8")

    result = validate_config(str(config))

    assert result["valid"] is False
    assert "error" in result


def test_validate_config_resolves_includes_next_to_the_file() -> None:
    """The bundled example uses ``<ConfigPath>.`` and must validate anywhere.

    The MCP layer resolves relative ``<config>`` includes next to the
    configuration file, not the current working directory.
    """
    result = validate_config("examples/main_example.xml")

    assert result["valid"] is True, result.get("error")
    assert "payload" in result["top_level_sections"]


def test_inspect_output_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test HDF5 output inspection returns datasets and attributes."""
    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    output = tmp_path / "output.h5"
    with h5py.File(output, "w") as handle:
        handle.attrs["creator"] = "test"
        group = handle.create_group("radiometric")
        group.create_dataset("values", data=[1, 2, 3])

    result = inspect_output(str(output))

    assert result["format"] == "hdf5"
    assert result["root_attrs"]["creator"] == "test"
    assert any(item["path"] == "radiometric/values" for item in result["items"])
    radiometric = next(i for i in result["items"] if i["path"] == "radiometric")
    assert "meaning" in radiometric
    assert radiometric["stage"] == "radiometric"


def test_explain_output_annotates_known_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """explain_output describes recognised nodes and guesses the pipeline."""
    from exosim.mcp.tools import explain_output

    monkeypatch.setenv("EXOSIM_MCP_REPO_ROOT", str(tmp_path))
    output = tmp_path / "ndrs.h5"
    with h5py.File(output, "w") as handle:
        channels = handle.create_group("channels")
        ch0 = channels.create_group("Photometer")
        ch0.create_dataset("NDRs", data=[[1, 2], [3, 4]])
        ch0.create_dataset("mystery_dataset", data=[0])

    result = explain_output(str(output))

    assert result["pipeline"] == "ndrs"
    explained = {e["path"] for e in result["explained"]}
    assert "channels" in explained
    assert any(e["node"] == "NDRs" for e in result["explained"])
    assert any("mystery_dataset" in p for p in result["unexplained"])


def test_dispatch_pipeline_calls_focalplane_recipe(tmp_path: Path) -> None:
    """Test pipeline dispatch calls the focal-plane recipe."""
    config = tmp_path / "config.xml"
    output = tmp_path / "mcp-runs" / "output.h5"
    config.write_text("<root />", encoding="utf-8")
    mock_recipe = MagicMock()
    fake_recipes = SimpleNamespace(CreateFocalPlane=mock_recipe)
    with patch.dict(sys.modules, {"exosim.recipes": fake_recipes}):
        result = dispatch_pipeline(
            "focalplane",
            str(config),
            str(output),
            repo_root=tmp_path,
            work_dir=tmp_path / "mcp-runs",
            overwrite=False,
        )

    mock_recipe.assert_called_once_with(
        options_file=str(config.resolve()),
        output_file=str(output.resolve()),
    )
    assert result["pipeline"] == "focalplane"


def test_dispatch_pipeline_maps_args_to_recipe_signature(tmp_path: Path) -> None:
    """Dispatch inspects the recipe __init__ and passes only what it accepts."""
    config = tmp_path / "config.xml"
    output = tmp_path / "mcp-runs" / "output.h5"
    config.write_text("<root />", encoding="utf-8")

    seen: dict[str, object] = {}

    class FakeFocalPlane:
        def __init__(
            self,
            options_file: str,
            output_file: str,
            store_config: bool = False,
            slim_output: bool = False,
        ) -> None:
            seen.update(options_file=options_file, output_file=output_file)

    fake_recipes = SimpleNamespace(CreateFocalPlane=FakeFocalPlane)
    with patch.dict(sys.modules, {"exosim.recipes": fake_recipes}):
        dispatch_pipeline(
            "focalplane",
            str(config),
            str(output),
            repo_root=tmp_path,
            work_dir=tmp_path / "mcp-runs",
            overwrite=False,
        )

    assert seen == {
        "options_file": str(config.resolve()),
        "output_file": str(output.resolve()),
    }


def test_run_recipe_restores_runconfig() -> None:
    """run_recipe must not leak the RunConfig values it sets."""
    from exosim.mcp.pipelines import run_recipe
    from exosim.utils import RunConfig

    before = (RunConfig.n_job, RunConfig.chunk_size)
    seen: dict[str, object] = {}

    class FakeRecipe:
        def __init__(self, **_: object) -> None:
            seen["n_job"] = RunConfig.n_job

    fake_recipes = SimpleNamespace(FakeRecipe=FakeRecipe)
    spec = {
        "recipe": "exosim.recipes.FakeRecipe",
        "recipe_kwargs": {},
        "n_threads": 1,
        "chunk_size": 2.0,
    }
    with patch.dict(sys.modules, {"exosim.recipes": fake_recipes}):
        run_recipe(spec)

    assert seen["n_job"] == 1  # applied during the run
    assert (RunConfig.n_job, RunConfig.chunk_size) == before  # restored after


def test_build_server_reports_missing_optional_dependency() -> None:
    """Test server build reports a clear error when MCP is not installed."""
    from exosim.mcp.server import build_server

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mcp.server":
            raise ImportError("missing mcp")
        return original_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=guarded_import),
        pytest.raises(ImportError, match=r"pip install \"exosim\[mcp\]\""),
    ):
        build_server()


def test_list_examples_catalogues_the_bundled_configs() -> None:
    from exosim.mcp.tools import list_examples

    result = list_examples()
    by_path = {e["path"]: e for e in result["examples"]}

    assert result["count"] >= 8
    assert not any(".ipynb_checkpoints" in p for p in by_path)
    main = by_path["examples/main_example.xml"]
    assert main["is_entrypoint"] is True
    assert "focalplane" in main["suits_pipelines"]
    assert any("sky_example.xml" in inc for inc in main["includes"])


def test_inspect_output_malformed_table_names_the_file(tmp_path: Path) -> None:
    """N2: a malformed ECSV gives a message naming the file, not an opaque error."""
    bad = tmp_path / "bad.ecsv"
    bad.write_text("# %ECSV 1.0\n# ---\n# datatype: [garbage\nx y\n1 2\n")
    from exosim.mcp.outputs import inspect_output_file

    with pytest.raises(ValueError, match=r"cannot read .*bad\.ecsv"):
        inspect_output_file(bad, repo_root=tmp_path)
