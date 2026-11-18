"""Tests for dynamic recipe discovery in the MCP layer."""

import pytest

from exosim.mcp.discovery import (
    _HIDDEN_RECIPES,
    discover_recipes,
    get_recipe_instance_params,
)


def test_discover_recipes():
    """Test that discover_recipes finds all available recipes."""
    pipelines = discover_recipes()

    # Should find at least the known recipes
    assert len(pipelines) > 0, "No pipelines discovered"

    # Check for known recipes
    known_recipes = ["focalplane", "radiometric", "subexposures", "ndrs"]
    for recipe_name in known_recipes:
        assert recipe_name in pipelines, f"Known recipe '{recipe_name}' not discovered"

    # Verify structure of discovered pipelines
    for name, pipeline_def in pipelines.items():
        assert pipeline_def.name == name, "Pipeline name mismatch"
        assert pipeline_def.recipe.startswith("exosim.recipes."), (
            "Recipe path should start with 'exosim.recipes.'"
        )
        assert pipeline_def.cli_command.startswith("exosim "), (
            "CLI command should start with 'exosim '"
        )
        assert len(pipeline_def.required_inputs) > 0, (
            "Should have at least one required input"
        )
        assert len(pipeline_def.output_description) > 0, (
            "Should have output description"
        )


def test_discover_recipes_includes_all_exported():
    """Test that discovery includes all recipes exported from exosim.recipes."""
    import exosim.recipes as recipes

    pipelines = discover_recipes()

    # Get all exported recipe classes
    recipe_classes = []
    for name in dir(recipes):
        obj = getattr(recipes, name)
        if isinstance(obj, type) and hasattr(obj, "__init__"):
            recipe_classes.append(name)

    # All exported recipes should be discovered
    # (except for base classes or utility classes)
    discovered_recipes = {p.recipe.split(".")[-1] for p in pipelines.values()}
    for recipe_class in recipe_classes:
        # Skip non-recipe classes
        if recipe_class.startswith("_") or recipe_class in _HIDDEN_RECIPES:
            continue
        assert recipe_class in discovered_recipes, (
            f"Recipe {recipe_class} exported but not discovered"
        )


def test_get_recipe_instance_params():
    """Test parameter extraction from recipe classes."""
    import exosim.recipes as recipes

    # Test with CreateFocalPlane
    params = get_recipe_instance_params(recipes.CreateFocalPlane)
    assert "options_file" in params, "Should have options_file parameter"
    assert "output_file" in params, "Should have output_file parameter"
    assert params["options_file"]["required"], "options_file should be required"
    assert params["output_file"]["required"], "output_file should be required"

    # Test with CreateSubExposures (should have input_file)
    params = get_recipe_instance_params(recipes.CreateSubExposures)
    assert "options_file" in params, "Should have options_file parameter"
    assert "input_file" in params, "Should have input_file parameter"
    assert "output_file" in params, "Should have output_file parameter"


def test_pipeline_required_inputs():
    """Test that required inputs are correctly identified."""
    pipelines = discover_recipes()

    # Focalplane and radiometric don't need input files
    for pipeline_name in ["focalplane", "radiometric"]:
        pipeline_def = pipelines.get(pipeline_name)
        assert pipeline_def is not None, f"Pipeline {pipeline_name} not found"
        assert "input" not in pipeline_def.required_inputs, (
            f"{pipeline_name} should not require input file"
        )
        assert "configuration" in pipeline_def.required_inputs, (
            f"{pipeline_name} should require configuration"
        )
        assert "output" in pipeline_def.required_inputs, (
            f"{pipeline_name} should require output"
        )

    # Subexposures and ndrs need input files
    for pipeline_name in ["subexposures", "ndrs"]:
        pipeline_def = pipelines.get(pipeline_name)
        assert pipeline_def is not None, f"Pipeline {pipeline_name} not found"
        assert "input" in pipeline_def.required_inputs, (
            f"{pipeline_name} should require input file"
        )


def test_discovery_fallback():
    """Test that discovery gracefully handles errors."""
    # Try to discover from non-existent module
    pipelines = discover_recipes("nonexistent.module")
    assert pipelines == {}, "Should return empty dict for non-existent module"


def test_pipeline_definition_serialization():
    """Test that pipeline definitions can be serialized."""
    pipelines = discover_recipes()

    for pipeline_def in pipelines.values():
        # Should be serializable to dict
        data = pipeline_def.to_dict()
        assert isinstance(data, dict), "Should serialize to dict"
        assert "name" in data, "Serialized dict should have name"
        assert "recipe" in data, "Serialized dict should have recipe"
        assert "cli_command" in data, "Serialized dict should have cli_command"
        assert "required_inputs" in data, "Serialized dict should have required_inputs"
        assert "output_description" in data, (
            "Serialized dict should have output_description"
        )
        assert isinstance(data["required_inputs"], list), (
            "required_inputs should be a list in serialized form"
        )


@pytest.mark.parametrize(
    ("class_name", "expected_snake"),
    [
        ("CreateFocalPlane", "create_focal_plane"),
        ("RadiometricModel", "radiometric_model"),
        ("CreateNDRs", "create_n_d_rs"),
        ("SimpleTest", "simple_test"),
    ],
)
def test_camel_to_snake(class_name, expected_snake):
    """Test CamelCase to snake_case conversion."""
    from exosim.mcp.discovery import _camel_to_snake

    result = _camel_to_snake(class_name)
    assert result == expected_snake, f"Expected {expected_snake}, got {result}"


# --- import errors are surfaced, not swallowed -------------------------------
@pytest.fixture
def clean_errors():
    from exosim.mcp import discovery

    discovery.clear_discovery_errors()
    yield discovery
    discovery.clear_discovery_errors()


def _fail_import(missing: str):
    import importlib

    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name.startswith("exosim.recipes"):
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return real(name, *args, **kwargs)

    return fake


def test_discovery_records_the_import_error(monkeypatch, clean_errors) -> None:
    monkeypatch.setattr("importlib.import_module", _fail_import("distutils"))
    assert clean_errors.discover_recipes() == {}
    (error,) = clean_errors.discovery_errors()
    assert error["missing_module"] == "distutils"
    assert "setuptools" in error["hint"]


def test_capabilities_and_list_pipelines_surface_the_error(
    monkeypatch, clean_errors
) -> None:
    from exosim.mcp import tools

    assert tools.capabilities()["discovery_errors"] == []
    monkeypatch.setattr("importlib.import_module", _fail_import("distutils"))
    clean_errors.discover_recipes()
    monkeypatch.undo()
    assert tools.capabilities()["discovery_errors"][0]["missing_module"] == "distutils"
    assert tools.list_pipelines()["discovery_errors"][0]["missing_module"] == (
        "distutils"
    )


def test_describe_pipeline_returns_a_structured_import_error(
    monkeypatch, clean_errors
) -> None:
    import importlib

    from exosim.mcp import pipelines, tools

    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == "exosim.recipes":
            raise ModuleNotFoundError("No module named 'distutils'", name="distutils")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(pipelines.importlib, "import_module", fake)
    result = tools.describe_pipeline("focalplane")
    assert result["missing_module"] == "distutils"
    assert "distutils" in result["error"]
    assert "setuptools" in result["hint"]
    assert tools.capabilities()["discovery_errors"]


def test_observation_recipe_is_not_exposed() -> None:
    """SimulateObservation writes fixed temp files in the cwd: hidden (D4)."""
    from exosim.mcp import tools
    from exosim.mcp.pipelines import PIPELINES

    assert "observation" not in discover_recipes()
    assert "observation" not in PIPELINES
    assert "observation" not in tools.capabilities()["pipelines"]
    assert "error" in tools.describe_pipeline("observation")
    assert "error" in tools.config_requirements("observation")
