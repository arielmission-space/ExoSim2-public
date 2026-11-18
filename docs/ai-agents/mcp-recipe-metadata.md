# How the MCP server discovers recipes

The MCP server exposes ExoSim's pipelines (`list_pipelines`, `describe_pipeline`,
`run_pipeline`) from the recipe classes in `src/exosim/recipes/`. It reads them
by introspection — there is no registry to keep in sync. A new recipe that
follows the existing conventions appears automatically.

## Automatic discovery

`exosim.mcp.discovery.discover_recipes()` walks `exosim.recipes`, takes every
class defined in that package, and builds a `PipelineDefinition` for each
(`exosim.mcp.pipelines.PIPELINES`). For a class with no explicit metadata it
derives:

- **name** — the class name with a leading `Create` / `Simulate` / `Generate` /
  `Build` and a trailing `Model` / `Recipe` / `Pipeline` removed, lower-cased
  with the word breaks dropped. `CreateFocalPlane` → `focalplane`,
  `RadiometricModel` → `radiometric`, `CreateNDRs` → `ndrs`.
- **cli_command** — `exosim <name>`.
- **required_inputs** — from the `__init__` parameter names: `options_file` or
  `configuration` → `configuration`; `input_file` or `input` → `input`;
  `output_file` or `output` → `output`. If none match, it falls back to
  `("configuration", "output")`.
- **output_description** — the first sentence of the class docstring that
  mentions "output" or "produce", else a value keyed off the class name.

`describe_pipeline` additionally parses the recipe docstring's numpydoc
`Parameters` section, so a recipe with a documented `Parameters` block gets
per-parameter types, defaults and descriptions in its tool output.

If discovery imports nothing (for example the `exosim.recipes` import fails),
`pipelines.py` falls back to a hard-coded `_FALLBACK_PIPELINES` map for the four
core pipelines.

## When to add `__mcp_metadata__`

Only when the derived values are wrong: the name is awkward, the CLI command
differs, or the required inputs are misread. It is a class attribute:

```python
class AdvancedCalibration(TimedClass, log.Logger):
    """Advanced calibration pipeline."""

    __mcp_metadata__ = {
        "name": "advanced_calib",
        "cli_command": "exosim advanced-calibration",
        "required_inputs": ["configuration", "input", "output"],
        "output_description": "Calibrated data with uncertainty estimates (HDF5).",
    }

    def __init__(
        self,
        options_file: str,
        input_file: str,
        output_file: str,
        calibration_mode: str = "standard",
    ) -> None:
        ...
```

Every key is optional; an absent key keeps the derived value. Recipes that
follow the naming conventions (`Create*` / `*Model`, `options_file` /
`input_file` / `output_file`) do not need it.

## How MCP arguments reach the recipe

`run_pipeline` takes JSON-primitive arguments (`configuration`, `output`,
`input_file`, `n_threads`, `chunk_size`, `overwrite`, `plot`). The dispatcher
(`exosim.mcp.pipelines._map_recipe_kwargs`) maps them onto whichever parameter
name the recipe's `__init__` actually uses:

| `run_pipeline` argument | recipe parameter (first that exists) |
|-------------------------|--------------------------------------|
| `configuration`         | `options_file`, `configuration`      |
| `output`                | `output_file`, `output`              |
| `input_file`            | `input_file`, `input`                |
| `plot`                  | `plot` (only if the recipe accepts it) |

A recipe whose `__init__` is `**kwargs` receives the canonical names
(`options_file`, `output_file`, `input_file`).

## Checking a new recipe

```python
from exosim.mcp.discovery import discover_recipes

pipelines = discover_recipes()
print(sorted(pipelines))                 # your name should be here
print(pipelines["yourname"].to_dict())   # inspect the derived definition
```

```bash
pytest tests/unit/mcp/test_mcp_discovery.py tests/unit/mcp/test_mcp_tools.py -q
```

If the recipe does not appear: confirm it is exported from
`src/exosim/recipes/__init__.py`, that its class is defined under
`exosim.recipes`, and that importing the module raises no error.
