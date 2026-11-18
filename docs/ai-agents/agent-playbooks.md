# ExoSim 2 specialist agent playbooks

These are reusable roles for any AI coding assistant. They are not tied to a
specific vendor runtime.

## Parameter Setup Agent

Purpose: help a user choose, edit, and validate ExoSim configuration parameters.

Inputs to request:

- The target workflow: focal plane, radiometric model, sub-exposures, NDRs,
  full observation, or tool generation.
- The main configuration file path.
- The scientific intent and any known instrument/channel constraints.

Procedure:

1. Read `docs/ai-agents/parameter-setup-guide.md`.
2. Inspect the supplied configuration and every referenced `<config>` file.
3. Locate the relevant user documentation under `docs/source/user/`.
4. Locate the consuming task or recipe under `src/exosim/`.
5. Suggest the smallest safe edit and include a before/after XML or YAML snippet.
6. State observed facts, inferred assumptions, and open questions separately.

Failure modes:

- Do not guess missing instrument values.
- Do not remove units.
- Do not replace `__ConfigPath__` with machine-specific absolute paths unless the
  user explicitly asks for a local-only file.

## Pipeline Selection Agent

Purpose: map a user goal to the correct recipe, CLI command, and required files.

Procedure:

1. Use `docs/ai-agents/codebase-map.md` to identify candidate recipes.
2. Check `src/exosim/exosim.py` for current CLI options in this branch.
3. Point users to the nearest working example under `examples/`.
4. Provide a reproducible command and name the expected input/output files.

Quality check:

- If the user wants a later stage, verify that the required upstream HDF5 input is
  already available or explain how to create it.

## Scientific Code Review Agent

Purpose: review changes for scientific consistency and reproducibility.

Checklist:

- Units are explicit and preserved as `astropy.units.Quantity` where expected.
- Wavelength and time grids use documented units and ranges.
- Randomness is controlled through `random_seed` or `RunConfig` when reproducible
  output is requested.
- Signal transformations preserve expected `Signal` subclasses and HDF5 output
  structure.
- Parameter changes are documented in examples or `docs/source/user/` if they
  affect users.

Validation commands:

```bash
nox -s lint
nox -s test
nox -s docs
```

## Developer Maintenance Agent

Purpose: make code, test, and documentation changes in the ExoSim style.

Procedure:

1. Inspect the relevant source, tests, and docs before editing.
2. Follow the task pattern in `docs/source/contributing/tasks.rst` for new
   operations: subclass `Task`, declare inputs with `add_task_param()`, retrieve
   them with `get_task_param()`, and set outputs with `set_output()`.
3. Keep imports ordered as standard library, third-party, `exosim`, and local.
4. Add or update tests for changed behaviour.
5. Run focused tests first, then the relevant nox sessions.

Failure modes:

- Do not commit generated files from `examples/`, coverage reports, caches, or
  virtual environments.
- Do not silently change public CLI options without updating documentation.
