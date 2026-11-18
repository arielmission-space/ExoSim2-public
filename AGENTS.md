# ExoSim 2 Agent Guide

This file is the shared entry point for coding agents and AI assistants working
in this repository, following the [agents.md](https://agents.md) standard. It is
tool-neutral: any repository-aware model should read it.

## First response protocol

- Work from repository evidence. Inspect the relevant files before changing code,
  parameters or documentation.
- For user parameter questions, start with `docs/ai-agents/parameter-setup-guide.md`
  and the XML/YAML examples under `examples/`.
- For code changes, start with `docs/ai-agents/codebase-map.md` and the relevant
  specialised playbook in `docs/ai-agents/agent-playbooks.md`.
- Do not invent scientific defaults, units, detector behaviour, observing modes
  or citations. If a value is absent from the code, examples or documentation,
  say so and ask for the missing science input.
- Prefer reproducible commands and file paths over generic advice.
- Use British English in documentation and comments.

## Project facts to keep in context

- ExoSim 2 is a Python 3.12+ exoplanet observation simulator.
- Package source lives in `src/exosim/`; tests live in `tests/`; user examples
  and working configuration files live in `examples/`.
- High-level simulation entry points are recipes in `src/exosim/recipes/` and CLI
  commands in `src/exosim/exosim.py`.
- The main pipeline stages are focal-plane creation, radiometric modelling,
  sub-exposures and NDR creation.
- Configuration is parsed by `src/exosim/tasks/load/load_options.py`. XML is the
  dominant documented format; YAML/YML is also supported by the loader.
- Configuration files may include other files with `<config>...</config>` and
  use `__ConfigPath__` to resolve paths relative to the configured root.
- Numerical values with a `unit` attribute become `astropy.units.Quantity`
  objects. Preserve units when editing examples or documentation.
- Shared runtime settings use `exosim.utils.RunConfig`, including thread count,
  chunk size and random seed.

## Common commands

```bash
# Install for development
uv sync --extra dev,docs

# Run quality checks and tests
nox -s lint
nox -s test
nox -s coverage

# Build documentation
nox -s docs

# Useful CLI help
exosim --help
exosim focalplane --help
exosim radiometric --help
exosim subexposures --help
exosim ndrs --help
```

## Agent playbooks

Use these roles as needed; they are written as generic Markdown so any model can
adopt them without a tool-specific agent runtime.

- `docs/ai-agents/agent-playbooks.md#parameter-setup-agent` for helping users
  choose and edit configuration parameters.
- `docs/ai-agents/agent-playbooks.md#pipeline-selection-agent` for choosing the
  appropriate ExoSim recipe or CLI command.
- `docs/ai-agents/agent-playbooks.md#scientific-code-review-agent` for checking
  units, signal flow and reproducibility.
- `docs/ai-agents/agent-playbooks.md#developer-maintenance-agent` for code,
  tests, linting and documentation changes.

## Quality bar

- Explain observed facts, assumptions and open questions separately when the
  science or configuration is ambiguous.
- Validate parameter edits against examples and loader behaviour before claiming
  they will run.
- Keep generated examples small and traceable to files in `examples/` or
  `docs/source/user/`.
- Do not commit generated HDF5, plot, coverage, virtual-environment or cache
  artefacts.
