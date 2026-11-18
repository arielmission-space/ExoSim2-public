# ExoSim 2 codebase map for agents

Use this map to ground answers in repository evidence.

## Main source areas

- `src/exosim/exosim.py` — Rich Click CLI entry point. Defines commands such as
  `focalplane`, `radiometric`, `subexposures`, and `ndrs` and provides common
  options (`--configuration`, `--output`, `--nThreads`, `--debug`, `--logger`,
  `--plot`).
- `src/exosim/recipes/` — high-level pipelines:
  - `create_focal_plane.py`
  - `radiometric_model.py`
  - `create_sub_exposures.py`
  - `create_ndrs.py`
  - `simulate_observation.py`
- `src/exosim/tasks/` — task-based implementation. New operations normally
  inherit from `exosim.tasks.task.Task` and declare inputs with
  `add_task_param()`.
- `src/exosim/models/` — core models such as channels and signals.
- `src/exosim/output/` — HDF5 output backends and output helpers.
- `src/exosim/tools/` — standalone tools for maps and detector/readout helper
  products.
- `src/exosim/mcp/` — the optional Model Context Protocol server that wraps the
  ExoSim API for AI clients: `tools.py` (the tool functions), `server.py`
  (`exosim-mcp` entry point), `retrieval.py` (BM25 knowledge-base search),
  `sandbox.py` (path allowlist), `jobs.py` (background pipeline runs),
  `resources.py`, `prompts.py`. Installed with `pip install exosim[mcp]`;
  documented in `docs/source/user/mcp.rst`.

## Configuration and examples

- `src/exosim/tasks/load/load_options.py` — XML/YAML configuration loader.
- `examples/main_example.xml` — main focal-plane example with grids, pointing,
  jitter, a sky include, and a payload include.
- `examples/payload_example.xml` — payload include with telescope and channel
  includes.
- `examples/sky_example.xml` — source configuration example.
- `examples/photometer.xml` and `examples/spectrometer.xml` — channel examples.
- `examples/radiometric/main_radiometric.xml` and
  `examples/radiometric/sky_radiometric.xml` — radiometric examples.
- `examples/tools_input_example.xml` — tool input example.

## Documentation paths

- `README.md` — installation, documentation build, and contribution overview.
- `docs/source/user/quickstart.rst` — CLI quick start and command flags.
- `docs/source/user/focal_plane/` — focal-plane setup and pipeline docs.
- `docs/source/user/radiometric_model/` — radiometric model modes and steps.
- `docs/source/user/sub-exposures/` — sub-exposure and readout configuration.
- `docs/source/user/ndrs/` — NDR generation documentation.
- `docs/source/user/tools/` — standalone tool documentation.
- `docs/source/contributing/tasks.rst` — task structure and `Task` API.
- `docs/source/contributing/runConfig.rst` — `RunConfig` settings for threads,
  chunk size and random seed.

## Validation paths

- `noxfile.py` — automation sessions: `lint`, `test`, `coverage`, `docs`,
  `format`, `check-agent-docs`, `check-wheel-assets`, `retrieval-eval`,
  `changelog-fragment`, `release`.
- `pyproject.toml` — Python requirement, dependencies, `setuptools-scm`
  versioning, Ruff and pytest settings, the `[tool.scriv]` changelog config.
- `tests/` — regression and unit tests. Tests use example XML configurations.
- `.github/workflows/ci_pipeline.yml` — lint + `tests/unit`, a `mcp` job
  (`--extra mcp`), an `agent-docs` job, a `wheel-assets` job.
- `scripts/check_agent_docs.py` — fails if these guides reference a path that
  no longer exists.

## Release and tooling

- Packaging and environments use **uv**; the version is derived from git tags by
  `setuptools-scm` (no version string in the repo).
- Changelog entries are hand-written fragments in `changelog.d/`; `nox -s
  release` merges them and drives the tag-triggered release pipeline
  (`.github/workflows/release.yml`). See
  `docs/source/contributing/releasing.rst`.
- Logging uses `structlog`.
