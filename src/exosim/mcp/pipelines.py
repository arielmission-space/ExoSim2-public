"""Pipeline dispatch helpers for the ExoSim MCP layer."""

from __future__ import annotations

import importlib
import inspect
import re
from typing import TYPE_CHECKING, Any

from exosim.mcp.discovery import discover_recipes, record_discovery_error
from exosim.mcp.schemas import PipelineDefinition

if TYPE_CHECKING:
    from pathlib import Path

# Hardcoded fallback definitions for backward compatibility
# These are used if dynamic discovery fails
_FALLBACK_PIPELINES = {
    "focalplane": PipelineDefinition(
        name="focalplane",
        recipe="exosim.recipes.CreateFocalPlane",
        cli_command="exosim focalplane",
        required_inputs=("configuration", "output"),
        output_description="HDF5 focal-plane output.",
    ),
    "radiometric": PipelineDefinition(
        name="radiometric",
        recipe="exosim.recipes.RadiometricModel",
        cli_command="exosim radiometric",
        required_inputs=("configuration", "output"),
        output_description="HDF5 output plus radiometric ECSV tables.",
    ),
    "subexposures": PipelineDefinition(
        name="subexposures",
        recipe="exosim.recipes.CreateSubExposures",
        cli_command="exosim subexposures",
        required_inputs=("configuration", "input", "output"),
        output_description="HDF5 sub-exposure output.",
    ),
    "ndrs": PipelineDefinition(
        name="ndrs",
        recipe="exosim.recipes.CreateNDRs",
        cli_command="exosim ndrs",
        required_inputs=("configuration", "input", "output"),
        output_description="HDF5 NDR output.",
    ),
}

# A client's tool call usually times out after about a minute: never block that
# long. Longer runs are polled with get_job_status.
MAX_WAIT_SECONDS = 50.0

# Dynamically discover pipelines from exosim.recipes
# Falls back to hardcoded definitions if discovery fails
try:
    PIPELINES = discover_recipes()
    # If discovery succeeded but returned empty, use fallback
    if not PIPELINES:
        PIPELINES = _FALLBACK_PIPELINES
except Exception as _exc:
    # If discovery failed for any reason, use fallback but keep the cause so
    # that capabilities() / list_pipelines() can report it.
    record_discovery_error(_exc, "exosim.recipes")
    PIPELINES = _FALLBACK_PIPELINES


def list_pipeline_definitions() -> list[dict[str, Any]]:
    """Return the supported pipeline definitions.

    Returns
    -------
    list[dict]
        Serialisable pipeline descriptions.
    """
    return [definition.to_dict() for definition in PIPELINES.values()]


def _numpydoc_params(doc: str | None) -> dict[str, str]:
    """Parse the ``Parameters`` (or ``Attributes``) section of a numpydoc string."""
    if not doc:
        return {}
    lines = doc.splitlines()
    out: dict[str, str] = {}
    current: str | None = None
    in_section = False
    skip_underline = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line in {"Parameters", "Attributes"}:
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            in_section = bool(nxt) and set(nxt) <= {"-"}
            skip_underline = in_section
            current = None
            continue
        if not in_section:
            continue
        if skip_underline:
            skip_underline = False
            continue
        if line and set(line) <= {"-"}:  # underline of the following section title
            in_section = False
            continue
        header = re.match(r"^(\w+)\s*:\s*\S.*$", line)
        if header:
            current = header.group(1)
            out.setdefault(current, "")
        elif current and line:
            out[current] = (out[current] + " " + line).strip()
    return out


def describe_pipeline_definition(pipeline: str) -> dict[str, Any]:
    """Describe a pipeline and the parameters its recipe accepts.

    Parameters
    ----------
    pipeline : str
        Pipeline name from :func:`list_pipeline_definitions`.

    Returns
    -------
    dict
        The pipeline definition plus a ``parameters`` list (name, type,
        required, default, description) and the MCP argument that fills each.
    """
    key = pipeline.lower()
    if key not in PIPELINES:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    definition = PIPELINES[key]
    module_name, class_name = definition.recipe.rsplit(".", 1)
    recipe_class = getattr(importlib.import_module(module_name), class_name)

    descriptions = _numpydoc_params(recipe_class.__doc__)
    descriptions |= _numpydoc_params(recipe_class.__init__.__doc__)
    mcp_arg = {
        "options_file": "configuration",
        "configuration": "configuration",
        "output_file": "output",
        "output": "output",
        "input_file": "input_file",
        "input": "input_file",
    }

    parameters = []
    for name, param in inspect.signature(recipe_class.__init__).parameters.items():
        if name == "self" or param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        has_default = param.default is not inspect.Parameter.empty
        annotation = param.annotation
        parameters.append(
            {
                "name": name,
                "type": (
                    annotation
                    if isinstance(annotation, str)
                    else getattr(annotation, "__name__", str(annotation))
                )
                if annotation is not inspect.Parameter.empty
                else None,
                "required": not has_default,
                "default": param.default if has_default else None,
                "description": descriptions.get(name, ""),
                "mcp_argument": mcp_arg.get(name),
            }
        )

    data = definition.to_dict()
    first_para = (recipe_class.__doc__ or "").strip().split("\n\n", 1)[0]
    data["summary"] = " ".join(first_para.split())
    data["parameters"] = parameters
    return data


def _map_recipe_kwargs(
    recipe_class: type,
    *,
    configuration: str,
    output: str,
    input_file: str | None,
    plot: bool,
) -> dict[str, Any]:
    """Map MCP argument names onto whatever the recipe constructor accepts.

    ExoSim recipes use ``options_file`` / ``output_file`` / ``input_file``; a
    recipe that instead takes ``**kwargs`` (or a mock in the tests) is assumed to
    follow the same convention.
    """
    params = inspect.signature(recipe_class.__init__).parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    def pick(*names: str) -> str | None:
        for name in names:
            if name in params:
                return name
        return names[0] if accepts_kwargs else None

    kwargs: dict[str, Any] = {}
    if name := pick("options_file", "configuration"):
        kwargs[name] = configuration
    if name := pick("output_file", "output"):
        kwargs[name] = output
    if input_file is not None and (name := pick("input_file", "input")):
        kwargs[name] = input_file
    if plot and ("plot" in params or accepts_kwargs):
        kwargs["plot"] = plot
    return kwargs


def run_recipe(spec: dict[str, Any]) -> None:
    """Instantiate and run one recipe from a resolved spec.

    This is the unit of work executed by :mod:`exosim.mcp.jobs` in a subprocess.
    It runs from the configuration file's directory (so relative ``<config>``
    includes resolve), and sets/restores the requested ``RunConfig`` values so
    an inline run does not leak state.
    """
    import os
    from pathlib import Path

    from exosim.utils import RunConfig

    saved = (RunConfig.n_job, RunConfig.chunk_size)
    saved_cwd = os.getcwd()
    try:
        if spec.get("configuration"):
            os.chdir(Path(spec["configuration"]).parent)
        if spec.get("n_threads") is not None:
            RunConfig.n_job = spec["n_threads"]
        if spec.get("chunk_size") is not None:
            RunConfig.chunk_size = spec["chunk_size"]
        module_name, class_name = spec["recipe"].rsplit(".", 1)
        recipe_class = getattr(importlib.import_module(module_name), class_name)
        recipe_class(**spec["recipe_kwargs"])
    finally:
        os.chdir(saved_cwd)
        RunConfig.n_job, RunConfig.chunk_size = saved


def _resolve_spec(
    pipeline: str,
    configuration: str,
    output: str,
    *,
    input_file: str | None,
    repo_root: str | Path | None,
    work_dir: str | Path | None,
    n_threads: int | None,
    chunk_size: float | None,
    overwrite: bool,
    plot: bool,
) -> tuple[Any, dict[str, Any]]:
    """Validate inputs against the sandbox and build a serialisable run spec."""
    from exosim.mcp.sandbox import Sandbox

    pipeline_key = pipeline.lower()
    if pipeline_key not in PIPELINES:
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    sandbox = Sandbox.from_env(repo_root=repo_root, work_dir=work_dir)
    if not sandbox.allow_exec:
        raise PermissionError(
            "Pipeline execution is disabled on this MCP server "
            "(started with --no-exec / EXOSIM_MCP_NO_EXEC). "
            "validate_config, list_pipelines and search_knowledge_base remain available."
        )

    # Drafts written by draft_config live in the work dir: look there first.
    configuration_path = sandbox.resolve_read(configuration, prefer_work_dir=True)
    output_path = sandbox.resolve_write(output)
    if not configuration_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Set overwrite=True to replace it."
        )

    pipeline_def = PIPELINES[pipeline_key]
    if "input" in pipeline_def.required_inputs and input_file is None:
        raise ValueError(f"input_file is required for the {pipeline_key} pipeline.")

    input_path = None
    if input_file is not None:
        # Outputs of earlier runs live in the work dir: look there first.
        input_path = sandbox.resolve_read(input_file, prefer_work_dir=True)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {input_path} (given as '{input_file}'; "
                "relative paths are tried under the working directory "
                f"{sandbox.work_dir}, then the content root {sandbox.repo_root})"
            )
    module_name, class_name = pipeline_def.recipe.rsplit(".", 1)
    recipe_class = getattr(importlib.import_module(module_name), class_name)

    # The output may be in a not-yet-existing subdirectory (already confined to
    # the work dir by resolve_write); the recipe would fail after starting.
    sandbox.ensure_work_dir()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = {
        "pipeline": pipeline_key,
        "recipe": pipeline_def.recipe,
        "recipe_kwargs": _map_recipe_kwargs(
            recipe_class,
            configuration=str(configuration_path),
            output=str(output_path),
            input_file=str(input_path) if input_path is not None else None,
            plot=plot,
        ),
        "configuration": str(configuration_path),
        "input": str(input_path) if input_path is not None else None,
        "output": str(output_path),
        "n_threads": sandbox.clamp_threads(n_threads),
        "chunk_size": chunk_size,
        "plot": plot,
    }
    return sandbox, spec


def _effective_runtime(requested: float | None, cap: float | None) -> float | None:
    """Combine a client-requested deadline with the server-side runtime cap.

    ``None`` keeps the server default; a request can only shorten the cap, never
    lift it (``0`` does not mean "unlimited").
    """
    if requested is None:
        return cap
    if requested <= 0:
        raise ValueError(
            "max_runtime_seconds must be positive (got "
            f"{requested}); the server-side limit is set with EXOSIM_MCP_MAX_RUNTIME."
        )
    if cap is not None and cap > 0:
        return min(requested, cap)
    return requested


def dispatch_pipeline(
    pipeline: str,
    configuration: str,
    output: str,
    *,
    input_file: str | None = None,
    repo_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    n_threads: int | None = None,
    chunk_size: float | None = None,
    overwrite: bool = False,
    plot: bool = False,
    wait_seconds: float = 30.0,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Run an ExoSim pipeline in a supervised subprocess.

    All paths are confined to the MCP sandbox: the configuration and input are
    read from the checkout (or an allowed directory), and the output is written
    under the working directory. Execution is refused when the server was
    started with ``--no-exec`` / ``EXOSIM_MCP_NO_EXEC``.

    The run happens in a separate process. This call blocks for up to
    ``wait_seconds``; if the run finishes it returns the final status (with the
    log tail on failure), otherwise it returns ``status="running"`` and a
    ``job_id`` to poll with ``get_job_status``. A run is hard-killed after
    ``max_runtime_seconds`` (default: ``EXOSIM_MCP_MAX_RUNTIME`` or one hour).

    Parameters
    ----------
    pipeline, configuration, output, input_file : str
        Pipeline name and sandboxed paths (``configuration`` / ``input_file``
        are read, ``output`` is written under the working directory).
    repo_root, work_dir : str or pathlib.Path, optional
        Sandbox roots (server configuration; not exposed to MCP clients).
    n_threads : int, optional
        Worker threads, clamped to ``EXOSIM_MCP_MAX_THREADS``.
    chunk_size : float, optional
        HDF5 chunk size in MB for downstream pipelines.
    overwrite : bool
        Whether an existing output file may be overwritten.
    plot : bool
        Whether recipes supporting plot generation should enable plotting.
    wait_seconds : float
        How long to block for the run before returning ``status="running"``
        (clamped to ``0 <= wait_seconds <= 50``; poll longer runs with
        ``get_job_status``).
    max_runtime_seconds : float, optional
        Hard wall-clock limit; defaults to the sandbox limit.

    Returns
    -------
    dict
        Job status (``succeeded`` / ``failed`` / ``running`` / ``timeout``).
    """
    sandbox, spec = _resolve_spec(
        pipeline,
        configuration,
        output,
        input_file=input_file,
        repo_root=repo_root,
        work_dir=work_dir,
        n_threads=n_threads,
        chunk_size=chunk_size,
        overwrite=overwrite,
        plot=plot,
    )

    if sandbox.exec_backend == "inline":
        run_recipe(spec)
        return {
            "status": "succeeded",
            "pipeline": spec["pipeline"],
            "configuration": spec["configuration"],
            "input": spec["input"],
            "output": spec["output"],
        }

    from exosim.mcp import jobs

    max_runtime = _effective_runtime(max_runtime_seconds, sandbox.max_runtime)
    try:
        job_id = jobs.JOBS.submit(
            spec, log_dir=sandbox.log_dir(), max_runtime=max_runtime
        )
    except jobs.JobConflictError as exc:
        return {
            "status": "refused",
            "error": str(exc),
            "conflicting_job": exc.job_id,
            "output": spec["output"],
            "pipeline": spec["pipeline"],
        }
    except jobs.JobLimitError as exc:
        return {
            "status": "refused",
            "error": str(exc),
            "limit": exc.limit,
            "running_jobs": exc.running,
            "pipeline": spec["pipeline"],
        }
    result = jobs.JOBS.wait(
        job_id, min(max(float(wait_seconds), 0.0), MAX_WAIT_SECONDS)
    )
    result.update(configuration=spec["configuration"], input=spec["input"])
    return result
