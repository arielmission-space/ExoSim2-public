"""Pure Python tool functions exposed by the ExoSim MCP server."""

from __future__ import annotations

import platform
import re
from typing import Any

from exosim.mcp.comparison import compare_configurations
from exosim.mcp.configuration import validate_configuration_file
from exosim.mcp.discovery import discovery_errors, record_discovery_error
from exosim.mcp.outputs import explain_output_file, inspect_output_file
from exosim.mcp.pipelines import (
    describe_pipeline_definition,
    dispatch_pipeline,
    list_pipeline_definitions,
)
from exosim.mcp.retrieval import knowledge_base_manifest, search_knowledge_sources
from exosim.mcp.schemas import resolve_repo_path

PROJECT_CONTEXT_FILES = (
    "AGENTS.md",
    "docs/ai-agents/codebase-map.md",
    "docs/ai-agents/parameter-setup-guide.md",
    "docs/ai-agents/agent-playbooks.md",
)


def exosim_project_context() -> dict[str, Any]:
    """Return the curated ExoSim guidance files for AI clients.

    Returns
    -------
    dict
        Guidance file contents keyed by repository-relative path.
    """
    root = resolve_repo_path(".")
    files = {}
    missing = []
    for relative_path in PROJECT_CONTEXT_FILES:
        path = root / relative_path
        if path.exists():
            files[relative_path] = path.read_text(encoding="utf-8")
        else:
            missing.append(relative_path)
    return {"files": files, "missing": missing, "capabilities": capabilities()}


def capabilities() -> dict[str, Any]:
    """Describe what this MCP server can do right now.

    Call this first: it reports the ExoSim and MCP SDK versions, the sandbox
    configuration, whether pipeline execution is enabled, the retrieval backend
    in use, and which tools are available — so the model does not attempt a
    disabled operation or guess at the environment.

    Returns
    -------
    dict
        ``versions``, ``sandbox``, ``execution_enabled``, ``retrieval``,
        ``pipelines``, ``discovery_errors`` (import failures met while finding
        the recipes, each with a fix hint; empty when healthy) and ``tools`` (grouped by always-available vs
        execution-only).
    """
    from importlib.metadata import (
        PackageNotFoundError,
        version as _pkg_version,
    )

    from exosim.mcp.retrieval import active_backend
    from exosim.mcp.sandbox import Sandbox

    def _version(package: str) -> str | None:
        try:
            return _pkg_version(package)
        except PackageNotFoundError:
            return None

    sandbox = Sandbox.from_env()
    embeddings_available = _version("sentence-transformers") is not None
    always_on = [
        "capabilities",
        "exosim_project_context",
        "knowledge_base_manifest",
        "search_knowledge_base",
        "customisation_guide",
        "validate_config",
        "compare_configs",
        "config_requirements",
        "draft_config",
        "list_pipelines",
        "describe_pipeline",
        "list_examples",
        "inspect_output",
        "explain_output",
        "suggest_next_steps",
        "diagnose_error",
    ]
    execution_only = [
        "run_pipeline",
        "get_job_status",
        "cancel_job",
        "list_jobs",
        "plot_output",
    ]

    return {
        "versions": {
            "exosim": _version("exosim"),
            "mcp_sdk": _version("mcp"),
            "sentence_transformers": _version("sentence-transformers"),
            "python": platform.python_version(),
        },
        "sandbox": sandbox.to_dict(),
        "execution_enabled": sandbox.allow_exec,
        "retrieval": {
            "backend": active_backend(),
            "embeddings_extra_installed": embeddings_available,
        },
        "pipelines": [definition["name"] for definition in list_pipeline_definitions()],
        "discovery_errors": discovery_errors(),
        "tools": {
            "always_available": always_on,
            "execution_only": execution_only if sandbox.allow_exec else [],
            "disabled": [] if sandbox.allow_exec else execution_only,
        },
    }


def search_knowledge_base(
    query: str,
    max_results: int = 5,
    max_chars: int = 0,
) -> dict[str, Any]:
    """Retrieve grounded passages for ExoSim questions.

    Parameters
    ----------
    query : str
        Natural-language query.
    max_results : int
        Maximum number of passages to return.
    max_chars : int
        Total character budget for the returned passage text (0 = no limit).
        The budget is split evenly across the passages; any passage cut short
        is marked ``truncated`` and keeps its ``path`` and line range so the
        full text can be fetched via the ``exosim://kb/<path>`` resource. Use
        this for clients with small context windows.

    Returns
    -------
    dict
        Retrieval result with explicit insufficient-evidence status.
    """
    sources = search_knowledge_sources(query, max_results=max_results)
    threshold = knowledge_base_manifest()["minimum_retrieval_score"]
    if not sources:
        return {
            "status": "insufficient_evidence",
            "grounded": False,
            "query": query,
            "sources": [],
            "minimum_retrieval_score": threshold,
            "message": (
                "No ExoSim source passage matched the query. Do not answer from "
                "general knowledge; tell the user the knowledge base has no "
                "evidence for this."
            ),
        }
    payload = [source.to_dict() for source in sources]
    truncated = False
    if max_chars and payload:
        per_source = max(200, max_chars // len(payload))
        for item in payload:
            text = item["text"]
            if len(text) > per_source:
                item["text"] = text[:per_source].rstrip() + " …"
                item["truncated"] = True
                item["full_text_chars"] = len(text)
                truncated = True
    return {
        "status": "ok",
        "grounded": True,
        "query": query,
        "sources": payload,
        "minimum_retrieval_score": threshold,
        "truncated": truncated,
    }


def knowledge_base_manifest_tool() -> dict[str, Any]:
    """Return approved retrieval sources and grounding policy metadata.

    Returns
    -------
    dict
        Knowledge-base manifest and grounding policy metadata.
    """
    return knowledge_base_manifest()


def validate_config(
    configuration: str,
) -> dict[str, Any]:
    """Validate an ExoSim XML, YAML or YML configuration file.

    Parameters
    ----------
    configuration : str
        Configuration file path.

    Returns
    -------
    dict
        Validation summary.
    """
    return validate_configuration_file(configuration)


def list_pipelines() -> dict[str, Any]:
    """List ExoSim pipelines exposed through the MCP layer.

    Returns
    -------
    dict
        Supported pipeline definitions, plus ``discovery_errors``: import
        failures met while discovering the recipes (empty when healthy). When it
        is not empty the list may be a hard-coded fallback and runs can fail.
    """
    return {
        "pipelines": list_pipeline_definitions(),
        "discovery_errors": discovery_errors(),
    }


def list_examples() -> dict[str, Any]:
    """List the bundled example configurations.

    Each entry has a short description, its ``<config>`` includes, whether it is
    a pipeline entry point, and which pipelines it suits. Use it to pick a real
    starting configuration instead of inventing one.

    Returns
    -------
    dict
        ``examples`` (list) and ``count``.
    """
    from exosim.mcp.examples import list_example_configs

    return list_example_configs()


def describe_pipeline(pipeline: str) -> dict[str, Any]:
    """Describe one pipeline and the parameters its recipe accepts.

    Use this before ``run_pipeline`` to see which parameters exist, their types,
    whether they are required, their defaults, and which ones the ``run_pipeline``
    arguments (``configuration``, ``output``, ``input_file``) fill.

    Parameters
    ----------
    pipeline : str
        Pipeline name from ``list_pipelines``.

    Returns
    -------
    dict
        The pipeline definition plus a ``parameters`` list.
    """
    try:
        return describe_pipeline_definition(pipeline)
    except ValueError as exc:
        return {"error": str(exc), "pipeline": pipeline}
    except ImportError as exc:
        entry = record_discovery_error(exc, "exosim.recipes")
        return {
            "error": f"Cannot import the recipe for '{pipeline}': {exc}",
            "pipeline": pipeline,
            "missing_module": entry["missing_module"],
            "hint": entry["hint"],
        }


def run_pipeline(
    pipeline: str,
    configuration: str,
    output: str,
    input_file: str | None = None,
    n_threads: int | None = None,
    chunk_size: float | None = None,
    overwrite: bool = False,
    plot: bool = False,
    wait_seconds: float = 30.0,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Run an ExoSim pipeline in a supervised background process.

    The configuration is read from the content root (or an allowed directory);
    the output is written **under the working directory** only. The tool is not
    registered when the server was started with ``--no-exec``. Running a
    configuration can execute the Python tasks it references, so execution is
    arbitrary code execution on the host.

    The call blocks for up to ``wait_seconds``. If the run finishes it returns
    the final status (with the log tail on failure); otherwise it returns
    ``status="running"`` and a ``job_id`` to poll with ``get_job_status``. A run
    is hard-killed after ``max_runtime_seconds`` (default: one hour).

    Parameters
    ----------
    pipeline : str
        Pipeline name.
    configuration : str
        Configuration file path. A relative path is looked up in the working
        directory first (drafts live there), then the content root; an absolute
        path must lie in an allowed directory.
    output : str
        Output file path, relative to the working directory.
    input_file : str, optional
        Input file for downstream pipelines. A relative path is looked up in the
        working directory first, then the content root; a missing file is
        rejected before anything starts.
    n_threads : int, optional
        Number of worker threads (clamped to the server's limit).
    chunk_size : float, optional
        HDF5 chunk size in MB.
    overwrite : bool
        Whether existing output files may be replaced.
    plot : bool
        Whether plotting should be enabled for supported recipes.
    wait_seconds : float
        How long to block before returning ``status="running"`` (clamped to 0-50
        seconds; poll longer runs with ``get_job_status``).
    max_runtime_seconds : float, optional
        Hard wall-clock limit for the run (positive; capped by the server limit
        ``EXOSIM_MCP_MAX_RUNTIME``).

    Returns
    -------
    dict
        Job status: ``succeeded`` / ``failed`` / ``running`` / ``timeout`` /
        ``cancelled``, with a ``reason`` for the non-success outcomes. When
        ``EXOSIM_MCP_MAX_JOBS`` runs are already active (default 2) nothing is
        started and the result is ``status="refused"`` with ``limit`` and
        ``running_jobs``; the same happens, with ``conflicting_job``, when a running
        job already writes ``output`` (``overwrite`` does not apply to it).
    """
    return dispatch_pipeline(
        pipeline,
        configuration,
        output,
        input_file=input_file,
        n_threads=n_threads,
        chunk_size=chunk_size,
        overwrite=overwrite,
        plot=plot,
        wait_seconds=wait_seconds,
        max_runtime_seconds=max_runtime_seconds,
    )


def get_job_status(job_id: str) -> dict[str, Any]:
    """Return the status of a background pipeline run.

    Parameters
    ----------
    job_id : str
        Identifier returned by ``run_pipeline``.

    Returns
    -------
    dict
        Job status, including the log tail once the run has finished.
    """
    from exosim.mcp.jobs import JOBS

    try:
        return JOBS.status(job_id)
    except KeyError as exc:
        return {"status": "unknown", "job_id": job_id, "error": str(exc.args[0])}


def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a running background pipeline run.

    Parameters
    ----------
    job_id : str
        Identifier returned by ``run_pipeline``.

    Returns
    -------
    dict
        Final job status.
    """
    from exosim.mcp.jobs import JOBS

    try:
        return JOBS.cancel(job_id)
    except KeyError as exc:
        return {"status": "unknown", "job_id": job_id, "error": str(exc.args[0])}


def list_jobs() -> dict[str, Any]:
    """List the background pipeline runs known to this server.

    Returns
    -------
    dict
        Every job seen since the server started, oldest first.
    """
    from exosim.mcp.jobs import JOBS

    return {"jobs": JOBS.list_jobs()}


def inspect_output(
    path: str,
    max_items: int = 80,
) -> dict[str, Any]:
    """Inspect an ExoSim HDF5, ECSV or CSV output.

    Parameters
    ----------
    path : str
        Output file path.
    max_items : int
        Maximum number of HDF5 nodes to report.

    Returns
    -------
    dict
        Output summary.
    """
    return inspect_output_file(path, max_items=max_items)


def explain_output(
    path: str,
    max_items: int = 200,
) -> dict[str, Any]:
    """Explain an ExoSim output file: structure plus the science of each node.

    Every recognised HDF5 group / dataset is annotated with what the numbers
    mean, which pipeline stage produced them and the doc page that explains
    them. Unrecognised nodes are returned under ``unexplained``.

    Parameters
    ----------
    path : str
        Output file path.
    max_items : int
        Maximum number of HDF5 nodes to walk.

    Returns
    -------
    dict
        ``pipeline``, ``explained`` and ``unexplained`` nodes.
    """
    return explain_output_file(path, max_items=max_items)


def plot_output(
    path: str,
    kind: str | None = None,
    time_step: int = 0,
    scale: str = "linear",
) -> dict[str, Any]:
    """Render figures from an ExoSim output file.

    Drives the same plotters as ``exosim-plot`` and writes PNGs under the
    working directory, for a vision-capable client to look at. The kind
    (``focalplane`` / ``radiometric`` / ``subexposures`` / ``ndrs``) is
    detected from the file layout when omitted.

    Parameters
    ----------
    path : str
        HDF5 output file path.
    kind : str, optional
        Plot kind; detected from the file when omitted.
    time_step : int
        Focal-plane time slice (``focalplane`` only).
    scale : str
        ``linear`` or ``dB`` (``focalplane`` only); anything else is an error.

    Returns
    -------
    dict
        ``kind``, ``output_dir`` and the ``figures`` written by this call.
    """
    from exosim.mcp.plotting import generate_plots

    return generate_plots(path, kind, time_step=time_step, scale=scale)


def customisation_guide(
    topic: str,
) -> dict[str, Any]:
    """Retrieve guidance for custom ExoSim tasks and pipelines.

    Parameters
    ----------
    topic : str
        Customisation topic to retrieve.

    Returns
    -------
    dict
        Retrieved customisation sources.
    """
    query = (
        f"{topic} custom task custom pipeline RunConfig recipe "
        "add_task_param get_task_param"
    )
    return search_knowledge_base(query, max_results=5)


def config_requirements(
    pipeline: str,
    configuration: str | None = None,
) -> dict[str, Any]:
    """Report which configuration sections a pipeline needs.

    The reference is the bundled example for that pipeline. If ``configuration``
    is given, also reports which sections are present, missing or extra, each
    with a documentation pointer — turning a "what do I still need?" question
    into a checklist.

    Parameters
    ----------
    pipeline : str
        Pipeline name (``focalplane``, ``radiometric``, ...).
    configuration : str, optional
        A configuration file to check against the reference.

    Returns
    -------
    dict
        ``reference_sections`` and, if a configuration was given,
        ``present`` / ``missing`` / ``extra``.
    """
    from exosim.mcp.configuration import config_requirements as _impl

    return _impl(pipeline, configuration)


def draft_config(
    pipeline: str,
    overrides: dict[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Draft a working configuration for a pipeline, then validate it.

    Starts from the bundled reference example, inlines its ``<config>``
    includes into one self-contained file, applies ``overrides``, writes the
    result under the working directory and loads it with the real ExoSim
    loader. Use ``config_requirements`` first to see the sections and keys.

    Parameters
    ----------
    pipeline : str
        Pipeline name (``focalplane``, ``radiometric``, ...).
    overrides : dict, optional
        ``{"section.subsection.key": "value"}`` — a value may carry a unit
        (``"1.1 m"``). Anything not found is reported under ``not_applied``
        with the keys that *are* available at that point. Path-like keys
        (``ConfigPath``, ``datafile``, ...) must stay inside the directories the
        server may read, and pointing a task at a ``.py`` file is refused when
        execution is disabled (``--no-exec``).
    overwrite : bool
        Whether an existing ``drafts/<pipeline>_draft.xml`` may be replaced.

    Returns
    -------
    dict
        ``draft`` (the file path), ``applied`` / ``not_applied``, and the
        validation outcome.
    """
    from exosim.mcp.authoring import draft_configuration

    return draft_configuration(pipeline, overrides, overwrite=overwrite)


def compare_configs(
    config1: str,
    config2: str,
) -> dict[str, Any]:
    """Compare two ExoSim configuration files.

    Parameters
    ----------
    config1 : str
        First configuration file path.
    config2 : str
        Second configuration file path.

    Returns
    -------
    dict
        Comparison showing differences, missing sections, and identical parts.
    """
    return compare_configurations(config1, config2)


def suggest_next_steps(
    context: str,
) -> dict[str, Any]:
    """Suggest next steps based on current workflow state.

    Parameters
    ----------
    context : str
        Current workflow context (e.g., 'config_validated', 'focalplane_complete').

    Returns
    -------
    dict
        Suggested next steps with commands and explanations.
    """
    suggestions = {
        "context": context,
        "suggestions": [],
    }

    context_lower = context.lower()

    if "config" in context_lower and "valid" in context_lower:
        suggestions["suggestions"] = [
            {
                "step": "run_focalplane",
                "command": "exosim focalplane -c config.xml -o focal.h5",
                "description": (
                    "Start with the focal plane pipeline to simulate the detector "
                    "layout and PSF."
                ),
                "required_files": ["config.xml"],
                "produces": ["focal.h5"],
            }
        ]

    elif "focalplane" in context_lower and "complete" in context_lower:
        suggestions["suggestions"] = [
            {
                "step": "run_subexposures",
                "command": (
                    "exosim subexposures -c config.xml -i focal.h5 -o subexp.h5"
                ),
                "description": (
                    "Simulate sub-exposures (pointing jitter, astronomical signal) "
                    "from the focal plane."
                ),
                "required_files": ["config.xml", "focal.h5"],
                "produces": ["subexp.h5"],
            },
            {
                "step": "run_radiometric",
                "command": "exosim radiometric -c config.xml -o radio.h5",
                "description": (
                    "Optional, independent branch: the analytic radiometric model "
                    "(photon counts and noise budget). It takes no input file."
                ),
                "required_files": ["config.xml"],
                "produces": ["radio.h5", "radiometric tables (.ecsv)"],
            },
        ]

    elif "radiometric" in context_lower and "complete" in context_lower:
        suggestions["suggestions"] = [
            {
                "step": "inspect_radiometric",
                "command": "plot_output('radio.h5')",
                "description": (
                    "The radiometric model is an independent branch and has no "
                    "downstream pipeline. Inspect or plot its tables, and run the "
                    "image pipeline (focalplane -> subexposures -> ndrs) separately "
                    "if a time-domain simulation is needed."
                ),
                "required_files": ["radio.h5"],
                "produces": ["figures"],
            }
        ]

    elif "subexposures" in context_lower and "complete" in context_lower:
        suggestions["suggestions"] = [
            {
                "step": "run_ndrs",
                "command": "exosim ndrs -c config.xml -i subexp.h5 -o ndrs.h5",
                "description": "Generate final non-destructive reads (NDRs).",
                "required_files": ["config.xml", "subexp.h5"],
                "produces": ["ndrs.h5"],
            }
        ]

    elif "ndrs" in context_lower and "complete" in context_lower:
        suggestions["suggestions"] = [
            {
                "step": "analyze_results",
                "command": "inspect_output('ndrs.h5')",
                "description": (
                    "Analyze the final output using ExoSim plotting tools or "
                    "custom analysis scripts."
                ),
                "required_files": ["ndrs.h5"],
                "produces": ["analysis results"],
            }
        ]

    elif "error" in context_lower:
        # Search for error solutions
        error_search = search_knowledge_base(
            "error troubleshooting common issues",
            max_results=3,
        )
        suggestions["suggestions"] = [
            {
                "step": "diagnose_error",
                "description": "Check the error message and logs for details.",
                "resources": error_search.get("sources", []),
            }
        ]
    else:
        # Generic suggestion
        suggestions["suggestions"] = [
            {
                "step": "validate_config",
                "command": "validate_config('config.xml')",
                "description": (
                    "Start by validating your configuration file to ensure it's correct."
                ),
            }
        ]

    return suggestions


def _parse_traceback(text: str) -> dict[str, str]:
    """Pull the exception type/message and the last frame out of a traceback."""
    lines = [line for line in text.splitlines() if line.strip()]
    info: dict[str, str] = {}
    for line in reversed(lines):
        m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning)):\s*(.*)$", line)
        if m:
            info["exception_type"] = m.group(1)
            info["exception_message"] = m.group(2).strip()
            break
    for line in reversed(lines):
        m = re.match(r'^\s*File "(.+?)", line (\d+), in (.+)$', line)
        if m:
            info["last_frame"] = f"{m.group(1)}:{m.group(2)} in {m.group(3)}"
            break
    return info


def diagnose_error(
    error_message: str,
    context: str = "",
    traceback_text: str = "",
) -> dict[str, Any]:
    """Diagnose an ExoSim error and suggest fixes.

    Parameters
    ----------
    error_message : str
        The error message or exception line.
    context : str, optional
        Additional context (e.g. which pipeline, configuration used).
    traceback_text : str, optional
        The full Python traceback or the tail of the ExoSim log. The exception
        type and the failing frame are extracted from it.

    Returns
    -------
    dict
        ``error_type``, a category, suggested fixes and related documentation.
    """
    diagnosis: dict[str, Any] = {
        "error_message": error_message,
        "context": context,
        "error_type": None,
        "category": "unknown",
        "suggestions": [],
        "related_docs": [],
    }

    parsed = _parse_traceback(traceback_text) if traceback_text else {}
    diagnosis.update(parsed)
    diagnosis["error_type"] = parsed.get("exception_type")

    error_lower = (error_message + " " + traceback_text).lower()

    # Common error patterns
    if "filenotfound" in error_lower or "no such file" in error_lower:
        diagnosis["category"] = "missing_file"
        diagnosis["suggestions"] = [
            "Check that all file paths in your configuration are correct.",
            "Verify that __ConfigPath__ is set correctly for relative paths.",
            "Ensure input files from previous pipeline steps exist.",
        ]

    elif "keyerror" in error_lower or "missing" in error_lower:
        diagnosis["category"] = "missing_parameter"
        diagnosis["suggestions"] = [
            "Check your configuration file for missing required sections.",
            "Compare with example configurations in the examples/ directory.",
            "Use validate_config() to see which sections are parsed correctly.",
        ]

    elif "valueerror" in error_lower or "invalid" in error_lower:
        diagnosis["category"] = "invalid_parameter"
        diagnosis["suggestions"] = [
            "Check parameter values and units in your configuration.",
            "Ensure numerical values are within physical ranges.",
            "Verify that units are specified correctly (e.g., 'um' for wavelength).",
        ]

    elif "memoryerror" in error_lower or "out of memory" in error_lower:
        diagnosis["category"] = "memory_issue"
        diagnosis["suggestions"] = [
            "Reduce the simulation resolution or time sampling.",
            "Use chunk_size parameter to reduce memory footprint.",
            "Consider running on a machine with more RAM.",
            "Enable slim_output mode to reduce intermediate data storage.",
        ]

    elif "importerror" in error_lower or "modulenotfound" in error_lower:
        diagnosis["category"] = "missing_dependency"
        diagnosis["suggestions"] = [
            "Install missing dependencies: pip install exosim[all]",
            "Check that all required packages are installed.",
            "Verify Python version compatibility (requires Python >=3.12).",
        ]

    else:
        diagnosis["category"] = "general_error"
        diagnosis["suggestions"] = [
            "Check the full stack trace for more details.",
            "Enable debug logging: exosim --debug ...",
            "Search for similar errors in the GitHub issues.",
        ]

    # Search knowledge base for related documentation
    search_query = f"{diagnosis['category']} error {context}"
    search_results = search_knowledge_base(
        search_query,
        max_results=3,
    )

    if search_results["status"] == "ok":
        diagnosis["related_docs"] = [
            {"path": s["path"], "score": s["score"]} for s in search_results["sources"]
        ]

    return diagnosis
