"""MCP prompts: ready-made instructions for common ExoSim workflows.

Each prompt returns text the client can drop into a conversation. They tell the
model which ExoSim MCP tools to use and in what order, keeping answers grounded.
"""

from __future__ import annotations

_GROUNDING = (
    "Use the ExoSim MCP tools for every factual claim. Cite the file path and "
    "line range from search_knowledge_base or the manifest. If a tool returns "
    "insufficient_evidence, say so instead of guessing."
)


def set_up_channel(instrument: str = "spectrometer") -> str:
    """Guide the user through configuring a payload channel."""
    return (
        f"Help me add a {instrument} channel to an ExoSim configuration.\n\n"
        "1. search_knowledge_base for the channel and focal-plane parameters "
        "and for a matching example under examples/.\n"
        "2. Walk me through each required parameter with its units, quoting the "
        "docs.\n"
        "3. When I have a file, call validate_config on it and explain any "
        "errors.\n\n"
        f"{_GROUNDING}"
    )


def diagnose_failed_run(pipeline: str = "radiometric") -> str:
    """Guide the user through diagnosing a failed pipeline run."""
    return (
        f"My ExoSim {pipeline} run failed. Help me fix it.\n\n"
        "1. Ask me for the error message or the tail of the log.\n"
        "2. Call diagnose_error with that text.\n"
        "3. Call validate_config on my configuration.\n"
        "4. search_knowledge_base for the failing component and propose a fix, "
        "quoting the docs or source.\n\n"
        f"{_GROUNDING}"
    )


def explain_output(path: str = "output.h5") -> str:
    """Guide the user through understanding a generated output file."""
    return (
        f"Explain the ExoSim output file {path}.\n\n"
        "1. Call inspect_output on it.\n"
        "2. For each top-level group, search_knowledge_base for what it "
        "contains and how it is produced.\n"
        "3. Summarise the datasets, their units and the pipeline stage that "
        "wrote them.\n\n"
        f"{_GROUNDING}"
    )


def new_simulation(pipeline: str = "focalplane") -> str:
    """Guide the user from nothing to a running simulation."""
    return (
        f"Help me set up and run an ExoSim {pipeline} simulation from scratch.\n\n"
        "1. Call config_requirements for the pipeline to see the sections and "
        "keys it needs, and list_examples for a real starting point.\n"
        "2. Ask me what I want to change from the bundled example (target, "
        "wavelength range, instrument, timing).\n"
        "3. Call draft_config with those changes as overrides; show me the "
        "applied and not_applied lists and fix the paths that missed.\n"
        "4. When draft_config reports valid, call run_pipeline on the drafted "
        "file and poll get_job_status.\n"
        "5. On success, call explain_output on the result.\n\n"
        f"{_GROUNDING}"
    )


ALL_PROMPTS = (set_up_channel, diagnose_failed_run, explain_output, new_simulation)
