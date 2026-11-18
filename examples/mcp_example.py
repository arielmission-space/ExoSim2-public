"""Example: using the ExoSim MCP tools programmatically.

Run this from the repository root (the MCP tools read their sandbox from the
environment; ``EXOSIM_MCP_REPO_ROOT`` defaults to the current directory)::

    python examples/mcp_example.py

For AI-client integration, start the server with ``exosim-mcp`` instead and
connect an MCP client -- see ``docs/source/user/mcp.rst``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Serve this checkout regardless of the current working directory.
os.environ.setdefault(
    "EXOSIM_MCP_REPO_ROOT", str(Path(__file__).resolve().parent.parent)
)

from exosim.mcp.tools import (
    customisation_guide,
    exosim_project_context,
    knowledge_base_manifest_tool,
    list_pipelines,
    search_knowledge_base,
    validate_config,
)

# 1. Curated agent-guidance files.
context = exosim_project_context()
print("Guidance files:", ", ".join(context["files"]))

# 2. The approved retrieval sources.
manifest = knowledge_base_manifest_tool()
print(
    f"Knowledge base: {manifest['matched_file_count']} files, "
    f"backend {manifest['retrieval_backend']}, "
    f"min score {manifest['minimum_retrieval_score']}"
)

# 3. Grounded question answering.
result = search_knowledge_base("how do I set up the radiometric model?", max_results=3)
print(f"\nQuery status: {result['status']} (grounded={result['grounded']})")
for source in result["sources"]:
    print(
        f"  {source['path']}:{source['start_line']}-{source['end_line']} "
        f"(score {source['score']})"
    )

# 4. Validate a configuration with the real ExoSim loader.
validation = validate_config("examples/main_example.xml")
print(f"\nexamples/main_example.xml valid: {validation['valid']}")

# 5. Available pipelines.
print("\nPipelines:")
for pipeline in list_pipelines()["pipelines"]:
    print(f"  {pipeline['name']} -> {pipeline['cli_command']}")

# 6. Developer guidance.
guide = customisation_guide("write a custom task")
print(f"\nCustomisation guidance: {guide['status']}, {len(guide['sources'])} sources")

# 7. Running a pipeline (commented out -- writes an output file).
# from exosim.mcp.tools import run_pipeline, get_job_status
# job = run_pipeline("focalplane", "examples/main_example.xml", "focal.h5")
# print(job)                       # {"status": "running", "job_id": ...} or final status
# print(get_job_status(job["job_id"]))
