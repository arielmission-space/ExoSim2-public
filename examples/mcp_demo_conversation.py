"""A scripted walk-through of the ExoSim MCP tools.

It plays out a few user questions and shows the tool calls an AI assistant would
make to answer them with grounded evidence. Run it from anywhere::

    python examples/mcp_demo_conversation.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "EXOSIM_MCP_REPO_ROOT", str(Path(__file__).resolve().parent.parent)
)

from exosim.mcp.tools import (
    knowledge_base_manifest_tool,
    list_pipelines,
    search_knowledge_base,
    validate_config,
)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n {title}\n" + "=" * 72)


def turn(question: str) -> None:
    print(f"\nUser: {question}\nAssistant (via the exosim MCP tools):")


section("1. What can the assistant search?")
turn("What documentation can you use to answer my ExoSim questions?")
manifest = knowledge_base_manifest_tool()
print(
    f"  {manifest['matched_file_count']} approved files, backend "
    f"'{manifest['retrieval_backend']}'. Categories:"
)
for pattern in manifest["source_patterns"][:5]:
    print(f"    - {pattern['pattern']}: {pattern['purpose']}")
print(
    f"  Below score {manifest['minimum_retrieval_score']} I report insufficient "
    "evidence instead of guessing."
)

section("2. A grounded answer")
turn("How do I configure the radiometric model?")
result = search_knowledge_base(
    "how do I configure the radiometric model?", max_results=2
)
print(f"  status={result['status']}, grounded={result['grounded']}")
for source in result["sources"]:
    preview = source["text"][:140].replace("\n", " ").strip()
    print(
        f"    {source['path']}:{source['start_line']}-{source['end_line']} "
        f"(score {source['score']})\n      {preview}..."
    )

section("3. Which pipeline first?")
turn("Which ExoSim pipeline should I run first?")
for pipeline in list_pipelines()["pipelines"]:
    print(f"    {pipeline['name']:<13} {pipeline['cli_command']}")
print("  Run focalplane, then radiometric, subexposures, ndrs in sequence.")

section("4. Validating a configuration")
turn("Is examples/main_example.xml a valid configuration?")
validation = validate_config("examples/main_example.xml")
print(f"  valid={validation['valid']}")

section("5. When there is no evidence")
turn("Does ExoSim model the quxzzy frobnication of the blorptastic detector?")
miss = search_knowledge_base("quxzzy frobnication blorptastic detector", max_results=3)
print(
    f"  status={miss['status']} -> the assistant says it has no evidence, "
    "rather than inventing an answer."
)
