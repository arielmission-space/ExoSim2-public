.. _agentic_support:

================
Agentic support
================

ExoSim 2 includes repository-local guidance for AI coding assistants and other
large language model (LLM) tools. This support is designed to help users and
maintainers get grounded answers about the codebase, choose the right pipeline
entry point, and set simulation parameters from repository evidence.

The static guidance is plain text stored in the repository. It adds no runtime
dependency and is not tied to a single AI provider. The :doc:`MCP server
<user/mcp>` is the runtime counterpart: it lets an AI client query this same
material through tools, validate configurations and run pipelines. See its
:ref:`per-client setup <mcp-per-client-setup>` to connect Claude Desktop, Claude
Code, VS Code, Cursor or opencode.

The shared guide
================

``AGENTS.md``
    The canonical shared guide (the `agents.md <https://agents.md>`__ standard).
    It defines the first-response protocol, project facts, common commands,
    quality expectations, and links to the detailed guides. Any repository-aware
    agent can read it; point tools that do not pick it up automatically at this
    file.

What the support includes
=========================

The detailed guides live in ``docs/ai-agents/``:

``docs/ai-agents/codebase-map.md``
    A concise map of the package layout, recipes, CLI, configuration loader,
    examples, documentation paths, and validation commands.

``docs/ai-agents/parameter-setup-guide.md``
    A parameter-focused guide that explains XML/YAML loading, ``__ConfigPath__``,
    ``<config>`` includes, Astropy units, data files, runtime settings, and the
    recommended workflow for answering configuration questions.

``docs/ai-agents/agent-playbooks.md``
    Reusable roles that any model can adopt, including a parameter setup agent,
    pipeline selection agent, scientific code review agent, and developer
    maintenance agent.

``docs/ai-agents/README.md``
    The index for these files.

The source tree also keeps developer notes about the MCP server itself. They are
implementation notes, not agent guidance, and are kept out of the MCP knowledge
base and the published documentation.

How this helps users
====================

These files help an LLM answer ExoSim questions with local evidence instead of
guesswork. In particular, they instruct agents to:

* inspect ``examples/`` and ``docs/source/user/`` before suggesting parameter
  edits;
* verify parser behaviour in ``src/exosim/tasks/load/load_options.py``;
* preserve ``__ConfigPath__`` and physical units in XML/YAML examples;
* identify the right recipe or CLI command for focal-plane creation,
  radiometric modelling, sub-exposures, and NDR creation;
* separate observed facts, assumptions, and open questions when scientific input
  is missing;
* avoid committing generated HDF5 files, plots, caches, or virtual environments.

How maintainers should update it
================================

Keep the agent guidance synchronised with code and documentation changes:

* update ``docs/ai-agents/parameter-setup-guide.md`` when configuration parsing
  or parameter behaviour changes;
* update ``docs/ai-agents/codebase-map.md`` when recipes, CLI commands, examples,
  or validation commands change;
* update ``docs/ai-agents/agent-playbooks.md`` when repeated support workflows
  appear;
* run ``nox -s check-agent-docs`` (also a CI job) so a moved file is caught
  before the guidance goes stale.
