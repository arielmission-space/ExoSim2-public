# AI agent guidance for ExoSim 2

This directory contains tool-neutral guidance that repository-aware AI agents can
use to answer user questions and help set up ExoSim parameters.

`../AGENTS.md` is the canonical shared guide (the [agents.md](https://agents.md)
standard). Any repository-aware model can read it; agents that do not pick it up
automatically should be pointed at it.

Core guides (an agent answering ExoSim questions should read these):

- `codebase-map.md` — repository structure and high-value source/documentation
  paths.
- `parameter-setup-guide.md` — how ExoSim configuration is parsed and how to
  help users edit parameters safely.
- `agent-playbooks.md` — reusable specialist roles for Q&A, setup, review, and
  maintenance.

The MCP server (`src/exosim/mcp/`, `docs/source/user/mcp.rst`) is the runtime
way for an AI client to query ExoSim. Its knowledge base is the three core
guides above plus `AGENTS.md` and the user documentation. Developer notes about
the server's internals, which live only in the source tree, are deliberately kept
out of it.
