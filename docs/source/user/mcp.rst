======================
MCP server
======================

ExoSim ships an optional **Model Context Protocol (MCP)** server. MCP is an open
standard for connecting AI assistants to external tools and data; through it an
MCP-compatible client (a desktop app, an IDE extension, an orchestrator) can:

- ask questions about ExoSim grounded in its documentation and source code,
  instead of guessing;
- validate configuration files with the real ExoSim loader;
- run simulation pipelines as supervised background jobs (see `Safety model`_);
- inspect output files and get customisation guidance.

It is an **access layer around the existing ExoSim API and CLI**. It adds no
scientific capability and does not replace the Python API or ``exosim`` command.

The server is the runtime half of ExoSim's :ref:`agentic_support`; the static
half is ``AGENTS.md`` and the guides under ``docs/ai-agents/``, which the server
also serves through ``exosim_project_context()``.

.. contents::
   :local:
   :depth: 1


Quickstart
==========

#. Install the extra. The install is self-contained: the wheel ships the
   knowledge base, the example configurations and the reference configurations
   (142 files in ``exosim/mcp/_data``: 2.06 MB unpacked, 0.79 MB compressed, in
   a wheel of 2.6 MB), so no ExoSim checkout is needed:

   .. code-block:: bash

       pip install "exosim[mcp]"

#. Check that the SDK imports and the server builds (``exosim-mcp --help`` is
   not enough: it works without the ``mcp`` SDK):

   .. code-block:: bash

       python -c "from exosim.mcp.server import build_server; build_server(); print('ok')"

   A missing or unusable SDK stops with an ``ImportError`` that names the
   installed ``mcp`` version (if any) and carries the original error text.

#. Tell your MCP client to launch it. Most clients read a JSON file listing MCP
   servers; add an ``exosim`` entry:

   .. code-block:: json

       {
         "mcpServers": {
           "exosim": {
             "command": "exosim-mcp",
             "args": ["--work-dir", "/path/to/exosim-runs"]
           }
         }
       }

   ``--work-dir`` is where simulation outputs, drafts, plots and logs go. Without
   it the server uses ``<checkout>/mcp-runs`` when it serves a checkout and
   ``./mcp-runs`` in *its* working directory when it serves the packaged
   content. A desktop client may start the server from ``/``, where that is
   unwritable: the server then falls back to ``~/.exosim/mcp-runs`` with one
   warning line on stderr. An explicit ``--work-dir`` (or ``EXOSIM_MCP_WORK_DIR``)
   that cannot be created is an error: the server prints one line and exits with
   a non-zero status. With ``--no-exec`` the directory is not needed at startup
   and is created when a tool needs it (``draft_config``).

#. Ask your assistant something like *"Using the exosim tools, how do I enable
   the radiometric model?"*. It should call ``search_knowledge_base`` and answer
   with file-and-line citations.

For a question-only server that cannot run simulations, add ``--no-exec``. Read
`Safety model`_ before enabling execution: it lets the assistant run code on your
machine.


Connecting a client
===================

The JSON above is the common shape. What varies between clients is the **file it
lives in** and the **top-level key** (``mcpServers``, ``servers``, ``mcp`` …).
The command is always ``exosim-mcp``; the recipes below fill in the rest.

Making ``exosim-mcp`` findable
------------------------------------------

A client launches ``exosim-mcp`` as a subprocess, so it has to be on the path it
runs with. If

.. code-block:: bash

    which exosim-mcp

prints nothing — usually because ExoSim lives in a project virtual environment —
give the client the **absolute path** to the console script instead of the bare
name, for example ``/path/to/ExoSim2.0/.venv/bin/exosim-mcp``.
``python -c "import exosim.mcp"`` in that environment should succeed.

Where the content comes from
----------------------------

A stdio client starts the server with an arbitrary working directory. The server
picks its **content root** (the documentation, examples and reference
configurations it answers from) in this order:

#. ``--repo-root`` / ``EXOSIM_MCP_REPO_ROOT``, if given (an override);
#. the current directory, if it is an ExoSim checkout, then the checkout the
   ``exosim`` package was imported from (an editable install);
#. the content packaged inside the wheel.

So ``--repo-root`` is only needed to serve a checkout other than the one you
installed from. ``capabilities()["sandbox"]["content"]`` reports which was
chosen: ``source`` (``override`` / ``checkout`` / ``packaged``), ``root``,
``knowledge_base_files``, ``example_configs`` and ``warnings``. An empty
knowledge base is also warned about on stderr at startup. The root doubles as the
read-only sandbox root; see `Safety model`_.

.. _mcp-per-client-setup:

Per-client setup
----------------

Each recipe adds one ``exosim`` server. Restart or reload the client afterwards,
then check `Verifying the connection`_.

**Claude Desktop** — edit ``claude_desktop_config.json`` (macOS
``~/Library/Application Support/Claude/``, Windows ``%APPDATA%\Claude\``, Linux
``~/.config/Claude/``); merge into ``mcpServers``:

.. code-block:: json

    {
      "mcpServers": {
        "exosim": {
          "command": "exosim-mcp",
          "args": ["--work-dir", "/path/to/exosim-runs"]
        }
      }
    }

Fully quit and reopen the app; the tools appear under the tools icon.

**Claude Code** — from any directory (runs are written to ``./mcp-runs`` there,
unless that directory is an ExoSim checkout, which is then served and gets
``mcp-runs`` inside it; with an editable install (``pip install -e``) runs go to
``<checkout>/mcp-runs`` wherever the client starts the server); the ``claude mcp add`` syntax was checked against
``claude mcp add --help``:

.. code-block:: bash

    claude mcp add exosim -- exosim-mcp

Add ``--scope user`` to make it available in every project, or commit a project
``.mcp.json`` carrying the same ``mcpServers`` block. ``claude mcp list`` shows
the connection state.

**VS Code** (GitHub Copilot, agent mode) — a workspace ``.vscode/mcp.json``, or
*MCP: Open User Configuration* for every workspace. The key is ``servers`` and
each entry names its transport:

.. code-block:: json

    {
      "servers": {
        "exosim": {
          "type": "stdio",
          "command": "exosim-mcp",
          "args": ["--work-dir", "${workspaceFolder}/mcp-runs"]
        }
      }
    }

Open the Chat view, switch to *Agent* mode, and tick ``exosim`` in the tools
picker.

**Cursor** — ``~/.cursor/mcp.json`` (global) or ``.cursor/mcp.json`` (project);
key ``mcpServers``, same shape as Claude Desktop. Enable it under
*Settings → MCP*.

**opencode** — a project ``opencode.json`` or
``~/.config/opencode/opencode.json``; the key is ``mcp`` and the command is a
list:

.. code-block:: json

    {
      "mcp": {
        "exosim": {
          "type": "local",
          "command": ["exosim-mcp"],
          "enabled": true
        }
      }
    }

**Cline** — the extension's *MCP Servers* panel writes
``cline_mcp_settings.json``; key ``mcpServers``, same shape as Claude Desktop.

**Continue** — ``~/.continue/config.yaml``:

.. code-block:: yaml

    mcpServers:
      - name: exosim
        command: exosim-mcp

**Other clients** — Windsurf uses ``~/.codeium/windsurf/mcp_config.json``
(``mcpServers``); Zed uses ``context_servers`` in its ``settings.json``. Any
client that speaks MCP over stdio can launch ``exosim-mcp``.

Verifying the connection
------------------------

Ask the assistant to *call the exosim* ``capabilities()`` *tool*. A healthy
connection returns the ExoSim and ``mcp`` SDK versions, ``execution_enabled``,
the retrieval backend and the list of pipelines. If the client never sees the
tool, run the server by hand with the inspector (see `Debugging`_).

Troubleshooting
---------------

The client lists ``exosim`` but no tools
    The ``mcp`` extra is missing or the package fails to import. In the same
    environment run the check from the `Quickstart`_ (``python -c "from
    exosim.mcp.server import build_server; build_server()"``); ``exosim-mcp
    --help`` is not a check, because it works without the SDK. Fix the error it
    prints; it names the installed ``mcp`` version and the original import error.

``run_pipeline`` is absent
    The server was started with ``--no-exec`` (or ``EXOSIM_MCP_NO_EXEC``). The
    five execution tools (``run_pipeline``, ``get_job_status``, ``cancel_job``,
    ``list_jobs`` and ``plot_output``) are then not registered, and
    ``capabilities()`` lists them under ``tools.disabled``.

A tool call fails with ``Error executing tool X: … is outside the directories this MCP server may read``
    The path is outside the sandbox (a ``PathNotAllowedError`` on the server).
    Tool errors reach the client as ``Error executing tool X: <message>``, where
    the message says what went wrong. Expected failures (``OSError``,
    ``ValueError`` including ``PathNotAllowedError``, ``ImportError``,
    ``KeyError``, ``LookupError``, and the missing-content and missing-embeddings
    errors) carry their own text; an unexpected internal error stays a generic
    ``Error executing tool X`` and its traceback is on the server's stderr. For
    a path problem, add the directory to ``EXOSIM_MCP_ALLOW_DIRS`` (see
    `Safety model`_).

The knowledge base is empty
    ``capabilities()["sandbox"]["content"]["warnings"]`` says so, and so does
    stderr at startup. Reinstall ``exosim``, or point ``--repo-root`` at a
    checkout.

Nothing happens after moving the checkout
    A client config that pins ``command`` or ``--repo-root`` still has the old
    path. Update it and reload.

Transports
----------

``exosim-mcp`` speaks three transports:

``--transport stdio`` (default)
    The server runs as a child process of one client over stdin/stdout. This is
    what a ``"command": "exosim-mcp"`` entry uses. Nothing is on the network.

``--transport sse`` / ``--transport http`` (with ``--host`` / ``--port``)
    The server listens on a socket (default ``127.0.0.1:8000``) so several — or
    remote — clients share one instance:

    .. code-block:: bash

        exosim-mcp --transport http --host 0.0.0.0 --port 8000 \
                   --no-exec

    Clients then point at the URL instead of a command. The HTTP endpoint is
    ``/mcp`` (the SSE endpoint is ``/sse``):

    .. code-block:: json

        {"mcpServers": {"exosim": {"url": "http://exosim-host:8000/mcp"}}}

    The HTTP and SSE transports have **no authentication**. Do not bind to
    ``0.0.0.0`` on a network you do not trust, and put an authenticating proxy in
    front of any shared instance. A shared instance almost always wants
    ``--no-exec`` (see `Safety model`_). When bound to a loopback address the
    SDK checks the ``Host`` and ``Origin`` headers (HTTP 421 / 403) against
    DNS rebinding; that protection is not enabled for other bind addresses.

Debugging
---------

Try the tools by hand with the official inspector:

.. code-block:: bash

    npx @modelcontextprotocol/inspector exosim-mcp --no-exec

Drop ``--no-exec`` to inspect the execution tools too. In the MCP server,
ExoSim's console logs go to stderr, never stdout, which carries the protocol.

SDK version
-----------

``exosim[mcp]`` supports the ``mcp`` Python SDK 2.x only (``mcp>=2,<3``; ``uv.lock``
pins 2.1.1). With the 1.x SDK installed, ``exosim-mcp`` stops with an
error asking for ``mcp>=2``: upgrade with ``pip install -U "exosim[mcp]"``.
``capabilities()`` reports the SDK version in use. ``setuptools`` is a runtime
dependency of ExoSim (``batman`` needs ``distutils`` on Python 3.12).


The tools
=========

Errors and refusals
-------------------

Two kinds of "no" reach the client, and they look different on purpose:

**A tool error** (the call fails, with ``isError`` set and the message as text)
    The *request itself* is not acceptable: a path outside the sandbox, a file
    that does not exist, an invalid argument (an unknown ``scale``, a
    non-positive ``max_runtime_seconds``), an output that exists without
    ``overwrite=True``, an unreadable table. Retrying the same call cannot
    succeed; the client has to change the arguments.

**A structured payload** (the call succeeds and the result says why not)
    The request was well formed and the answer is a *state* or the *outcome* of
    an operation: ``validate_config`` returns ``valid: false`` with the loader's
    ``error``; ``run_pipeline`` returns ``status: "refused"`` (too many jobs, or
    the output belongs to a running job) or the job's ``failed`` / ``timeout``
    status with a ``log_tail``; ``get_job_status`` and ``cancel_job`` return
    ``status: "unknown"`` for an unknown id; ``describe_pipeline`` returns an
    ``error`` for an unknown pipeline name; ``plot_output`` returns ``error``
    for a missing file or an unplottable kind. Retrying later, or after a job
    ends, can succeed.

``plot_output`` is the one exception: a missing file is a payload there, not a
tool error.

Information and grounding
-------------------------

``capabilities()``
    What this server can do right now: the ExoSim and MCP SDK versions, the
    sandbox configuration, whether pipeline execution is enabled, the retrieval
    backend, the available pipelines, which tools are live vs disabled, the
    content root (``sandbox.content``) and any ``discovery_errors`` met while
    importing the recipes. Call it first so the model does not attempt a
    disabled operation.

``search_knowledge_base(query, max_results=5, max_chars=0)``
    Ranked passages over a curated set of ExoSim files, each with its file path
    and line range (see `How grounding works`_). Returns
    ``status="insufficient_evidence"`` and no passages when nothing matches — the
    signal to *not* answer from general knowledge.

    ``max_chars`` caps the total passage text for small-context clients; a
    passage cut short is flagged ``truncated`` and keeps its path so the full
    text can be fetched from the ``exosim://kb/<path>`` resource.

``knowledge_base_manifest()``
    The approved source patterns, the matched files, the retrieval backend and
    the minimum score used to decide "insufficient evidence".

``exosim_project_context()``
    The curated agent-guidance files (``AGENTS.md``, ``docs/ai-agents/*``).

``customisation_guide(topic)``
    Grounded guidance on custom tasks, custom pipelines and ``RunConfig``.

Configuration
-------------

``validate_config(configuration)``
    Loads an XML / YAML configuration with the real ExoSim loader and reports
    ``valid``, the parsed ``top_level_sections``, the ``included_files`` and a
    short ``summary``; a failure comes back as ``valid: false`` with the loader's
    message in ``error``. Loading does not execute the Python tasks a
    configuration names (checked with a ``*_task`` entry pointing at a ``.py``
    file under ``--no-exec``). A
    configuration's own ``<ConfigPath>`` is honoured (the radiometric example
    sets ``<ConfigPath>..``); the same holds for ``config_requirements``.

``compare_configs(config1, config2)``
    Differences, missing sections and identical parts between two configs.

``config_requirements(pipeline, configuration=None)``
    The configuration sections a pipeline expects, taken from its bundled
    reference example — each with the keys it sets and a pointer to the doc page
    that explains it. Pass a ``configuration`` to also get the ``present`` /
    ``missing`` / ``extra`` sections for that file — turning "invalid" into
    "here is what to add".

``draft_config(pipeline, overrides=None, overwrite=False)``
    Produce a working configuration: it inlines the reference example's
    ``<config>`` includes into one self-contained file, applies
    ``{"section.key": "value"}`` overrides, writes it to
    ``drafts/<pipeline>_draft.xml`` under the working directory and validates it. Anything an override cannot find is reported with the
    keys that *are* available there. An existing draft is not replaced unless
    ``overwrite=True`` (the refusal says ``Set overwrite=True``). See
    :doc:`common_edits` for worked examples.

    Overrides are checked. A value ending in ``.py`` (as in a ``*_task`` or
    ``task_model`` entry) points a task at Python code that ExoSim will execute:
    it is refused under ``--no-exec``, and otherwise the file must still lie in a
    readable root. A plain class name such as ``LoadOpticalElement`` is looked up
    among ExoSim's own tasks and is accepted.
    Path-like keys (``ConfigPath``, ``datafile``, ``path``, ``filename``,
    ``file_name``, ``dp_map``, ``limb_darkening_coefficients``, ``url``, and keys
    ending in ``_path``, ``_file``, ``_filepath``, ``_files``, ``_filename`` or
    ``_dir``) must resolve inside the sandbox; a ``url`` override with a remote
    scheme (``http://``, ``https://``) is refused, and only ``file://`` is
    accepted.

Pipelines
---------

``list_pipelines()``
    The supported pipelines and their equivalent ``exosim`` CLI commands, plus
    ``discovery_errors`` (import failures met while discovering the recipes;
    empty when healthy). Four are exposed: ``focalplane``, ``subexposures``,
    ``ndrs`` and ``radiometric``. The ``observation`` recipe is deliberately not
    exposed: it writes fixed files into the configuration's directory and has no
    working-directory parameter.

``describe_pipeline(pipeline)``
    The recipe parameters for one pipeline — name, type, required, default,
    description (from the recipe docstring) and which ``run_pipeline`` argument
    fills each (``mcp_argument``). Call this before assembling a ``run_pipeline``
    request. An unknown name returns ``{"error": "Unsupported pipeline: …"}``; if
    the recipe cannot be imported it returns a structured error with
    ``missing_module`` and a ``hint``.

``list_examples()``
    The bundled ``examples/*.xml`` with a description, their ``<config>``
    includes, whether each is a pipeline entry point, and which pipelines it
    suits (``suits_pipelines``) — so the model starts from a real config.

``run_pipeline(pipeline, configuration, output, …)``
    Runs ``focalplane`` / ``radiometric`` / ``subexposures`` / ``ndrs`` in a
    supervised background process (see `Running simulations`_). Returns the
    final status for a quick run, a ``job_id`` to poll, or ``status="refused"``
    when too many runs are active. A missing ``input_file``, an existing
    ``output`` without ``overwrite=True`` and a non-positive
    ``max_runtime_seconds`` are rejected immediately (as tool errors). Not
    registered under ``--no-exec``.

``get_job_status(job_id)`` / ``cancel_job(job_id)`` / ``list_jobs()``
    Poll, stop or list background runs. Non-success statuses carry the log tail
    (``log_tail``, the last 4000 characters) and a ``reason``. An unknown
    ``job_id`` returns ``{"status": "unknown", "error": "Unknown job id: …"}``
    rather than a tool error.

Outputs and assistance
----------------------

``inspect_output(path, max_items=80)``
    Structural summary of an HDF5 / ECSV / CSV output — groups, datasets, shapes
    and attributes, at most ``max_items`` HDF5 nodes. Recognised nodes also carry
    a one-line ``meaning``, the producing ``stage`` and a doc pointer (``see``).
    A relative ``path`` is looked up in the working directory first, then the
    content root; the same holds for ``explain_output``, ``plot_output``,
    ``validate_config``, ``compare_configs`` and ``run_pipeline``'s
    ``configuration`` and ``input_file``. A table that cannot be parsed (a
    malformed ECSV) is an error ``cannot read <path>: <reason>``.

``explain_output(path, max_items=200)``
    The structure *plus* the science: every recognised group / dataset annotated
    with what the numbers mean, which pipeline stage produced them and the doc
    page that explains them, from a built-in map. Unrecognised nodes are listed
    under ``unexplained``, and the producing pipeline is guessed from the layout.

``plot_output(path, kind=None, time_step=0, scale="linear")``
    Render figures from an output with the same plotters as ``exosim-plot`` and
    write the PNGs under the working directory, for a vision-capable client to
    look at. The kind (``focalplane`` / ``radiometric`` / ``subexposures`` /
    ``ndrs``) is detected from the file when omitted. ``scale`` is ``linear`` or
    ``dB`` (case-insensitive, focal plane only); anything else is an error. The
    figures go to ``<work-dir>/plots/<name>-<hash>/``, where the hash of the
    resolved path keeps two outputs with the same file name apart, and
    ``figures`` lists only the files written by *this* call, also when it ends
    in an ``error``. The plotters share matplotlib's global state, so concurrent
    calls are rendered one at a time. An execution tool — not registered under
    ``--no-exec``.

``diagnose_error(error_message, context="", traceback_text="")``
    Extracts the exception type and failing frame from a traceback or log tail,
    classifies the problem and links related docs. ``context`` is free text
    (which pipeline, which configuration) echoed back with the diagnosis.

``suggest_next_steps(context)``
    Proposes the next pipeline stage for a workflow state (e.g.
    ``"focalplane_complete"``). The image chain is ``focalplane`` →
    ``subexposures -i focal.h5`` → ``ndrs -i subexp.h5``; ``radiometric`` is an
    independent branch that takes no input file and feeds no other pipeline.


Resources and prompts
=====================

Besides tools, the server exposes:

**Resources** — the knowledge-base files, browsable by URI. ``exosim://kb``
lists them; ``exosim://kb/<repo-relative-path>`` (a ``{+path}`` template, so
nested paths such as ``exosim://kb/docs/source/user/mcp.rst`` work) returns one
file's text (only files in the manifest, capped at 200 kB). A path that is not in
the manifest is refused, but the client sees only a generic ``Error creating
resource from template`` (the explanation is in the server's stderr); use the
paths ``exosim://kb`` lists. Use these to pull a whole page rather than only the
passages a search returned.

**Prompts** — ready-made workflows a client can start from:

``new_simulation``
    Go from nothing to a running simulation: requirements, draft, run, explain.
``set_up_channel``
    Configure a payload channel parameter by parameter, then validate.
``diagnose_failed_run``
    Work through a failed pipeline run from its error text.
``explain_output``
    Inspect an output file and explain each group from the docs.

Every prompt tells the model to ground its claims in the tools.


How grounding works
===================

``search_knowledge_base`` runs a dependency-light **BM25** ranker over the files
listed by ``knowledge_base_manifest``:

- **Structure-aware chunking**: reStructuredText is split by section, Python by
  top-level ``def`` / ``class``, Markdown by heading, other files by a fixed
  line window. Each hit reports its file and line range for fact-checking.
- **Identifier-aware tokenisation**: ``LoadOptions`` also matches ``load`` and
  ``options``; ``wl_grid`` matches ``wl`` and ``grid``.
- A passage is returned only if it contains a meaningful fraction of the
  distinct query terms (``minimum_retrieval_score``, default 0.15); otherwise
  the tool returns ``insufficient_evidence`` and an empty result, and the
  client should not answer from general knowledge.
- The index is cached and rebuilt automatically when a source file changes.

To add a source class, extend ``DEFAULT_SOURCE_PATTERNS`` in
``src/exosim/mcp/retrieval.py``.

Hybrid (semantic) retrieval
---------------------------

BM25 misses paraphrase — "shot noise" will not find a page that only says
"photon noise". Install the extra and switch backend:

.. code-block:: bash

    pip install "exosim[mcp,mcp-embeddings]"
    export EXOSIM_MCP_RETRIEVAL=hybrid

The extra does not include ``mcp``, so install both:
``pip install "exosim[mcp,mcp-embeddings]"`` (it pulls in PyTorch through
``sentence-transformers``). A passage then also qualifies on sentence-embedding
similarity (``all-MiniLM-L6-v2``, downloaded on first use, cosine at least
0.35); a passage's score is the higher of its normalised BM25 score and its
cosine similarity. The lexical backend stays the default and the fallback if the extra
is missing; ``knowledge_base_manifest`` reports which backend is live. A passage
below both the lexical gate and the cosine floor is still dropped, so
``insufficient_evidence`` behaves the same.

Retrieval eval suite
--------------------

``tests/unit/mcp/retrieval_cases.py`` holds ground-truth (question, expected
source) pairs; ``nox -s retrieval-eval`` runs each one through the real ranker
and checks aggregate recall@k (at least 80 %). It is the guardrail that a doc rewrite or a
change to the ranker does not quietly make a topic unfindable. When you add or
move a documentation page, add or update the matching case.


.. _mcp-safety-model:

Safety model
============

An MCP client is an autonomous program driven by a language model. Treat it as
an untrusted user of the machine the server runs on.

**With execution enabled (the default) the server runs code on the host with
your privileges.** ``run_pipeline`` and ``plot_output`` run ExoSim recipes, and
an ExoSim configuration can reference Python tasks (``*_task`` / ``task_model``
entries pointing at a ``.py`` file), ``<config>`` includes and data files. The
sandbox confines the **paths the MCP tools accept**; it does not confine what a
configuration's includes, data files or custom tasks do once a run starts. A
client that can write a configuration and run it can therefore execute arbitrary
code. Use ``--no-exec`` whenever that is not acceptable.

Confinement applies to the **arguments of the tools** (a path passed to
``validate_config``, ``run_pipeline``, ``draft_config`` overrides and so on). The
files that an already-readable configuration refers to (``<config>`` includes,
``datafile`` entries, task modules) are loaded as ExoSim loads them, with no
further check by the server.

**The content root** (``--repo-root``; see `Where the content comes from`_)
    The ExoSim checkout or the packaged content. Tools may *read* configuration
    files, examples and source here; no tool writes here.

**The working directory** (``--work-dir``, default ``./mcp-runs``)
    ``<repo-root>/mcp-runs`` for a checkout (git-ignored), or ``./mcp-runs`` in the
    server's current directory when the content is packaged. Every file the
    tools write (outputs, logs, drafts, plots) goes here, and results are read
    back from it. All writes go to the working directory (plus any
    ``EXOSIM_MCP_ALLOW_DIRS`` entry).

Relative paths in read tools are tried against the working directory first (so
a draft or an output of an earlier run is found), then the content root; this
holds for ``validate_config``, ``compare_configs``, ``inspect_output``,
``explain_output``, ``plot_output`` and ``run_pipeline``'s ``configuration`` and
``input_file``. Any path outside the roots, such as an absolute path elsewhere,
a ``~`` expansion, a symbolic link that leaves the roots or a ``../`` that
escapes them, is rejected with a ``PathNotAllowedError``
before any file is touched. Add directories with ``EXOSIM_MCP_ALLOW_DIRS``
(``os.pathsep``-separated, read and write) if your configurations live elsewhere.
In the packaged layout the current directory is not readable (only the packaged
content, the working directory ``./mcp-runs`` and ``EXOSIM_MCP_ALLOW_DIRS`` are),
so list it there if your configurations live in it.

``--no-exec`` (``EXOSIM_MCP_NO_EXEC``) removes the execution tools
(``run_pipeline``, ``get_job_status``, ``cancel_job``, ``list_jobs`` and
``plot_output``) and makes ``draft_config`` refuse overrides that point a task at
a Python file. The server then only answers questions, drafts and validates
configurations, and inspects outputs.

**Paths in results are absolute.** Tools show the server's real directory
layout (``output``, ``log_file``, ``draft``, error messages that list the
allowed roots); there is no abbreviated display form, and log tails are returned
as written. Do not treat a result as free of local paths when you share it.

The HTTP and SSE transports have no authentication (see `Transports`_).

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Environment variable
     - Flag
     - Meaning
   * - ``EXOSIM_MCP_REPO_ROOT``
     - ``--repo-root``
     - Override of the content root (checkout or packaged; read-only).
   * - ``EXOSIM_MCP_WORK_DIR``
     - ``--work-dir``
     - Writable directory for outputs and logs (default ``./mcp-runs``; if the
       default cannot be created, ``~/.exosim/mcp-runs`` with a warning; an
       explicit one that cannot be used stops the server with an error).
   * - ``EXOSIM_MCP_ALLOW_DIRS``
     - —
     - Extra allowed read/write roots, ``os.pathsep``-separated (``:`` on
       POSIX).
   * - ``EXOSIM_MCP_NO_EXEC``
     - ``--no-exec``
     - ``1``, ``true``, ``yes`` or ``on`` disables the five execution tools.
   * - ``EXOSIM_MCP_MAX_THREADS``
     - —
     - Hard cap on the worker threads a run may request (unset: no cap).
   * - ``EXOSIM_MCP_MAX_RUNTIME``
     - —
     - Hard wall-clock limit per run, seconds (default 3600). Zero, a negative
       number or anything that is not a number is invalid: the default is used
       and a warning is printed (it never means "unlimited").
   * - ``EXOSIM_MCP_MAX_JOBS``
     - —
     - Concurrent runs allowed (default 2; invalid or 0 gives the default).
   * - ``EXOSIM_MCP_EXEC_BACKEND``
     - —
     - ``process`` (default) or ``inline`` (runs synchronously in the server
       process; debugging only; anything else gives ``process``).
   * - ``EXOSIM_MCP_RETRIEVAL``
     - —
     - ``bm25`` (default) or ``hybrid`` (needs the ``mcp-embeddings`` extra;
       without it ``capabilities()`` still reports ``bm25``).

The content root and working directory are **server configuration** — set once at
startup, never exposed as tool arguments — so a client cannot widen its sandbox.


.. _mcp-running-simulations:

Running simulations
===================

A full ExoSim run can take minutes to hours, so ``run_pipeline`` never blocks
the server: each run happens in a **separate, supervised process**.

#. ``run_pipeline`` validates the request, starts a worker, and blocks for up to
   ``wait_seconds`` (default 30, at most 50: larger values are clamped, so a
   call stays below typical client tool timeouts; poll longer runs with
   ``get_job_status``).
#. If the run finishes in that window it returns the final status:

   .. code-block:: text

       {"job_id": "1a2b3c4d5e6f", "pipeline": "focalplane",
        "status": "succeeded", "output": "/…/mcp-runs/focal.h5",
        "log_file": "/…/mcp-runs/logs/1a2b3c4d5e6f.log",
        "submitted_at": …, "started_at": …, "finished_at": …,
        "runtime_seconds": 11.19, "configuration": "/…/examples/main_example.xml",
        "input": null}

#. Otherwise it returns a handle to poll:

   .. code-block:: text

       {"job_id": "1a2b3c4d5e6f", "pipeline": "focalplane", "status": "running",
        "output": "…", "log_file": "…", "finished_at": null, "runtime_seconds": 30.0,
        …}

#. The client polls ``get_job_status(job_id)`` until the status is
   ``succeeded``, ``failed``, ``timeout`` or ``cancelled``. On any non-success
   outcome the response carries ``log_tail`` and a ``reason``:

   .. list-table::
      :header-rows: 1
      :widths: 16 84

      * - Status
        - ``reason``
      * - ``succeeded``
        - none
      * - ``failed``
        - ``exit code N`` (the traceback is in ``log_tail``)
      * - ``timeout``
        - ``exceeded max_runtime and was terminated by the watchdog``
      * - ``cancelled``
        - ``cancelled by request`` (``cancel_job``) or ``server shut down``

   ``cancel_job`` stops a run; ``list_jobs`` lists every run since startup,
   oldest first.

Details:

- outputs go under ``--work-dir`` (a not yet existing subdirectory of it, such
  as ``newsub/deeper/x.h5``, is created); logs to ``<work-dir>/logs/<job_id>.log``.
  The log holds everything the run printed, including the ``RUN CONFIGURATION``
  banner and the output of any subprocess, without ANSI colour codes. Runs do not
  write ``__pycache__`` files into the content root;
- ``n_threads`` is clamped to ``EXOSIM_MCP_MAX_THREADS``; a run's ``RunConfig``
  changes stay in its own process and never leak into the server or other runs.
  Each run starts from the configuration file's directory, so relative
  ``<config>`` includes resolve;
- a run is hard-killed after ``max_runtime_seconds`` (per call) or
  ``EXOSIM_MCP_MAX_RUNTIME`` and reported as ``timeout``; a watchdog enforces
  this without anyone polling. ``max_runtime_seconds`` must be positive and can
  only shorten the server-side ``EXOSIM_MCP_MAX_RUNTIME`` limit, never lift it;
- at most ``EXOSIM_MCP_MAX_JOBS`` runs (default 2) are active. A further
  ``run_pipeline`` returns ``{"status": "refused", "error": "Too many
  concurrent jobs: …", "limit": 2, "running_jobs": […], "pipeline": …}`` instead
  of starting; cancel one or wait;
- a second ``run_pipeline`` on the ``output`` of a run that is still active is
  refused whatever ``overwrite`` says: ``{"status": "refused", "error":
  "Output … is already being written by running job …", "conflicting_job": …}``;
- ``run_pipeline`` rejects a missing ``input_file``, an existing ``output``
  (unless ``overwrite=True``) and an unsupported pipeline before starting
  anything; the downstream pipelines ``subexposures`` and ``ndrs`` need
  ``input_file``;
- running jobs are stopped when the server goes away: on SIGTERM the server
  terminates its children and exits; on stdin EOF (the client closes the
  connection) it exits and its children are terminated; on SIGINT it terminates
  the children but the MCP SDK's stdio loop then stays up until stdin is
  closed, so prefer SIGTERM; if the server is killed with SIGKILL the children
  notice the missing parent and exit on their own within a few seconds;
- jobs live in memory only and do not survive a restart;
- ``EXOSIM_MCP_EXEC_BACKEND=inline`` runs synchronously in the server process
  (debugging only).


Known limits
============

- **Search is lexical.** The default retrieval matches words, so a question
  that shares ordinary words with the documentation can still return
  ``status="ok"`` with passages that do not answer it (``"Ariel AIRS detector
  read noise value"`` returns the read-noise pages, which hold no instrument
  value). The assistant has to read the passages and say so;
  ``insufficient_evidence`` appears only when almost no word of the question is
  in the knowledge base (an unrelated trivia question does). This guide is part
  of the knowledge base too, so any question that repeats words printed here,
  including the queries quoted in the examples, ranks this page first.
- **Semantic search needs both extras.** ``mcp-embeddings`` does not include
  ``mcp``: install ``pip install "exosim[mcp,mcp-embeddings]"``. It pulls in
  PyTorch, and the model is downloaded on first use.
- **Stopping the stdio server.** SIGTERM and a closed stdin end the server and
  its job processes. After SIGINT the server terminates its job processes but
  keeps running until stdin closes; send SIGTERM (or close the client).
- **Resource URIs.** An ``exosim://kb/`` path that is not in the manifest is
  refused with a generic ``Error creating resource from template`` message; the
  explanation is only on the server's stderr.
- **No authentication on HTTP and SSE.** Both bind to ``127.0.0.1`` by default;
  see `Transports`_. Host and ``Origin`` checks apply only to a loopback bind.
- **Execution is arbitrary code execution.** With execution enabled, a
  configuration can name Python tasks and the sandbox does not confine what a
  run does (see `Safety model`_). ``--no-exec`` is the only switch.
- **Absolute paths** appear in results (see `Safety model`_).
- **Jobs live in memory.** A restart forgets them, and the log files stay in
  ``<work-dir>/logs``.
- **Windows** has not been tested. The job supervision uses POSIX signals and
  the ``spawn`` start method; use ``--no-exec`` there.
- **Configuration loading is serialised.** ``validate_config``,
  ``compare_configs`` and ``draft_config`` load a configuration in the server
  process, where the loader records files in the process-wide ``RunConfig``;
  they hold a lock for it, so concurrent calls queue. Runs are separate
  processes and do not touch the server's ``RunConfig``.
- **Not verified.** The guide has not been checked on Windows, with a client
  other than the stdio test client of the MCP SDK (the per-client recipes follow
  each client's documented format but were not run, apart from the
  ``claude mcp add`` syntax), the ``npx @modelcontextprotocol/inspector``
  command of `Debugging`_, or the ``mcp-embeddings`` extra in the test
  environment.


Example conversations
=====================

Three exchanges with a client configured as in `Quickstart`_. Tool calls the
assistant makes are shown as ``→ tool(args)`` with the result under them. The
keys and values are those the tools returned in a real run of the bundled
example (ExoSim 2.3 from a checkout, one thread), with ``…`` where a long part
is left out; times and job ids change from run to run.

Grounding an answer
-------------------

*You* — How is the radiometric model switched on, and what do I run?

*Assistant* —

.. code-block:: text

    → search_knowledge_base("how is the radiometric model switched on")
      {"status": "ok", "grounded": true, …}
      docs/source/user/mcp.rst:<lines>   (this page's own example)
      docs/source/user/radiometric_model/index.rst:4-14
      docs/source/user/radiometric_model/operating_modes.rst:5-11
      src/exosim/recipes/radiometric_model.py:401-520

The radiometric model is a separate pipeline
(:doc:`radiometric_model/index`, :doc:`radiometric_model/operating_modes`). Run
it with ``exosim radiometric -c <config> -o <output>``, or point ``run_pipeline``
at the ``radiometric`` pipeline; the bundled entry point is
``examples/radiometric/main_radiometric.xml``.

*You* — (A trivia question with nothing to do with ExoSim.)

*Assistant* —

.. code-block:: text

    → search_knowledge_base("<the trivia question>")
      status: insufficient_evidence

The knowledge base has no evidence for that, so I will not answer it from
general knowledge. (A question that does share words with the documentation,
such as one about the read noise of a named instrument, would return ``ok``;
read the passages before relying on them. See `Known limits`_.)

Running a pipeline and explaining the output
--------------------------------------------

*You* — Run the focal plane for the bundled example and tell me what came out.

*Assistant* —

.. code-block:: text

    → list_examples()
      examples/main_example.xml  is_entrypoint: true  suits_pipelines: [focalplane, subexposures, ndrs]
    → describe_pipeline("focalplane")
      required_inputs: [configuration, output]
    → run_pipeline("focalplane", "examples/main_example.xml", "focal.h5")
      {"job_id": "d7d2ef12b4c7", "pipeline": "focalplane", "status": "succeeded",
       "output": "…/mcp-runs/focal.h5", "log_file": "…/mcp-runs/logs/d7d2ef12b4c7.log",
       "runtime_seconds": 11.19, …}
    → explain_output("focal.h5")
      {"format": "hdf5", "pipeline": "focalplane", "explained_count": …, …}

The run finished inside the default 30-second wait, so ``run_pipeline`` returned
the final status itself; a longer run returns ``"status": "running"`` and the
client polls ``get_job_status``. ``focal.h5`` is in the working directory, with
one group per channel. For the photometer:

- ``channels/Photometer/focal_plane`` — the target-star focal plane, a
  (time, y, x) cube of counts/s per pixel
  (:doc:`focal_plane/resulting_focal_plane`);
- ``channels/Photometer/frg_focal_plane`` — the foreground (zodi, instrument
  emission, sky) focal plane, kept separate (:doc:`focal_plane/foregrounds`);
- ``channels/Photometer/responsivity`` — detector responsivity vs wavelength
  (:doc:`focal_plane/channel`).

``explain_output`` also lists the nodes it does not recognise under
``unexplained`` (17 for this file, among them ``sky/foregrounds/zodiacal``).

A request the sandbox refuses
-----------------------------

*You* — Validate ``/etc/passwd``, then draft a focal-plane configuration whose
jitter task is ``evil.py`` (server started with ``--no-exec``).

*Assistant* —

.. code-block:: text

    → validate_config("/etc/passwd")
      Error executing tool validate_config: /etc/passwd is outside the
      directories this MCP server may read (<content root>, <work dir>)

    → draft_config("focalplane", {"jitter.jitter_task": "evil.py"})
      Error executing tool draft_config: Override of 'jitter_task' points a task
      at a Python file (evil.py); this is arbitrary code execution and is
      refused while execution is disabled (--no-exec).

Both refusals carry the reason in the message, so the assistant can explain what
to do instead (copy the file into an allowed directory, or use a built-in task
by its class name). On macOS the first message shows the resolved
``/private/etc/passwd``.

``examples/mcp_demo_conversation.py`` is a separate scripted walk-through of
the read-only tools; run it to see their exact inputs and outputs. Its last step
sends a nonsense question that still returns ``status="ok"`` (see `Known
limits`_).


Using the tools from Python
===========================

Every tool in ``exosim.mcp.tools`` is a plain function — useful for scripting, or
for testing the server without a client. They read their sandbox from the
environment (``EXOSIM_MCP_REPO_ROOT`` etc.); ``exosim.mcp.pipelines`` also
accepts ``repo_root`` / ``work_dir`` directly.

``examples/mcp_example.py`` is a runnable walk-through (the test suite executes
it, so it cannot drift from the API):

.. literalinclude:: ../../../examples/mcp_example.py
   :language: python
