.. _releasing:

=========================
Releasing ExoSim 2
=========================

This page documents how a new version of ExoSim 2 is produced, from a commit on
``develop`` to a package on PyPI and a mirrored release on the public
repository.

Overview
========

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Piece
     - Role
   * - ``develop``
     - The single active development branch. Every commit gets an automatic
       development version.
   * - ``main``
     - Holds the latest stable release only. Fast-forwarded by the release
       process; never committed to directly.
   * - ``setuptools-scm``
     - Derives the version from git tags. No version string is stored in the
       repository.
   * - ``changelog.d/``
     - Hand-written changelog fragments, merged into ``CHANGELOG.rst`` at
       release time by `scriv <https://scriv.readthedocs.io/>`__.
   * - ``nox -s release``
     - The one command that cuts a release: asks major/minor/patch, updates the
       changelog and metadata, pushes the ``vX.Y.Z`` tag.
   * - ``.github/workflows/release.yml``
     - Triggered by the tag. Builds, publishes to TestPyPI then PyPI, creates
       the GitHub Release.
   * - ``.github/workflows/sync-to-public.yml``
     - Mirrors ``main`` to ``arielmission-space/ExoSim2-public`` and replicates
       the GitHub Release there.

Versioning
==========

The version is computed by ``setuptools-scm`` from the git history:

* on a release tag ``vX.Y.Z``           → ``X.Y.Z``
* ``N`` commits after the last tag       → ``X.Y.(Z+1).devN``

Every commit on ``develop`` therefore has a unique, PEP 440-compliant
development version, with **no** bot commits and **no** ``-devN`` tags. The
value is written to ``src/exosim/_version.py`` at build/install time (this file
is git-ignored) and exposed as ``exosim.__version__``.

To see the version of a working copy::

    uv run python -c "import exosim; print(exosim.__version__)"

Writing changelog entries
=========================

Changelog entries are **not** tied to individual commits. When a unit of work is
complete (it may span several commits), add one fragment describing it:

.. code-block:: console

    nox -s changelog-fragment

This opens an editor with a template. Uncomment the relevant categories
(``Added``, ``Changed``, ``Deprecated``, ``Removed``, ``Fixed``, ``Security``)
and write the entry in Keep a Changelog style. The file is created under
``changelog.d/`` with a unique name, so parallel work never conflicts on the
changelog.

Fragments accumulate on ``develop`` until the next release, when
``nox -s release`` merges them into ``CHANGELOG.rst`` under the new version
heading and deletes them.

Cutting a release
=================

From a clean, up-to-date ``develop`` checkout:

.. code-block:: console

    nox -s release

The script (``scripts/release.py``) will:

#. verify you are on ``develop``, the tree is clean and in sync with ``origin``,
   and that there is at least one changelog fragment;
#. run the quality gate (``ruff`` + ``pytest``); skip it with ``nox -s release -- --skip-checks``;
#. ask whether the bump is ``major``, ``minor`` or ``patch`` (or pass it:
   ``nox -s release -- minor``) and compute the next ``X.Y.Z`` from the last
   ``vX.Y.Z`` tag;
#. ask for a short **release name** (e.g. "CLI fixes"), used as the changelog
   heading, the git tag subject and the GitHub Release title;
#. run ``scriv collect`` to fold the fragments into ``CHANGELOG.rst`` under
   ``[X.Y.Z] - <name>``, add the release link, and mirror the file into
   ``docs/source/``;
#. update ``CITATION.cff`` and ``codemeta.json``;
#. commit on ``develop`` as ``docs: release X.Y.Z``;
#. after a confirmation prompt, push ``develop``, fast-forward ``main`` and push
   the annotated tag ``vX.Y.Z``.

Pushing the tag triggers ``release.yml``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Job
     - Action
   * - ``build``
     - ``uv build`` + ``twine check``; asserts the built version equals the tag.
   * - ``testpypi``
     - Publishes to TestPyPI via OIDC trusted publishing.
   * - ``verify``
     - Installs ``exosim==X.Y.Z`` from TestPyPI in a clean environment and
       checks ``exosim.__version__``.
   * - ``pypi``
     - Publishes to PyPI via OIDC trusted publishing.
   * - ``github-release``
     - Creates the GitHub Release (title ``X.Y.Z — <name>``, notes from
       ``CHANGELOG.rst``, ``dist/*`` attached).
   * - ``mirror-release``
     - Waits for ``sync-to-public.yml`` to mirror the release commit, then
       clones the tag, release notes and wheels into
       ``arielmission-space/ExoSim2-public``.

``sync-to-public.yml`` mirrors ``main`` to the public repo on every push; the
release/tag clone is done by ``mirror-release`` above rather than by a
``release: published`` trigger (which never fires for a release created with
``GITHUB_TOKEN``).

One-time setup
==============

PyPI trusted publishing
-----------------------

Publishing uses `PyPI trusted publishing
<https://docs.pypi.org/trusted-publishers/>`__ (OpenID Connect): no API tokens
are stored anywhere. On **both** https://pypi.org and https://test.pypi.org, for
the ``exosim`` project, add a trusted publisher:

============================  ===================
Field                         Value
============================  ===================
Owner                         ``arielmission-space``
Repository                    ``ExoSim2.0``
Workflow name                 ``release.yml``
Environment                   ``pypi`` (on PyPI) / ``testpypi`` (on TestPyPI)
============================  ===================

GitHub environments
-------------------

In ``arielmission-space/ExoSim2.0`` → *Settings → Environments*, create
``pypi`` and ``testpypi``. Optionally add a required reviewer on ``pypi`` so a
human approves the final upload.

Branch protection
-----------------

The release script fast-forwards ``main`` with
``git push origin HEAD:main``. If ``main`` is branch-protected, either allow the
release author to bypass, or merge ``develop`` into ``main`` through the GitHub
UI/API and then run::

    git tag -a vX.Y.Z -m "Release X.Y.Z" && git push origin vX.Y.Z

The public mirror
=================

``arielmission-space/ExoSim2-public`` is a **one-way mirror** of ``main``. It has
no CI of its own. ``sync-to-public.yml`` uses ``rsync --delete`` and excludes the
private dev tooling (``.github/``, ``noxfile.py``, ``scripts/``,
``changelog.d/``, ``.pre-commit-config.yaml``) and local build/coverage
artifacts. ``src/``, ``tests/`` and ``docs/`` are mirrored.

Do not commit to the public repository directly; changes there are overwritten
on the next sync.

Troubleshooting
===============

``setuptools-scm`` reports ``0.0.0``
    The build happened without git history/tags (for example a shallow clone).
    ``fallback_version`` in ``pyproject.toml`` prevents a hard failure; on Read
    the Docs the ``post_checkout`` job fetches the tags.

``verify`` job fails to install from TestPyPI
    The TestPyPI index can lag a minute or two behind an upload; the job retries
    for ~5 minutes. If a dependency is missing from TestPyPI, the
    ``--extra-index-url https://pypi.org/simple/`` fallback covers it.

The release commit is on ``develop`` but the push was aborted
    Run ``git reset --hard origin/develop`` to discard it and start over.
