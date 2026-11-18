"""Content root of the ExoSim MCP server: knowledge base, examples, reference configs.

The server answers from documentation, example configurations and a few source
files. In a source checkout these are simply the repository files; in a
``pip install exosim[mcp]`` they are shipped *inside the wheel* under
``exosim/mcp/_data`` (same layout as the repository), so an installed server is
never empty.

This module has two halves:

* **Build time** (:func:`collect_assets`, :data:`ASSET_RULES`): copies the repo
  files into a destination directory. ``setup.py`` calls it from ``build_py``,
  so the repository stays the single source of truth and nothing is committed
  twice. It uses the standard library only and is loaded by path from
  ``setup.py`` (``exosim`` is not importable at that point).
* **Run time** (:func:`resolve_content_root`, :func:`content_report`): picks the
  content root.

Precedence of the content root:

1. an explicit ``repo_root`` argument or ``EXOSIM_MCP_REPO_ROOT`` (``override``);
2. the current directory, if it is an ExoSim source checkout (``checkout``);
3. the checkout the ``exosim`` package was imported from, e.g. an editable
   install (``checkout``);
4. the packaged assets (``packaged``).

If none exists the current directory is used and ``source`` is ``none``.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ASSET_RULES",
    "ContentMissingError",
    "ContentRoot",
    "collect_assets",
    "content_report",
    "is_checkout",
    "packaged_root",
    "require_content_file",
    "resolve_content_root",
    "warn_if_content_missing",
]

ENV_REPO_ROOT = "EXOSIM_MCP_REPO_ROOT"
DATA_DIRNAME = "_data"

# Files shipped in the wheel, relative to the repository root. ``(glob, suffixes)``:
# an empty suffix tuple means "any suffix". Files above ``MAX_ASSET_BYTES`` and
# anything under a dot-directory are skipped, which keeps ``examples/customsed.csv``
# (21 MB) and notebooks out of the wheel.
ASSET_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AGENTS.md", ()),
    ("README.md", ()),
    ("CHANGELOG.rst", ()),
    ("docs/ai-agents/*.md", ()),
    ("docs/source/**/*.rst", ()),
    (
        "examples/**/*",
        (".xml", ".csv", ".ecsv", ".h5", ".py", ".yaml", ".yml", ".txt"),
    ),
    ("src/exosim/recipes/*.py", ()),
    ("src/exosim/tasks/load/*.py", ()),
    ("src/exosim/tasks/radiometric/**/*.py", ()),
)
# Internal process notes: not knowledge about *using* ExoSim.
EXCLUDED_ASSETS = frozenset(
    {
        "docs/ai-agents/mcp-hardening-roadmap.md",
        "docs/ai-agents/mcp-recipe-metadata.md",
        "docs/ai-agents/mcp-release-plan.md",
    }
)
MAX_ASSET_BYTES = 1_048_576


def _iter_assets(repo: Path):
    seen: set[Path] = set()
    for pattern, suffixes in ASSET_RULES:
        for path in sorted(repo.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            rel = path.relative_to(repo)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            if rel.as_posix() in EXCLUDED_ASSETS:
                continue
            if suffixes and path.suffix.lower() not in suffixes:
                continue
            if path.stat().st_size > MAX_ASSET_BYTES:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            seen.add(path)
            yield rel


def collect_assets(repo: str | Path, dest: str | Path) -> list[str]:
    """Copy the packaged-content files of a checkout into ``dest``.

    Parameters
    ----------
    repo : str or pathlib.Path
        Repository root (the single source of truth).
    dest : str or pathlib.Path
        Destination directory; recreated from scratch.

    Returns
    -------
    list of str
        Repository-relative paths copied.
    """
    repo, dest = Path(repo), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    copied: list[str] = []
    for rel in _iter_assets(repo):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, target)
        copied.append(rel.as_posix())
    return copied


# ---------------------------------------------------------------------------
# Run time
# ---------------------------------------------------------------------------
class ContentMissingError(RuntimeError):
    """The knowledge base / examples the tool needs are not available."""


@dataclass(frozen=True)
class ContentRoot:
    """Where the server reads its content from."""

    path: Path
    source: str  # "override" | "checkout" | "packaged" | "none"


def is_checkout(path: Path) -> bool:
    """True if ``path`` looks like an ExoSim source checkout."""
    return (
        (path / "pyproject.toml").is_file()
        and (path / "src" / "exosim").is_dir()
        and (path / "docs").is_dir()
    )


def packaged_root() -> Path | None:
    """Directory of the assets shipped in the wheel, or ``None`` if absent."""
    from importlib.resources import files

    try:
        candidate = Path(str(files("exosim.mcp").joinpath(DATA_DIRNAME)))
    except (ModuleNotFoundError, TypeError, ValueError):  # pragma: no cover
        return None
    return candidate if candidate.is_dir() else None


def _package_checkout() -> Path | None:
    # src/exosim/mcp/assets.py -> repository root is parents[3].
    candidate = Path(__file__).resolve().parents[3]
    return candidate if is_checkout(candidate) else None


def resolve_content_root(
    repo_root: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ContentRoot:
    """Resolve the content root following the documented precedence."""
    environ = os.environ if env is None else env
    explicit = repo_root or environ.get(ENV_REPO_ROOT)
    if explicit:
        return ContentRoot(Path(explicit).expanduser().resolve(), "override")
    here = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    if is_checkout(here):
        return ContentRoot(here, "checkout")
    checkout = _package_checkout()
    if checkout is not None:
        return ContentRoot(checkout, "checkout")
    packaged = packaged_root()
    if packaged is not None:
        return ContentRoot(packaged, "packaged")
    return ContentRoot(here, "none")


def content_report(root: Path, source: str) -> dict[str, object]:
    """Describe the content at ``root``: source, file counts, warnings."""
    from exosim.mcp.retrieval import _iter_source_files

    kb_files = sum(1 for _ in _iter_source_files(root))
    examples = sum(
        1
        for p in root.glob("examples/**/*.xml")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    warnings: list[str] = []
    hint = (
        "Reinstall exosim (the wheel ships the content), or point the server at "
        "an ExoSim checkout with --repo-root / EXOSIM_MCP_REPO_ROOT."
    )
    if kb_files == 0:
        warnings.append(
            f"Knowledge base is empty at {root}: search_knowledge_base cannot "
            f"answer. {hint}"
        )
    if examples == 0:
        warnings.append(
            f"No example configurations found at {root}: list_examples, "
            f"draft_config and config_requirements will not work. {hint}"
        )
    return {
        "source": source,
        "root": str(root),
        "knowledge_base_files": kb_files,
        "example_configs": examples,
        "warnings": warnings,
    }


_WARNED: set[str] = set()


def warn_if_content_missing(root: Path, source: str) -> None:
    """Log (once per root) a warning to stderr when the content is empty."""
    key = f"{source}:{root}"
    if key in _WARNED:
        return
    _WARNED.add(key)
    for message in content_report(root, source)["warnings"]:  # type: ignore[attr-defined]
        logging.getLogger("exosim.mcp").warning("%s", message)


def require_content_file(path: Path, what: str) -> Path:
    """Return ``path`` or raise an actionable :class:`ContentMissingError`."""
    if not path.exists():
        raise ContentMissingError(
            f"{what} not found at {path}. The ExoSim MCP content (examples and "
            "documentation) is missing: reinstall exosim, or run the server with "
            "--repo-root / EXOSIM_MCP_REPO_ROOT pointing at an ExoSim checkout."
        )
    return path
