"""
Package metadata for ExoSim 2.

This module centralizes all package metadata following modern Python practices.
All version and package information should be imported from here.
"""

from __future__ import annotations

import os.path
from datetime import date
from importlib.metadata import (
    PackageNotFoundError,
    version as metadata_version,
)
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # fallback for older Python
    except ImportError:
        tomllib = None  # No TOML support available


def _read_pyproject_toml() -> dict:
    """Read and parse pyproject.toml from project root."""
    if tomllib is None:
        raise ImportError("No TOML library available (tomllib or tomli required)")

    # Find project root (directory containing pyproject.toml)
    current_dir = Path(__file__).parent
    project_root = None

    # Walk up the directory tree to find pyproject.toml
    for parent in [current_dir, *current_dir.parents]:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.exists():
            project_root = parent
            break

    if project_root is None:
        raise FileNotFoundError("Could not locate pyproject.toml")

    pyproject_path = project_root / "pyproject.toml"

    try:
        with open(pyproject_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read pyproject.toml: {e}") from e


def _read_installed_version(package_name: str) -> str | None:
    """Read package version from installed distribution metadata."""
    try:
        return metadata_version(package_name)
    except PackageNotFoundError:
        return None


def _read_version_file() -> str | None:
    """Read the version from the setuptools-scm generated ``exosim._version``.

    The file is written at build/install time and is git-ignored, so it may be
    absent in a bare source checkout; in that case ``None`` is returned and the
    caller falls back to the installed distribution metadata.
    """
    try:
        from exosim._version import version as scm_version
    except Exception:
        return None
    return scm_version or None


def _resolve_version(project: dict, package_name: str) -> str:
    """
    Resolve package version from available metadata sources.

    Resolution order:

    1. an explicit ``version`` key in ``project`` (used by tests and static
       builds; ``pyproject.toml`` no longer carries one because the version is
       now derived from git by setuptools-scm);
    2. the setuptools-scm generated ``exosim._version`` module;
    3. the installed distribution metadata;
    4. ``"unknown"`` when nothing else is available.

    Parameters
    ----------
    project : dict
        Project metadata loaded from ``pyproject.toml``.
    package_name : str
        Distribution name used to query installed package metadata.

    Returns
    -------
    str
        The resolved version string, or ``"unknown"`` when unavailable.
    """
    return (
        project.get("version")
        or _read_version_file()
        or _read_installed_version(package_name)
        or "unknown"
    )


# Load project configuration from pyproject.toml
try:
    _pyproject = _read_pyproject_toml()
    _project = _pyproject.get("project", {})
except (FileNotFoundError, RuntimeError):
    # Fallback for cases where pyproject.toml is not accessible
    _project = {}

# Core package metadata - single source of truth from pyproject.toml
__pkg_name__ = _project.get("name", "exosim")
__version__ = _resolve_version(_project, __pkg_name__)
__title__ = "ExoSim 2"
__description__ = _project.get("description", "Exoplanet Observation Simulator")

# Extract URLs
_urls = _project.get("urls", {})
__url__ = _urls.get("Homepage", "https://github.com/arielmission-space/ExoSim2-public")

# Extract author information
_authors = _project.get("authors", [])
if _authors:
    __author__ = _authors[0].get("name", "L. V. Mugnai")
    __author_email__ = _authors[0].get("email", "mugnail@cardiff.ac.uk")
else:
    __author__ = "L. V. Mugnai"
    __author_email__ = "mugnail@cardiff.ac.uk"

# Extract license
_license_info = _project.get("license", {})
__license__ = _license_info.get("text", "BSD-3-Clause")
__copyright__ = f"2020-{date.today().year}, {__author__}"

# Scientific citation
__citation__ = """@article{Mugnai2025,
    author  = {Mugnai, Lorenzo V. and Bocchieri, Andrea and Pascale, Enzo and Lorenzani, Andrea and Papageorgiou, Andreas},
    title   = {ExoSim 2: the new exoplanet observation simulator applied to the Ariel space mission},
    journal = {Experimental Astronomy},
    year    = {2025},
    volume  = {59},
    number  = {1},
    pages   = {9},
    doi     = {10.1007/s10686-024-09976-2},
    url     = {https://doi.org/10.1007/s10686-024-09976-2},
    eprint        = {2501.12809},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.IM},
    ascl_id       = {ascl:2503.031},
}"""


def _get_git_info() -> tuple[str | None, str | None]:
    """
    Extract git branch and commit information.

    Returns:
        tuple: (branch_name, commit_hash) or (None, None) if not in git repo
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return None, None

    git_folder = os.path.join(base_dir, ".git")
    if not os.path.exists(git_folder):
        return None, None

    try:
        # Read HEAD to find current branch/ref
        with open(os.path.join(git_folder, "HEAD")) as fp:
            ref = fp.read().strip()

        if ref.startswith("ref: "):
            # We're on a branch
            ref_path = ref[5:]  # Remove "ref: " prefix
            branch_name = ref_path.split("/")[
                -1
            ]  # Extract branch name from refs/heads/branch_name

            # Read commit hash
            try:
                with open(os.path.join(git_folder, ref_path)) as fp:
                    commit_hash = fp.read().strip()
            except FileNotFoundError:
                # Might be a packed ref or detached HEAD
                commit_hash = None
        else:
            # Detached HEAD - ref is the commit hash directly
            branch_name = "HEAD"
            commit_hash = ref

        return branch_name, commit_hash

    except OSError:
        return None, None


# Git information (computed once on import)
__branch__, __commit__ = _get_git_info()

# Version information tuple for programmatic access
__version_info__ = (
    tuple(int(x) for x in __version__.split(".") if x.isdigit())
    if __version__ != "unknown"
    else (0, 0, 0)
)


def is_development_version() -> bool:
    """Check if this is a development version."""
    return (
        "dev" in __version__.lower()
        or "alpha" in __version__.lower()
        or "beta" in __version__.lower()
    )


def is_release_version() -> bool:
    """Check if this is a stable release version."""
    return not is_development_version()


# All metadata for easy access
__all__ = [
    "__author__",
    "__author_email__",
    "__branch__",
    "__citation__",
    "__commit__",
    "__copyright__",
    "__description__",
    "__license__",
    "__pkg_name__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "is_development_version",
    "is_release_version",
]
