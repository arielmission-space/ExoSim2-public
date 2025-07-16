import argparse

import nox

# Default sessions to run if no session is specified
nox.options.sessions = ["lint", "test"]


def get_version_from_config() -> str:
    """Get current version from .bumpversion.cfg"""
    with open(".bumpversion.cfg") as f:
        for line in f:
            if "current_version" in line:
                return line.split("=")[1].strip()
    raise ValueError("Could not find current_version in .bumpversion.cfg")


@nox.session
def lint(session: nox.Session) -> None:
    """Run pre-commit hooks."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "-a")


@nox.session
def test(session: nox.Session) -> None:
    """Run tests with pytest."""
    session.install(".[dev]")
    session.run("pytest", "tests", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.install(".[dev]")
    session.run("sphinx-build", "-b", "html", "docs/source", "build")


@nox.session(name="docs-live", reuse_venv=True)
def docs_live(session: nox.Session) -> None:
    """Build documentation with live reload."""
    session.install(".[dev]")
    session.run(
        "sphinx-autobuild",
        "-b",
        "html",
        "docs/source",
        "build",
        *session.posargs,
    )


@nox.session
def release(session: nox.Session) -> None:
    """Create a new release.

    Usage:
        nox -s release -- VERSION "Release description"
    Example:
        nox -s release -- 2.0.2 "Add new features"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="New version number (e.g. 2.0.2)")
    parser.add_argument("description", help="Release description")
    args = parser.parse_args(session.posargs)

    # Confirm release
    current = get_version_from_config()
    if (
        input(
            f"Release {current} → {args.version} ({args.description})? [y/N] "
        ).lower()
        != "y"
    ):
        session.error("Release cancelled")

    # Run checks
    session.run("pre-commit", "run", "-a", external=True)
    session.run("pytest", "tests", external=True)

    # Update changelog
    update_changelog(args.version, args.description)
    session.run(
        "git",
        "add",
        "CHANGELOG.rst",
        "docs/source/CHANGELOG.rst",
        external=True,
    )
    session.run(
        "git",
        "commit",
        "-m",
        f"docs: update changelog for {args.version}",
        external=True,
    )

    # Bump version and create tag
    session.install("bump2version")
    session.run(
        "bump2version",
        "--new-version",
        args.version,
        "--message",
        f"Release {args.version}: {args.description}",
        "patch",
        external=True,
    )

    # Verify tag creation
    tags = session.run(
        "git", "tag", "-l", f"v{args.version}", external=True, silent=True
    )
    if not tags:
        session.run(
            "git",
            "tag",
            "-a",
            f"v{args.version}",
            "-m",
            f"Release {args.version}: {args.description}",
            external=True,
        )

    # Push changes
    session.run("git", "push", "--follow-tags", external=True)


def update_changelog(version, description):
    """This function updates the changelog file with the new version and description

    Parameters
    ----------
    version : str
        version number
    description : str
        version description
    """
    import os
    import pathlib
    import shutil

    path = pathlib.Path(__file__).parent.absolute()
    changelog_file = os.path.join(path, "CHANGELOG.rst")
    docs_path = os.path.join(path, "docs/source")

    # open current changelog
    with open(changelog_file, "r+") as f:
        text = f.readlines()

    # prepare the new version title
    versioning_format = [
        "[{}_] - {}\n".format(version, description),
        "=======================================================\n",
    ]

    # add the new version title to the changelog
    for i, line in enumerate(versioning_format):
        text.insert(9 + i, line)

    # find the last index of the keepachangelog link
    last_index = [i for i, s in enumerate(text) if ".. _keepachangelog" in s][
        0
    ]

    # prepare the link to the new version
    link = ".. _{}: https://github.com/arielmission-space/ExoSim2.0/releases/tag/v{}\n".format(
        version, version
    )

    # add the link to the changelog
    text.insert(last_index - 1, link)

    # overwrite the changelog
    with open(changelog_file, "w+") as file_obj:
        file_obj.writelines(text)

    # Copy changelog to docs
    shutil.copy(changelog_file, docs_path)


def enable_full_tests():
    """this function enables the full test suite by changing the fast_test flag to False in the inputs.py file"""
    import os
    import pathlib

    path = pathlib.Path(__file__).parent.absolute()
    test_input_file = os.path.join(path, "tests/inputs.py")

    # open current input file
    with open(test_input_file, "r+") as f:
        text = f.readlines()

    # find the input keyword
    last_index = [i for i, s in enumerate(text) if "fast_test" in s][0]

    text[last_index] = "fast_test = False\n"

    with open(test_input_file, "w+") as file_obj:
        file_obj.writelines(text)


@nox.session
def sync_public(session: nox.Session) -> None:
    """Sync with public repository before release.

    Workflow:
    1. check if on develop branch
    2. add public remote if not exists
    3. pull changes from public
    4. merge changes from public/main into develop

    Usage:
        nox -s sync_public
    """
    # check if on develop branch
    current_branch = session.run(
        "git", "rev-parse", "--abbrev-ref", "HEAD", external=True, silent=True
    )

    if current_branch.strip() != "develop":
        session.error(
            "⚠ Devi essere sul branch 'develop' per eseguire sync_public"
        )

    # Configure git
    session.run(
        "git", "config", "user.name", "github-actions[bot]", external=True
    )
    session.run(
        "git",
        "config",
        "user.email",
        "github-actions[bot]@users.noreply.github.com",
        external=True,
    )

    # Add public remote if not exists
    try:
        session.run(
            "git",
            "remote",
            "add",
            "public",
            "https://github.com/arielmission-space/ExoSim2-public.git",
            external=True,
        )
    except Exception:
        # Remote might already exist
        pass

    # Fetch and merge
    session.run("git", "fetch", "public", external=True)
    try:
        # Allow unrelated histories merge
        session.run(
            "git",
            "merge",
            "public/main",
            "--no-ff",
            "--allow-unrelated-histories",
            external=True,
        )
        print("✓ Successfully merged changes from public repository")
    except Exception as e:
        print("⚠ No changes to merge or merge conflict")
        try:
            session.run("git", "merge", "--abort", external=True)
        except Exception:
            # No merge to abort
            pass
        raise e
