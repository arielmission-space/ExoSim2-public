import argparse

import nox


@nox.session
def lint(session):
    """
    Run the linters.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "-a")


@nox.session(reuse_venv=True)
def docs(session):
    session.install(".")
    session.run("sphinx-build", "-b", "html", "docs/source", "build")


@nox.session(name="docs-live", reuse_venv=True)
def docs_live(session):
    session.install(".")
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
    """
    Kicks off an automated release process by creating and pushing a new tag.

    Invokes bump2version with the posarg setting the version.

    Usage:
    $ nox -s release -- new-version -- description
    """

    parser = argparse.ArgumentParser(description="Release a semver version.")
    parser.add_argument(
        "version",
        type=str,
        nargs=1,
        help="The version to release. Must be structured as [major.minor.patch-release.update].",
    )
    parser.add_argument(
        "description",
        type=str,
        nargs=1,
        help="Short description of the version to release.",
    )
    args: argparse.Namespace = parser.parse_args(args=session.posargs)
    version: str = args.version.pop()
    description: str = args.description.pop()

    # If we get here, we should be good to go
    # Let's do a final check for safety
    confirm = input(
        f"You are about to bump the {version!r} version, described as: {description!r}.\nAre you sure? [y/n]: "
    )

    # Abort on anything other than 'y'
    if confirm.lower().strip() != "y":
        session.error(
            f"You said no when prompted to bump the {version!r} version."
        )

    enable_full_tests()

    # run pre-commit to ensure all checks pass
    session.install("pre-commit")
    session.log(f"pre-commit")
    session.run("pre-commit", "run", "-a")

    # update changelogs
    update_changelog(version, description)

    # add changelog updates to git
    session.run("git", "add", "-u", external=True)

    # find current version from .bumpversion.cfg
    with open(".bumpversion.cfg") as f:
        text = f.readlines()
    current_version = (
        [l for l in text if "current_version" in l][0]
        .replace("current_version = ", "")
        .replace("\n", "")
    )

    # commit the changelog updates
    session.install("bump2version")
    session.log(f"Bumping {current_version!r} to {version!r} version")
    session.run(
        "bump2version",
        "--current-version",
        current_version,
        "--new-version",
        version,
        "--allow-dirty",
        "patch",
        external=True,
    )

    # push the new tag
    session.log("Pushing the new tag")
    session.run("git", "push", external=True)
    session.run("git", "push", "--tags", external=True)


def update_changelog(version, description):
    """This function updates the changelog file with the new version and description

    Parameters
    ----------
    version : str
        version number
    description : str
        version description
    """
    import pathlib
    import os
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
    link = ".. _{}: https://github.com/arielmission-space/ExoSim2-public/releases/tag/v{}\n".format(
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
    import pathlib
    import os

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
