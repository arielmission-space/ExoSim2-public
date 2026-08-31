"""
Behavioural tests for the package-metadata helpers in ``exosim.__about__``:
version resolution order, the git-info reader, and the release/dev predicates.
"""

import pytest

import exosim.__about__ as about


class TestVersionResolution:
    def test_explicit_project_version_wins(self):
        assert about._resolve_version({"version": "9.9.9"}, "exosim") == "9.9.9"

    def test_falls_back_to_unknown(self, monkeypatch):
        monkeypatch.setattr(about, "_read_version_file", lambda: None)
        monkeypatch.setattr(about, "_read_installed_version", lambda _n: None)
        assert about._resolve_version({}, "not-a-real-package") == "unknown"

    def test_installed_version_of_a_known_package(self):
        v = about._read_installed_version("pytest")
        assert isinstance(v, str)
        assert v[0].isdigit()

    def test_installed_version_of_missing_package_is_none(self):
        assert about._read_installed_version("definitely-not-installed-xyz") is None

    def test_read_pyproject_toml_returns_project_table(self):
        data = about._read_pyproject_toml()
        assert data["project"]["name"] == "exosim"

    def test_read_pyproject_toml_without_a_toml_library_raises(self, monkeypatch):
        monkeypatch.setattr(about, "tomllib", None)
        with pytest.raises(ImportError, match="No TOML library"):
            about._read_pyproject_toml()

    def test_read_version_file_swallows_import_errors(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def boom(name, *a, **k):
            if name == "exosim._version":
                raise ImportError("gone")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", boom)
        assert about._read_version_file() is None

    def test_resolve_version_uses_the_scm_version_file(self, monkeypatch):
        monkeypatch.setattr(about, "_read_version_file", lambda: "1.2.3")
        assert about._resolve_version({}, "exosim") == "1.2.3"


class TestGitInfo:
    def test_returns_branch_and_commit_in_a_checkout(self):
        branch, commit = about._get_git_info()
        # this test runs inside the repository, so both are populated
        assert isinstance(branch, str)
        assert commit is None or (isinstance(commit, str) and len(commit) >= 7)

    def test_no_git_folder_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            about, "__file__", str(tmp_path / "src" / "exosim" / "x.py")
        )
        assert about._get_git_info() == (None, None)

    def test_reads_branch_and_commit_from_a_fake_git_dir(self, tmp_path, monkeypatch):
        root = tmp_path
        git = root / ".git"
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/feature-x\n")
        (git / "refs" / "heads" / "feature-x").write_text("abc123def456\n")
        monkeypatch.setattr(
            about, "__file__", str(root / "src" / "exosim" / "__about__.py")
        )
        assert about._get_git_info() == ("feature-x", "abc123def456")

    def test_detached_head_reports_the_commit_directly(self, tmp_path, monkeypatch):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("0123456789abcdef\n")
        monkeypatch.setattr(
            about, "__file__", str(tmp_path / "src" / "exosim" / "__about__.py")
        )
        assert about._get_git_info() == ("HEAD", "0123456789abcdef")

    def test_missing_ref_file_yields_a_none_commit(self, tmp_path, monkeypatch):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/packed-branch\n")
        monkeypatch.setattr(
            about, "__file__", str(tmp_path / "src" / "exosim" / "__about__.py")
        )
        assert about._get_git_info() == ("packed-branch", None)


class TestVersionPredicates:
    def test_dev_and_release_are_mutually_exclusive(self):
        assert about.is_development_version() != about.is_release_version()

    def test_module_exposes_the_expected_metadata(self):
        assert about.__pkg_name__ == "exosim"
        assert isinstance(about.__version__, str)
        assert isinstance(about.__version_info__, tuple)
        assert about.__title__ == "ExoSim 2"
