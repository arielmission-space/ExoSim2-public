"""Tests for package metadata and version resolution logic."""

import exosim.__about__ as about


class TestVersionResolution:
    """Test package version resolution logic."""

    def test_resolve_version_prefers_explicit_project_value(self, monkeypatch):
        """An explicit ``version`` key wins over every other source."""
        monkeypatch.setattr(about, "_read_version_file", lambda: "8.8.8")
        monkeypatch.setattr(about, "_read_installed_version", lambda _: "9.9.9")

        resolved = about._resolve_version({"version": "2.1.1-dev2"}, "exosim")

        assert resolved == "2.1.1-dev2"

    def test_resolve_version_prefers_scm_version_file(self, monkeypatch):
        """The setuptools-scm ``_version.py`` is used before installed metadata."""
        monkeypatch.setattr(about, "_read_version_file", lambda: "2.2.0.dev5")
        monkeypatch.setattr(about, "_read_installed_version", lambda _: "2.1.1")

        resolved = about._resolve_version({}, "exosim")

        assert resolved == "2.2.0.dev5"

    def test_resolve_version_falls_back_to_installed_metadata(self, monkeypatch):
        """Installed metadata is used when no version file is available."""
        monkeypatch.setattr(about, "_read_version_file", lambda: None)
        monkeypatch.setattr(about, "_read_installed_version", lambda _: "2.1.1")

        resolved = about._resolve_version({}, "exosim")

        assert resolved == "2.1.1"

    def test_resolve_version_returns_unknown_when_no_sources(self, monkeypatch):
        """Unknown is returned when no metadata source provides a version."""
        monkeypatch.setattr(about, "_read_version_file", lambda: None)
        monkeypatch.setattr(about, "_read_installed_version", lambda _: None)

        resolved = about._resolve_version({}, "exosim")

        assert resolved == "unknown"
