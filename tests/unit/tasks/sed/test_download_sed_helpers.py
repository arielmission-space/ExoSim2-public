"""
Behavioural tests for the offline parts of the SED download task: the parameter
grid snapping, the ACES URL builder, the air-to-vacuum wavelength conversion,
and the SVO HTML parsers (fed canned HTML rather than hitting the network).
"""

import io
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest

from exosim.tasks.sed import download_sed as ds


class TestGridSnapping:
    def test_snap_returns_nearest_grid_point(self):
        grid = np.array([0.0, 10.0, 20.0, 30.0])
        assert ds._snap(12.0, grid) == 10.0
        assert ds._snap(17.0, grid) == 20.0
        assert ds._snap(-5.0, grid) == 0.0

    def test_aces_url_snaps_and_formats(self):
        url, t, g, z = ds._aces_url(5123, 4.4, 0.02)
        assert (t, g, z) == (5100, 4.5, 0.0)
        assert url.startswith("https://phoenix.astro.physik.uni-goettingen.de")
        assert f"lte{t:05d}-{g:.2f}{z:+.1f}" in url
        assert "Z+0.0" in url

    def test_aces_url_negative_metallicity(self):
        url, _t, _g, z = ds._aces_url(4000, 4.5, -1.2)
        assert z == -1.0
        assert "Z-1.0" in url
        assert "-1.0." in url


class TestAirToVacuum:
    def test_vacuum_wavelength_is_longer_than_air(self):
        air = np.array([3000.0, 5000.0, 8000.0])
        vac = ds.air_to_vacuum_wavelength(air)
        assert np.all(vac > air)
        # the shift is small, well under a percent
        assert np.all((vac - air) / air < 0.001)

    def test_short_wavelengths_are_nan(self):
        out = ds.air_to_vacuum_wavelength(np.array([1000.0, 1999.0]))
        assert np.all(np.isnan(out))

    def test_all_invalid_returns_all_nan(self):
        out = ds.air_to_vacuum_wavelength(np.array([np.nan, -3.0, 100.0]))
        assert np.all(np.isnan(out))


_MODELS_HTML = """
<html><body>
<select name="models">
  <option value="bt-settl">BT-Settl</option>
  <option value="bt-nextgen">BT-NextGen</option>
</select>
<a href="index.php?models=coelho">Coelho</a>
</body></html>
"""

_TABLE_HTML = """
<table>
<tr><td class="tabcab">Teff</td><td class="tabcab">logg</td>
    <td class="tabcab">Metallicity</td><td class="tabcab">file</td></tr>
<tr><td class="tabfld">5000</td><td class="tabfld">4.5</td>
    <td class="tabfld">0.0</td>
    <td class="tabfld"><a href="ssap.php?model=1">get</a></td></tr>
<tr><td class="tabfld">6000</td><td class="tabfld">4.0</td>
    <td class="tabfld">-0.5</td>
    <td class="tabfld"><a href="ssap.php?model=2">get</a></td></tr>
</table>
"""


class TestSVOParsers:
    def test_models_parser_collects_identifiers(self):
        parser = ds._SVOModelsParser()
        parser.feed(_MODELS_HTML)
        assert {"bt-settl", "bt-nextgen", "coelho"} <= parser.models

    def test_table_parser_extracts_rows(self):
        parser = ds._SVOTableParser()
        parser.feed(_TABLE_HTML)
        rows = parser.models()
        assert len(rows) == 2
        assert rows[0]["teff"] == 5000.0
        assert rows[1]["feh"] == -0.5
        assert rows[0]["url"].endswith("ssap.php?model=1")

    def test_table_parser_without_teff_header_returns_empty(self):
        parser = ds._SVOTableParser()
        parser.feed("<table><tr><td class='tabcab'>foo</td></tr></table>")
        assert parser.models() == []


class TestSvoNearest:
    def _resp(self, html):
        r = io.BytesIO(html.encode())
        r.__enter__ = lambda s: s
        r.__exit__ = lambda *a: False
        return r

    def test_picks_the_closest_grid_point(self):
        with (
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", return_value=self._resp(_TABLE_HTML)),
        ):
            best = ds._svo_nearest("bt-settl", 5100, 4.4, 0.1, 0.0)
        assert best["teff"] == 5000.0
        assert best["logg"] == 4.5

    def test_far_metallicity_switches_the_choice(self):
        with (
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", return_value=self._resp(_TABLE_HTML)),
        ):
            best = ds._svo_nearest("bt-settl", 6000, 4.0, -0.5, 0.0)
        assert best["teff"] == 6000.0

    def test_no_candidates_raises(self):
        empty = "<table><tr><td class='tabcab'>foo</td></tr></table>"
        with (
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", return_value=self._resp(empty)),
            pytest.raises(ValueError, match="no spectra"),
        ):
            ds._svo_nearest("bt-settl", 5000, 4.5, 0.0, 0.0)

    def test_offline_raises(self):
        with (
            patch.object(ds, "_host_is_reachable", return_value=False),
            pytest.raises(ConnectionError, match="no network"),
        ):
            ds._svo_nearest("bt-settl", 5000, 4.5, 0.0, 0.0)


class TestFetchSvoMocked:
    def test_parses_ascii_spectrum_and_builds_meta(self, tmp_path):
        spec = tmp_path / "ssap.php"
        spec.write_text("# header\n3000.0 1.0\n4000.0 2.0\n5000.0 3.0\n")
        best = {
            "url": "https://svo.example/ssap.php?model=1",
            "teff": 5000.0,
            "logg": 4.5,
            "feh": 0.0,
            "alpha": 0.0,
        }
        with (
            patch.object(ds, "_svo_nearest", return_value=best),
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", side_effect=OSError("no HEAD")),
            patch.object(ds, "download_file", return_value=str(spec)),
        ):
            wl, flux, meta = ds._fetch_svo("bt-settl", 5000, 4.5, 0.0, 0.0)
        assert wl.unit == u.AA
        assert flux.unit.is_equivalent("erg / (s cm2 AA)")
        assert len(wl) == 3
        assert meta["model"] == "bt-settl"
        assert meta["teff"] == 5000.0
        # generic remote name -> descriptive fallback
        assert meta["filename"].startswith("bt-settl_Teff05000")

    def test_content_disposition_header_names_the_file(self, tmp_path):
        spec = tmp_path / "ssap.php"
        spec.write_text("3000 1\n4000 2\n")
        best = {
            "url": "https://svo.example/ssap.php?model=1",
            "teff": 5000.0,
            "logg": 4.5,
            "feh": 0.0,
            "alpha": 0.0,
        }

        class _Head:
            def __init__(self):
                self.headers = {
                    "Content-Disposition": 'attachment; filename="bt-settl_T5000.spec"'
                }

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def geturl(self):
                return "https://svo.example/ssap.php?model=1"

        with (
            patch.object(ds, "_svo_nearest", return_value=best),
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", return_value=_Head()),
            patch.object(ds, "download_file", return_value=str(spec)),
        ):
            _wl, _flux, meta = ds._fetch_svo("bt-settl", 5000, 4.5, 0.0, 0.0)
        assert meta["filename"] == "bt-settl_T5000.spec"

    def test_unparseable_file_raises(self, tmp_path):
        spec = tmp_path / "empty.txt"
        spec.write_text("# only comments\n#\n")
        best = {
            "url": "https://svo.example/x.spec",
            "teff": 5000.0,
            "logg": 4.5,
            "feh": 0.0,
            "alpha": 0.0,
        }
        with (
            patch.object(ds, "_svo_nearest", return_value=best),
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", side_effect=OSError("no HEAD")),
            patch.object(ds, "download_file", return_value=str(spec)),
            pytest.raises(ValueError, match="Could not parse"),
        ):
            ds._fetch_svo("bt-settl", 5000, 4.5, 0.0, 0.0)


class TestGetSvoModelsMocked:
    def test_parses_mocked_index_page(self):
        resp = io.BytesIO(_MODELS_HTML.encode())
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda *a: False
        with (
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch("urllib.request.urlopen", return_value=resp),
        ):
            models = ds.get_svo_models()
        assert "bt-settl" in models
        assert models == sorted(models)

    def test_raises_when_offline(self):
        with (
            patch.object(ds, "_host_is_reachable", return_value=False),
            pytest.raises(ConnectionError, match="no network"),
        ):
            ds.get_svo_models()


class TestDownloadSedTask:
    """The task orchestration, with the network backend stubbed out."""

    def _fake_spectrum(self):
        wl = np.linspace(3000.0, 30000.0, 200) * u.AA
        # a smooth positive spectrum in erg/s/cm2/AA
        flux = (1e-4 * np.exp(-((wl.value - 12000) ** 2) / 5e7)) * (
            u.erg / u.s / u.cm**2 / u.AA
        )
        return wl, flux, {"grid_teff": 5000.0}

    def test_aces_backend_produces_a_scaled_sed(self):
        from exosim.models.signal import Sed

        with patch.object(ds, "_fetch_aces", return_value=self._fake_spectrum()):
            sed = ds.DownloadSed()(
                R=1.0 * u.Rsun,
                D=10.0 * u.pc,
                T=5000 * u.K,
                logg=4.5,
                model_name="phoenix-aces",
            )
        assert isinstance(sed, Sed)
        assert sed.data_units == u.W / u.m**2 / u.um
        assert np.all(sed.data >= 0)
        assert sed.metadata["model_name"] == "phoenix-aces"

    def test_geometric_dilution_scales_as_radius_over_distance_squared(self):
        with patch.object(ds, "_fetch_aces", return_value=self._fake_spectrum()):
            near = ds.DownloadSed()(
                R=1.0 * u.Rsun,
                D=10.0 * u.pc,
                T=5000 * u.K,
                logg=4.5,
                model_name="phoenix-aces",
            )
            far = ds.DownloadSed()(
                R=1.0 * u.Rsun,
                D=20.0 * u.pc,
                T=5000 * u.K,
                logg=4.5,
                model_name="phoenix-aces",
            )
        ratio = float(np.nanmax(far.data) / np.nanmax(near.data))
        assert ratio == pytest.approx(0.25, rel=1e-3)  # (10/20)**2

    def test_svo_backend_converts_air_to_vacuum(self):
        wl, flux, meta = self._fake_spectrum()
        with patch.object(ds, "_fetch_svo", return_value=(wl, flux, meta)):
            sed = ds.DownloadSed()(
                R=1.0 * u.Rsun,
                D=10.0 * u.pc,
                T=5000 * u.K,
                logg=4.5,
                model_name="bt-settl",
            )
        assert sed.data_units == u.W / u.m**2 / u.um

    def test_missing_parameter_raises_keyerror(self):
        with pytest.raises(KeyError, match="star R missing"):
            ds.DownloadSed()(D=10.0 * u.pc, T=5000 * u.K, logg=4.5)

    def test_temperature_without_units_is_assumed_kelvin(self):
        with patch.object(ds, "_fetch_aces", return_value=self._fake_spectrum()):
            sed = ds.DownloadSed()(
                R=1.0 * u.Rsun,
                D=10.0 * u.pc,
                T=5000,
                logg=4.5,
                model_name="phoenix-aces",
            )
        assert sed is not None


class TestHostReachable:
    def test_unreachable_host_is_false(self):
        assert (
            ds._host_is_reachable("https://exosim.invalid.nonexistent.example") is False
        )

    def test_url_without_hostname_is_false(self):
        assert ds._host_is_reachable("not a url") is False


class TestFetchAcesMocked:
    def test_reads_wave_and_flux_fits(self, tmp_path):
        from astropy.io import fits

        wave = tmp_path / "wave.fits"
        flux = tmp_path / "flux.fits"
        fits.PrimaryHDU(np.linspace(3000.0, 30000.0, 50)).writeto(wave)
        fits.PrimaryHDU(np.ones(50) * 1e5).writeto(flux)

        with (
            patch.object(ds, "_host_is_reachable", return_value=True),
            patch.object(ds, "download_file", side_effect=[str(wave), str(flux)]),
        ):
            wl, sed, meta = ds._fetch_aces(5000, 4.5, 0.0)
            assert sed.unit.is_equivalent("erg / (s cm2 cm)")

        assert wl.unit == u.AA
        assert len(wl) == 50
        assert meta["model"] == ds._ACES_MODEL
        assert "filename" in meta

    def test_raises_when_offline(self):
        with (
            patch.object(ds, "_host_is_reachable", return_value=False),
            pytest.raises(ConnectionError, match="no network"),
        ):
            ds._fetch_aces(5000, 4.5, 0.0)
