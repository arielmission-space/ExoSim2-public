"""
Behavioural tests for :class:`~exosim.tasks.parse.parse_source.ParseSource` and
:class:`~exosim.tasks.parse.parse_source.ParseSources`: they turn a source
description dictionary into rebinned :class:`~exosim.models.signal.Sed` objects,
optionally estimating ``logg`` from the stellar mass and writing to an output
file.

(``ParseZodi`` and ``ParseOpticalElement`` are exercised in
``tests/unit/tasks/test_foreground_modeling.py`` and
``tests/unit/tasks/test_optical_path_parsing.py``.)
"""

import os
from collections import OrderedDict

import astropy.units as u
import numpy as np
import pytest
from astropy import constants as cc

import exosim.output as output
from exosim.models.signal import Sed
from exosim.tasks.parse.parse_source import ParseSource, ParseSources


@pytest.fixture
def grids():
    wl = np.linspace(0.5, 7.8, 300) * u.um
    tt = np.linspace(0.0, 1.0, 4) * u.hr
    return wl, tt


def _planck_source():
    return {
        "value": "HD 209458",
        "source_type": "planck",
        "R": 1.18 * u.R_sun,
        "D": 47 * u.pc,
        "T": 6086 * u.K,
    }


class TestParseSource:
    def test_planck_source_is_parsed_and_rebinned(self, grids):
        wl, tt = grids
        out = ParseSource()(parameters=_planck_source(), wavelength=wl, time=tt)

        assert set(out) == {"HD 209458"}
        sed = out["HD 209458"]
        assert isinstance(sed, Sed)
        np.testing.assert_allclose(np.asarray(sed.spectral), wl.to_value(u.um))
        assert sed.data.shape[0] == tt.size
        assert sed.metadata["name"] == "HD 209458"
        assert sed.metadata["parsed_parameters"]["T"] == 6086 * u.K

    def test_logg_is_estimated_from_mass_and_radius(self, grids):
        wl, tt = grids
        params = _planck_source()
        params["M"] = 1.0 * u.M_sun
        out = ParseSource()(parameters=params, wavelength=wl, time=tt)

        g = (cc.G * (1.0 * u.M_sun).si / (1.18 * u.R_sun).si ** 2).to(u.cm / u.s**2)
        expected_logg = np.log10(g.value)
        assert out["HD 209458"].metadata["logg"] == pytest.approx(expected_logg)

    def test_missing_mass_leaves_logg_unset(self, grids):
        wl, tt = grids
        out = ParseSource()(parameters=_planck_source(), wavelength=wl, time=tt)
        assert out["HD 209458"].metadata["logg"] is None

    def test_sed_is_written_to_output_group(self, grids, tmp_path):
        wl, tt = grids
        fname = os.path.join(tmp_path, "sources.h5")
        with output.HDF5Output(fname) as o:
            ParseSource()(parameters=_planck_source(), wavelength=wl, time=tt, output=o)
        import h5py

        with h5py.File(fname, "r") as f:
            assert "sources/HD 209458" in f


class TestParseSources:
    def test_ordered_dict_of_sources_are_all_parsed(self, grids):
        wl, tt = grids
        sources = OrderedDict(
            {
                "HD 209458": _planck_source(),
                "GJ 1214": {
                    "value": "GJ 1214",
                    "source_type": "planck",
                    "R": 0.218 * u.R_sun,
                    "D": 13 * u.pc,
                    "T": 3026 * u.K,
                },
            }
        )
        out = ParseSources()(parameters=sources, wavelength=wl, time=tt)
        assert set(out) == {"HD 209458", "GJ 1214"}
        assert all(isinstance(s, Sed) for s in out.values())

    def test_single_source_dict_is_parsed(self, grids):
        wl, tt = grids
        out = ParseSources()(parameters=_planck_source(), wavelength=wl, time=tt)
        assert set(out) == {"HD 209458"}
        assert isinstance(out["HD 209458"], Sed)


class TestParseSourceExtraBranches:
    def test_online_database_lookup_is_used(self, grids):
        from unittest.mock import MagicMock, patch

        wl, tt = grids
        params = {
            "value": "HD 209458",
            "source_type": "planck",
            "online_database": {"url": "https://exodb.example/api/v1/star"},
        }
        fake_json = {
            "data": {
                "Properties": {
                    "Radius": {"value": 1.18, "unit": "Rsun"},
                    "Effective Temperature": {"value": 6086, "unit": "K"},
                    "Distance from Earth": {"value": 47, "unit": "pc"},
                    "Metallicity": {"value": 0.0},
                    "Mass": {"value": 1.15, "unit": "Msun"},
                }
            }
        }
        with patch(
            "exosim.tasks.parse.parse_source.requests.post",
            return_value=MagicMock(json=lambda: fake_json),
        ) as post:
            out = ParseSource()(parameters=params, wavelength=wl, time=tt)
        post.assert_called_once()
        sed = out["HD 209458"]
        # parameters were filled in from the database response
        assert sed.metadata["T"].to_value(u.K) == 6086
        assert sed.metadata["logg"] == pytest.approx(
            np.log10(
                (cc.G * (1.15 * u.M_sun).si / (1.18 * u.R_sun).si ** 2).to_value(
                    u.cm / u.s**2
                )
            )
        )

    def test_missing_mass_and_radius_warns_but_still_parses(self, grids):
        wl, tt = grids
        warnings = []
        task = ParseSource()
        params = {"value": "bare", "source_type": "planck", "T": 5000 * u.K}
        # no logg, no M, no R for the star -> logg stays None, a warning is logged
        orig = task.warning
        task.warning = lambda msg, *a, **k: (warnings.append(msg), orig(msg))
        with pytest.raises((KeyError, TypeError)):
            # PrepareSed needs R for a planck star, so it raises downstream
            task(parameters=params, wavelength=wl, time=tt)
        assert any("mass (M) and radius (R)" in w for w in warnings)
