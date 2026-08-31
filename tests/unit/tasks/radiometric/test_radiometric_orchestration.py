"""
Behavioural tests for the radiometric-model orchestration helpers and the
spectral-binning task: the small ``utils`` functions that glue the per-task
models into the radiometric pipeline.
"""

from collections import OrderedDict

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.radiometric.estimate_spectral_binning import (
    EstimateSpectralBinning,
)
from exosim.tasks.radiometric.utils.compute_multiaccum import compute_multiaccum
from exosim.tasks.radiometric.utils.compute_photon_noise import compute_photon_noise
from exosim.tasks.radiometric.utils.create_table import create_table
from exosim.tasks.radiometric.utils.update_total_noise import update_total_noise


def _photometer(name="phot", wl_min=1.0, wl_max=2.0):
    return {
        "value": name,
        "type": "photometer",
        "wl_min": wl_min * u.um,
        "wl_max": wl_max * u.um,
    }


class TestEstimateSpectralBinning:
    def test_photometer_single_bin(self):
        table = EstimateSpectralBinning()(parameters=_photometer())
        assert len(table) == 1
        assert table["wavelength"][0].to_value(u.um) == pytest.approx(1.5)
        assert table["bandwidth"][0].to_value(u.um) == pytest.approx(1.0)
        assert table["left_bin_edge"][0].to_value(u.um) == pytest.approx(1.0)
        assert table["right_bin_edge"][0].to_value(u.um) == pytest.approx(2.0)

    def test_spectrometer_fixed_r_uses_grid(self):
        params = {
            "value": "spec",
            "type": "spectrometer",
            "targetR": 50,
            "wl_min": 1.0 * u.um,
            "wl_max": 2.0 * u.um,
        }
        table = EstimateSpectralBinning()(parameters=params)
        assert len(table) > 1
        # bins follow the resolving power: dlambda / lambda ~ 1/R
        ratio = (table["bandwidth"] / table["wavelength"]).to_value(
            u.dimensionless_unscaled
        )
        assert np.allclose(ratio, 1.0 / 50, rtol=0.05)

    def test_spectrometer_native_uses_wl_grid(self):
        wl_grid = np.linspace(1.0, 2.0, 16) * u.um
        params = {"value": "spec", "type": "spectrometer", "targetR": "native"}
        table = EstimateSpectralBinning()(parameters=params, wl_grid=wl_grid)
        assert len(table) == 16
        np.testing.assert_allclose(
            table["wavelength"].to_value(u.um), wl_grid.to_value(u.um)
        )

    def test_missing_target_r_defaults_to_native(self):
        wl_grid = np.linspace(1.0, 2.0, 8) * u.um
        params = {"value": "spec", "type": "spectrometer"}
        table = EstimateSpectralBinning()(parameters=params, wl_grid=wl_grid)
        assert len(table) == 8

    def test_bad_target_r_raises(self):
        params = {
            "value": "spec",
            "type": "spectrometer",
            "targetR": "weird",
            "wl_min": 1.0 * u.um,
            "wl_max": 2.0 * u.um,
        }
        with pytest.raises(KeyError, match="targetR format unsupported"):
            EstimateSpectralBinning()(parameters=params)

    def test_unknown_channel_type_raises(self):
        with pytest.raises(AttributeError):
            EstimateSpectralBinning()(parameters={"value": "x", "type": "bolometer"})


class TestCreateTable:
    def test_single_channel(self):
        cfg = {"channel": _photometer()}
        table = create_table(cfg)
        assert {
            "ch_name",
            "wavelength",
            "bandwidth",
            "left_bin_edge",
            "right_bin_edge",
        }.issubset(table.colnames)
        assert list(table["ch_name"]) == ["phot"]

    def test_multiple_channels_are_stacked(self):
        cfg = {
            "channel": OrderedDict(
                {
                    "a": _photometer("a", 1.0, 2.0),
                    "b": _photometer("b", 3.0, 4.0),
                }
            )
        }
        table = create_table(cfg)
        assert sorted(table["ch_name"]) == ["a", "b"]


class TestComputeMultiaccum:
    def test_defaults_to_unity_gain_without_config(self):
        table = QTable({"ch_name": ["phot", "phot"], "wavelength": [1.0, 2.0] * u.um})
        out, view = compute_multiaccum(table, {"channel": _photometer()})
        assert np.all(out["multiaccum_read_gain"] == 1)
        assert np.all(out["multiaccum_shot_gain"] == 1)
        assert "multiaccum_shot_gain" in view.colnames

    def test_uses_multiaccum_parameters_when_present(self):
        params = _photometer()
        params["radiometric"] = {"multiaccum": {"n": 4, "m": 2, "tf": 1.0, "tg": 10.0}}
        table = QTable({"ch_name": ["phot"], "wavelength": [1.5] * u.um})
        out, _ = compute_multiaccum(table, {"channel": params})
        assert out["multiaccum_read_gain"][0] == pytest.approx(0.9)
        assert out["multiaccum_shot_gain"][0] > 1.0


class TestComputePhotonNoise:
    def test_adds_photon_noise_column_per_signal(self):
        table = QTable(
            {
                "ch_name": ["phot"],
                "wavelength": [1.5] * u.um,
                "source_signal_in_aperture": [100.0] * u.ct / u.s,
                "multiaccum_shot_gain": [1.0],
            }
        )
        cfg = {"channel": {**_photometer(), "radiometric": {}}}
        out = compute_photon_noise(table, cfg)
        assert "source_photon_noise" in out.colnames
        # sqrt(N) statistics: noise = sqrt(signal) in ct/s
        assert out["source_photon_noise"][0].to_value(u.ct / u.s) == pytest.approx(10.0)

    def test_observation_efficiency_scales_the_signal(self):
        cfg = {"channel": {**_photometer(), "radiometric": {}}}
        base = compute_photon_noise(
            QTable(
                {
                    "ch_name": ["phot"],
                    "wavelength": [1.5] * u.um,
                    "source_signal_in_aperture": [100.0] * u.ct / u.s,
                    "multiaccum_shot_gain": [1.0],
                }
            ),
            cfg,
        )["source_photon_noise"][0]
        scaled = compute_photon_noise(
            QTable(
                {
                    "ch_name": ["phot"],
                    "wavelength": [1.5] * u.um,
                    "source_signal_in_aperture": [100.0] * u.ct / u.s,
                    "multiaccum_shot_gain": [1.0],
                    "observation_efficiency": [0.25],
                }
            ),
            cfg,
        )["source_photon_noise"][0]
        assert (scaled / base).to_value(u.dimensionless_unscaled) == pytest.approx(0.5)


class TestUpdateTotalNoise:
    def test_adds_total_noise_column(self):
        table = QTable(
            {
                "ch_name": ["phot"],
                "wavelength": [1.5] * u.um,
                "source_signal_in_aperture": [1.0] * u.ct / u.s,
                "read_noise": [0.3] * u.hr**0.5,
                "dark_noise": [0.4] * u.hr**0.5,
            }
        )
        full, view = update_total_noise(table)
        assert "total_noise" in full.colnames
        assert view["total_noise"][0].value == pytest.approx(0.5)  # sqrt(.09+.16)
