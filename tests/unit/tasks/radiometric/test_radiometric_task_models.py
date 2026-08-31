"""
Behavioural tests for the radiometric ``model``-level tasks.

These tests call the real tasks through their public ``__call__`` interface with
realistic inputs and check the physics: the multiaccum gain formulae, the
read/dark-current noise scaling laws, the quadrature combination of noise terms,
aperture photometry on a known array, and the error paths guarding each task.
"""

from collections import OrderedDict

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.radiometric.aperture_photometry import AperturePhotometry
from exosim.tasks.radiometric.compute_constant_dark_current_noise import (
    ComputeConstantDarkCurrentNoise,
)
from exosim.tasks.radiometric.compute_constant_read_noise import (
    ComputeConstantReadNoise,
)
from exosim.tasks.radiometric.compute_custom_noise import ComputeCustomNoise
from exosim.tasks.radiometric.compute_integration_time import ComputeIntegrationTime
from exosim.tasks.radiometric.compute_observation_efficiency import (
    ComputeObservationEfficiency,
)
from exosim.tasks.radiometric.compute_total_noise import ComputeTotalNoise
from exosim.tasks.radiometric.load_apertures import LoadApertures
from exosim.tasks.radiometric.multiaccum import Multiaccum


class TestMultiaccum:
    """The MULTIACCUM gain factors (Rauscher & Fox 2007 / Batalha 2017)."""

    @staticmethod
    def _expected(n, m, tf, tg):
        read = 12.0 * (n - 1.0) / (m * (n**2 + n))
        shot = (
            6.0 / 5.0 * (n**2 + 1.0) / (n**2 + n) * (n - 1.0) * tg
            + (m**2 - 1.0) * (n - 1.0) / (m * (n**2 + n)) * tf
        )
        return read, shot

    def test_matches_closed_form(self):
        read, shot = Multiaccum()(parameters={"n": 4, "m": 2, "tf": 1.0, "tg": 10.0})
        exp_read, exp_shot = self._expected(4, 2, 1.0, 10.0)
        assert read == pytest.approx(exp_read)
        assert shot == pytest.approx(exp_shot)
        assert read == pytest.approx(0.9)

    def test_n_below_two_is_forced_to_cds(self):
        base = {"m": 2, "tf": 1.0, "tg": 10.0}
        read1, shot1 = Multiaccum()(parameters={**base, "n": 1})
        read2, shot2 = Multiaccum()(parameters={**base, "n": 2})
        assert read1 == pytest.approx(read2)
        assert shot1 == pytest.approx(shot2)

    def test_more_groups_reduce_read_gain(self):
        base = {"m": 2, "tf": 1.0, "tg": 10.0}
        read_few, _ = Multiaccum()(parameters={**base, "n": 3})
        read_many, _ = Multiaccum()(parameters={**base, "n": 30})
        assert read_many < read_few


class TestComputeIntegrationTime:
    """Integration time is the shortest saturation time on the array."""

    def _table(self):
        return QTable(
            {
                "ch_name": ["a", "a", "b"],
                "saturation_time": [12.0, 5.0, 30.0] * u.s,
            }
        )

    def test_minimum_over_whole_table(self):
        out = ComputeIntegrationTime()(
            saturation_table=self._table(), description={}, channel_name=None
        )
        assert np.allclose(out.to_value(u.s), 5.0)
        assert out.shape == (3,)

    def test_minimum_per_channel(self):
        out = ComputeIntegrationTime()(
            saturation_table=self._table(), description={}, channel_name="b"
        )
        assert np.allclose(out.to_value(u.s), 30.0)
        assert out.shape == (1,)

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="Saturation time is missing"):
            ComputeIntegrationTime()(
                saturation_table=QTable({"ch_name": ["a"]}),
                description={},
                channel_name=None,
            )


class TestComputeObservationEfficiency:
    """The constant observation-efficiency model and its default."""

    def test_reads_value_from_description(self):
        eff = ComputeObservationEfficiency()(
            radiometric_table=QTable({"ch_name": ["a"]}),
            description={"radiometric": {"observation_efficiency": 0.8}},
            channel_name=None,
        )
        assert eff == pytest.approx(0.8)
        assert isinstance(eff, float)

    def test_defaults_to_one(self):
        from unittest.mock import patch

        task = ComputeObservationEfficiency()
        with patch.object(task, "warning") as warn:
            eff = task.model(QTable({"ch_name": ["a"]}), {}, None)
        assert eff == 1.0
        warn.assert_called_once()

    def test_execute_rejects_non_float_output(self):
        task = ComputeObservationEfficiency()
        task.model = lambda *a, **k: np.array([1.0, 1.0])
        with pytest.raises(TypeError, match="wrong output format"):
            task(
                radiometric_table=QTable({"ch_name": ["a"]}),
                description={},
                channel_name=None,
            )


class TestComputeTotalNoise:
    """Relative-noise columns combine in quadrature."""

    def test_quadrature_of_relative_terms(self):
        table = QTable(
            {
                "wavelength": [1.0, 2.0] * u.um,
                "source_signal_in_aperture": [1.0, 1.0] * u.ct / u.s,
                "read_noise": [0.1, 0.1] * u.hr**0.5,
                "dark_noise": [0.2, 0.2] * u.hr**0.5,
            }
        )
        total = ComputeTotalNoise()(table=table)
        assert np.allclose(total.value, np.sqrt(0.1**2 + 0.2**2))

    def test_custom_columns_are_ignored(self):
        table = QTable(
            {
                "wavelength": [1.0] * u.um,
                "source_signal_in_aperture": [1.0] * u.ct / u.s,
                "read_noise": [0.3] * u.hr**0.5,
                "systematics_custom_noise": [10.0] * u.hr**0.5,
            }
        )
        total = ComputeTotalNoise()(table=table)
        assert total.value[0] == pytest.approx(0.3)


class TestComputeConstantReadNoise:
    """Read-noise variance = G . sigma_RN^2 . A / t_frame, scaled by 1/signal."""

    def _call(self, *, gain=1.0, sigma=10.0, area=100.0, frame=10.0, signal=2.0):
        table = QTable({"aperture_size": [area], "frame_time": [frame] * u.s})
        desc = {"detector": {"read_noise_sigma": sigma * u.ct}}
        noise_table, noise = ComputeConstantReadNoise()(
            signal=np.array([signal]) * u.ct / u.s,
            aperture_table=table,
            description=desc,
            multiaccum_gain=gain,
        )
        return noise_table, noise

    def test_scaling_laws(self):
        _, base = self._call()
        _, more_gain = self._call(gain=4.0)
        _, more_sigma = self._call(sigma=20.0)
        _, more_area = self._call(area=400.0)
        _, more_frame = self._call(frame=40.0)
        _, more_signal = self._call(signal=4.0)
        assert float((more_gain / base)[0]) == pytest.approx(2.0)
        assert float((more_sigma / base)[0]) == pytest.approx(2.0)
        assert float((more_area / base)[0]) == pytest.approx(2.0)
        assert float((more_frame / base)[0]) == pytest.approx(0.5)
        assert float((more_signal / base)[0]) == pytest.approx(0.5)

    def test_output_table_columns(self):
        noise_table, _ = self._call()
        assert set(noise_table.colnames) == {"read_noise_variance", "read_noise"}

    def test_missing_read_noise_sigma_raises(self):
        with pytest.raises(ValueError, match="Read noise sigma is missing"):
            ComputeConstantReadNoise()(
                signal=np.array([1.0]) * u.ct / u.s,
                aperture_table=QTable(
                    {"aperture_size": [1.0], "frame_time": [1.0] * u.s}
                ),
                description={"detector": {}},
                multiaccum_gain=1.0,
            )

    def test_missing_frame_time_raises(self):
        with pytest.raises(ValueError, match="Frame time is missing"):
            ComputeConstantReadNoise()(
                signal=np.array([1.0]) * u.ct / u.s,
                aperture_table=QTable({"aperture_size": [1.0]}),
                description={"detector": {"read_noise_sigma": 1.0 * u.ct}},
                multiaccum_gain=1.0,
            )

    def test_non_quantity_signal_raises(self):
        with pytest.raises(TypeError, match="Signal must be a Quantity"):
            ComputeConstantReadNoise()(
                signal=np.array([1.0]),
                aperture_table=QTable(
                    {"aperture_size": [1.0], "frame_time": [1.0] * u.s}
                ),
                description={"detector": {"read_noise_sigma": 1.0 * u.ct}},
                multiaccum_gain=1.0,
            )


class TestComputeConstantDarkCurrentNoise:
    """Dark-current noise ~ sqrt(G . dc_mean . A), scaled by 1/signal."""

    def _call(self, *, gain=1.0, dc=0.1, area=100.0, signal=2.0):
        table = QTable({"aperture_size": [area]})
        desc = {"detector": {"dark_current": True, "dc_mean": dc * u.ct / u.s}}
        return ComputeConstantDarkCurrentNoise()(
            signal=np.array([signal]) * u.ct / u.s,
            aperture_table=table,
            description=desc,
            multiaccum_gain=gain,
        )

    def test_scaling_laws(self):
        _, base = self._call()
        _, more_gain = self._call(gain=4.0)
        _, more_dc = self._call(dc=0.4)
        _, more_area = self._call(area=400.0)
        _, more_signal = self._call(signal=4.0)
        assert float((more_gain / base)[0]) == pytest.approx(2.0)
        assert float((more_dc / base)[0]) == pytest.approx(2.0)
        assert float((more_area / base)[0]) == pytest.approx(2.0)
        assert float((more_signal / base)[0]) == pytest.approx(0.5)

    def test_execute_produces_named_column(self):
        out_table, _ = self._call()
        assert "darkcurrent_noise" in out_table.colnames

    def test_missing_dc_mean_raises(self):
        with pytest.raises(ValueError, match="mean value is missing"):
            ComputeConstantDarkCurrentNoise()(
                signal=np.array([1.0]) * u.ct / u.s,
                aperture_table=QTable({"aperture_size": [1.0]}),
                description={"detector": {"dark_current": True}},
                multiaccum_gain=1.0,
            )

    def test_missing_dark_current_key_raises(self):
        with pytest.raises(ValueError, match="description is missing"):
            ComputeConstantDarkCurrentNoise()(
                signal=np.array([1.0]) * u.ct / u.s,
                aperture_table=QTable({"aperture_size": [1.0]}),
                description={"detector": {}},
                multiaccum_gain=1.0,
            )


class TestComputeCustomNoise:
    """User-defined noise: quadrature combination, scale factor, guard rails."""

    WL = np.array([1.0, 2.0, 3.0]) * u.um

    def test_no_description_returns_zero(self):
        table, total = ComputeCustomNoise()(wavelength=self.WL, description=None)
        assert len(table.colnames) == 0
        assert np.all(total.value == 0)

    def test_no_custom_noise_section_returns_zero(self):
        _, total = ComputeCustomNoise()(
            wavelength=self.WL, description={"radiometric": {}}
        )
        assert np.all(total.value == 0)

    def test_empty_wavelength_raises(self):
        with pytest.raises(ValueError, match="empty wavelength"):
            ComputeCustomNoise()(wavelength=np.array([]) * u.um, description=None)

    def test_single_source_with_scale(self):
        desc = {
            "radiometric": {
                "custom_noise": {
                    "value": "thermal",
                    "noise_level": {"value": 100.0, "scale": 1e-6},
                }
            }
        }
        table, total = ComputeCustomNoise()(wavelength=self.WL, description=desc)
        assert np.allclose(total.value, 1e-4)
        assert "thermal_noise" in table.colnames

    def test_multiple_sources_combine_in_quadrature(self):
        desc = {
            "radiometric": {
                "custom_noise": OrderedDict(
                    {
                        "a": {"name": "a", "noise_level": 50.0},
                        "b": {"name": "b", "noise_level": 30.0},
                        "c": {"name": "c", "noise_level": 20.0},
                    }
                )
            }
        }
        _, total = ComputeCustomNoise()(wavelength=self.WL, description=desc)
        assert np.allclose(total.value, np.sqrt(50.0**2 + 30.0**2 + 20.0**2))

    def test_bad_type_raises(self):
        desc = {"radiometric": {"custom_noise": [1, 2, 3]}}
        with pytest.raises(TypeError, match="OrderedDict or dict"):
            ComputeCustomNoise()(wavelength=self.WL, description=desc)

    def test_dict_contrib_without_a_noise_value_raises(self):
        desc = {"radiometric": {"custom_noise": {"name": "x"}}}
        with pytest.raises(KeyError, match="noise value not found"):
            ComputeCustomNoise()(wavelength=self.WL, description=desc)

    def test_spectral_data_is_rebinned_onto_the_grid(self):
        # a wavelength-dependent noise table, constant at 5e-4 hr**0.5
        data = {
            "wavelength": np.array([0.5, 1.5, 2.5, 3.5]) * u.um,
            "systematics": np.array([5e-4, 5e-4, 5e-4, 5e-4]) * u.hr**0.5,
        }
        desc = {"radiometric": {"custom_noise": {"name": "spec", "data": data}}}
        table, total = ComputeCustomNoise()(wavelength=self.WL, description=desc)
        assert any("spec" in c for c in table.colnames)
        assert np.allclose(total.value, 5e-4, rtol=1e-6)


class TestAperturePhotometry:
    """Rectangular aperture photometry sums the enclosed pixels."""

    def test_rectangular_sum_matches_block(self):
        img = np.zeros((20, 20))
        img[8:12, 8:12] = 1.0  # 16 unit pixels
        img = img * (u.ct / u.s)
        table = QTable(
            {
                "wavelength": [1.0] * u.um,
                "spectral_center": [9.5],
                "spectral_size": [4.0],
                "spatial_center": [9.5],
                "spatial_size": [4.0],
                "aperture_shape": ["rectangular"],
            }
        )
        phot = AperturePhotometry()(table=table, focal_plane=img)
        assert phot.unit == u.ct / u.s
        assert phot.value[0] == pytest.approx(16.0, rel=1e-6)

    def test_larger_aperture_collects_more(self):
        rng = np.random.default_rng(0)
        img = rng.random((30, 30)) * (u.ct / u.s)

        def _phot(size):
            table = QTable(
                {
                    "wavelength": [1.0] * u.um,
                    "spectral_center": [15.0],
                    "spectral_size": [size],
                    "spatial_center": [15.0],
                    "spatial_size": [size],
                    "aperture_shape": ["rectangular"],
                }
            )
            return AperturePhotometry()(table=table, focal_plane=img).value[0]

        assert _phot(10.0) > _phot(4.0)


_APERTURE_COLUMNS = (
    "spectral_center",
    "spectral_size",
    "spatial_center",
    "spatial_size",
    "aperture_shape",
    "aperture_size",
)


class TestLoadApertures:
    """LoadApertures reads a stored aperture table and validates its columns."""

    def test_reads_required_columns(self, tmp_path):
        path = tmp_path / "apertures.ecsv"
        QTable(
            {
                c: (["rectangular"] if c == "aperture_shape" else [1.0])
                for c in [*_APERTURE_COLUMNS, "extra"]
            }
        ).write(path)
        out = LoadApertures()(
            table=None, focal_plane=None, description={"file_name": str(path)}
        )
        assert set(out.colnames) == set(_APERTURE_COLUMNS)

    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.ecsv"
        QTable({"spectral_center": [1.0]}).write(path)
        with pytest.raises(ValueError, match="Missing required columns"):
            LoadApertures()(
                table=None, focal_plane=None, description={"file_name": str(path)}
            )
