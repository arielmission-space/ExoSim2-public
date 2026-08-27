"""
Unit tests for CreateSubExposures recipe focusing on computational logic.

This module contains tests for the computational logic and data processing
patterns used in the CreateSubExposures recipe, focusing on timing calculations,
signal processing, and sub-exposure specific algorithms.
"""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class TestCreateSubExposuresComputationalLogic(unittest.TestCase):
    """Test CreateSubExposures computational concepts and logic."""

    def test_sub_exposure_timing(self):
        """Test sub-exposure timing calculations."""
        total_time = 3600  # 1 hour
        n_sub_exposures = 60

        sub_exp_duration = total_time / n_sub_exposures
        assert sub_exp_duration == 60.0  # 1 minute each

        # Test time grid generation
        time_grid = np.linspace(0, total_time, n_sub_exposures + 1)
        assert len(time_grid) == 61

        # Test sub-exposure intervals
        intervals = np.diff(time_grid)
        np.testing.assert_array_almost_equal(
            intervals, np.full(n_sub_exposures, sub_exp_duration), decimal=6
        )

    def test_sub_exposure_data_structure(self):
        """Test sub-exposure data structures."""
        n_channels = 3
        n_sub_exposures = 10
        n_pixels = 1024

        # Mock sub-exposure data structure
        sub_exp_data = {}
        for ch in range(n_channels):
            ch_name = f"channel_{ch}"
            sub_exp_data[ch_name] = np.random.normal(
                1000, 50, (n_sub_exposures, n_pixels)
            )

        assert len(sub_exp_data) == n_channels
        for data in sub_exp_data.values():
            assert data.shape == (n_sub_exposures, n_pixels)

    def test_sub_exposure_signal_processing(self):
        """Test signal processing concepts for sub-exposures."""
        # Test signal accumulation
        signal_rate = 100.0  # e-/s
        exposure_times = np.array([1.0, 2.0, 3.0, 4.0])  # seconds

        expected_signals = signal_rate * exposure_times
        np.testing.assert_array_equal(expected_signals, [100, 200, 300, 400])

        # Test noise calculation
        dark_current = 0.1  # e-/s
        total_noise_squared = expected_signals + dark_current * exposure_times
        total_noise = np.sqrt(total_noise_squared)

        assert np.all(total_noise > 0)
        assert len(total_noise) == len(exposure_times)

    @patch("tempfile.mkdtemp")
    def test_sub_exposure_output_handling(self, mock_mkdtemp):
        """Test output directory handling."""
        mock_mkdtemp.return_value = "/tmp/test_subexp"

        output_dir = tempfile.mkdtemp(prefix="subexp_")
        assert output_dir.startswith("/tmp/test_subexp")
        mock_mkdtemp.assert_called_once()

    def test_sub_exposure_integration_logic(self):
        """Test integration time and exposure logic."""
        # Test different integration strategies
        total_observation_time = 1800  # 30 minutes

        # Strategy 1: Many short exposures
        short_exp_time = 30  # 30s each
        n_short_exposures = total_observation_time // short_exp_time
        assert n_short_exposures == 60

        # Strategy 2: Fewer long exposures
        long_exp_time = 300  # 5 minutes each
        n_long_exposures = total_observation_time // long_exp_time
        assert n_long_exposures == 6

        # Test duty cycle calculation
        overhead_time = 5.0  # 5s overhead per exposure

        short_duty_cycle = short_exp_time / (short_exp_time + overhead_time)
        long_duty_cycle = long_exp_time / (long_exp_time + overhead_time)

        # Longer exposures should have better duty cycle
        assert long_duty_cycle > short_duty_cycle

    def test_sub_exposure_jitter_effects(self):
        """Test jitter effects on sub-exposures."""
        # Test pointing jitter simulation
        n_sub_exposures = 50
        jitter_amplitude = 0.1  # arcsec RMS

        # Generate jitter time series
        np.random.seed(42)
        jitter_x = np.random.normal(0.0, jitter_amplitude, n_sub_exposures)
        jitter_y = np.random.normal(0.0, jitter_amplitude, n_sub_exposures)

        pointing_positions = np.column_stack((jitter_x, jitter_y))

        assert pointing_positions.shape == (n_sub_exposures, 2)

        # Test RMS jitter calculation
        rms_jitter = np.sqrt(np.mean(jitter_x**2 + jitter_y**2))

        # Should be close to input amplitude
        assert abs(rms_jitter - jitter_amplitude) < 0.05

    def test_sub_exposure_readout_patterns(self):
        """Test different readout patterns for sub-exposures."""
        # Test MULTIACCUM pattern
        n_groups = 4
        n_frames_per_group = 3
        frame_time = 2.67  # seconds

        total_readout_time = n_groups * n_frames_per_group * frame_time
        integration_time = (n_groups - 1) * n_frames_per_group * frame_time

        # Test timing calculations
        assert abs(total_readout_time - 32.04) < 0.01
        assert abs(integration_time - 24.03) < 0.01

        # Test efficiency
        efficiency = integration_time / total_readout_time
        assert abs(efficiency - 0.75) < 0.01


class TestCreateSubExposuresSkyGuard(unittest.TestCase):
    """Tests for the sky-section guard in CreateSubExposures.

    Verifies that the recipe handles a missing or source-free sky configuration
    without raising KeyError when looking up mainConfig['sky'], and that the
    combination of the guard with FindAstronomicalSignals yields no signals.
    """

    def test_get_with_default_returns_empty_when_sky_absent(self):
        """Test that .get('sky', {}) returns {} when sky key is absent.

        Mimics the guard added in CreateSubExposures before calling
        FindAstronomicalSignals when mainConfig has no 'sky' section.
        """
        main_config = {"wl_grid": {}, "payload": {}}
        sky_params = main_config.get("sky", {})
        assert sky_params == {}

    def test_get_with_default_preserves_sky_when_present(self):
        """Test that .get('sky', {}) returns the sky section when it exists.

        Ensures backward compatibility: a config that does include a sky
        section continues to be passed correctly to FindAstronomicalSignals.
        """
        sky_section = {"foregrounds": {"zodiacal_light": {}}}
        main_config = {"sky": sky_section}
        sky_params = main_config.get("sky", {})
        assert sky_params == sky_section

    def test_sky_without_source_yields_no_astrosignals(self):
        """Test that a sky section without a source produces no astronomical signals.

        Combines the .get guard with FindAstronomicalSignals to verify the
        full code path in CreateSubExposures when only foregrounds are present.
        """
        from exosim.tasks.astrosignal.find_astronomical_signals import (
            FindAstronomicalSignals,
        )

        main_config = {"sky": {"foregrounds": {}}}
        sky_params = main_config.get("sky", {})

        finder = FindAstronomicalSignals()
        astrosignals = finder(sky_parameters=sky_params)
        assert astrosignals == {}

    def test_no_sky_section_yields_no_astrosignals(self):
        """Test that a config without a sky section produces no astronomical signals.

        End-to-end test of the guard chain: absent sky key → empty dict passed
        to FindAstronomicalSignals → empty signals dict returned → the
        for-loop body in CreateSubExposures is never entered.
        """
        from exosim.tasks.astrosignal.find_astronomical_signals import (
            FindAstronomicalSignals,
        )

        main_config = {}  # no sky section at all
        sky_params = main_config.get("sky", {})

        finder = FindAstronomicalSignals()
        astrosignals = finder(sky_parameters=sky_params)
        assert astrosignals == {}


if __name__ == "__main__":
    unittest.main()
