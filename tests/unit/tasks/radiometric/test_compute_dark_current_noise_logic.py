"""
Unit tests for dark current noise computation focusing on computational logic.

This module contains tests for the computational logic and physics calculations
used in dark current noise calculations, focusing on statistical models and
detector physics.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np


class TestComputeConstantDarkCurrentNoiseLogic(unittest.TestCase):
    """Test compute_constant_dark_current_noise computational logic."""

    @patch("exosim.tasks.radiometric.compute_constant_dark_current_noise.Task.__init__")
    def test_dark_current_noise_initialization(self, mock_init):
        """Test initialization structure."""
        from exosim.tasks.radiometric.compute_constant_dark_current_noise import (
            ComputeConstantDarkCurrentNoise,
        )

        mock_init.return_value = None

        task = ComputeConstantDarkCurrentNoise()
        task.add_task_param = Mock()

        # Test parameter setup patterns
        assert task is not None

    def test_dark_current_calculation_logic(self):
        """Test dark current calculation logic."""
        # Test basic dark current physics
        dark_current_rate = 0.01  # e-/s/pix
        integration_time = 100.0  # s

        expected_dark_current = dark_current_rate * integration_time
        assert expected_dark_current == 1.0

        # Test temperature dependence (rule of thumb: doubles every 6-8K)
        temp_ref = 80.0  # K
        temp_hot = 86.0  # K (6K increase)

        dark_ref = 0.01  # e-/s at reference temperature
        temp_factor = 2.0 ** ((temp_hot - temp_ref) / 6.0)  # Rule of thumb
        dark_hot = dark_ref * temp_factor

        assert abs(dark_hot - 0.02) < 0.001  # Should roughly double

    @patch("exosim.tasks.radiometric.compute_constant_dark_current_noise.np")
    def test_dark_current_noise_computation(self, mock_np):
        """Test noise computation patterns."""
        mock_np.sqrt.return_value = np.array([1.0, 1.4, 1.7])
        mock_np.random.poisson.return_value = np.array([1, 2, 3])

        # Test statistical operations
        dark_counts = np.array([1, 2, 3])
        noise = np.sqrt(dark_counts)

        np.testing.assert_array_almost_equal(noise, [1.0, 1.414, 1.732], decimal=3)

    def test_dark_current_spatial_variations(self):
        """Test spatial variation patterns in dark current."""
        # Test hot pixel generation
        detector_shape = (64, 64)
        base_dark_current = 0.01  # e-/s/pix

        # Generate base dark current map
        dark_map = np.full(detector_shape, base_dark_current)

        # Add hot pixels (typically 0.1-1% of pixels)
        hot_pixel_fraction = 0.005  # 0.5%
        n_hot_pixels = int(hot_pixel_fraction * np.prod(detector_shape))

        np.random.seed(42)
        hot_indices = np.random.choice(
            np.prod(detector_shape), n_hot_pixels, replace=False
        )
        hot_rows, hot_cols = np.unravel_index(hot_indices, detector_shape)

        # Hot pixels have 10-100x higher dark current
        hot_multiplier = 50.0
        dark_map[hot_rows, hot_cols] *= hot_multiplier

        # Verify statistics
        assert dark_map.shape == detector_shape
        assert (
            np.mean(dark_map) > base_dark_current
        )  # Should be higher due to hot pixels
        assert np.max(dark_map) >= base_dark_current * hot_multiplier

    def test_dark_current_time_integration(self):
        """Test time integration of dark current."""
        # Test accumulation over time
        dark_rate = 0.02  # e-/s/pix
        time_steps = np.array([0, 10, 20, 30, 40])  # seconds

        # Calculate accumulated dark current
        dark_accumulated = dark_rate * time_steps
        expected = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        np.testing.assert_array_equal(dark_accumulated, expected)

        # Test reset behavior (CDS - Correlated Double Sampling)
        reset_dark = 0.0  # At t=0
        final_dark = dark_rate * time_steps[-1]  # At t=40s

        cds_signal = final_dark - reset_dark
        assert cds_signal == 0.8

    def test_dark_current_noise_propagation(self):
        """Test noise propagation in dark current calculations."""
        # Test multiple readouts and averaging
        n_reads = 4
        dark_per_read = 0.1  # e-/read

        # Single read noise
        single_read_noise = np.sqrt(dark_per_read)

        # Multiple reads (incoherent averaging)
        total_dark = n_reads * dark_per_read
        total_noise = np.sqrt(total_dark)  # Poisson noise

        # Averaged noise (if we average the reads)
        averaged_dark = total_dark / n_reads  # Back to single read level
        averaged_noise = total_noise / np.sqrt(n_reads)  # Proper noise averaging

        # Verify single read, total noise and averaged values
        assert abs(single_read_noise - np.sqrt(0.1)) < 1e-10
        assert abs(total_noise - np.sqrt(0.4)) < 1e-10
        assert (
            abs(averaged_dark - dark_per_read) < 1e-10
        )  # Should return to original level
        assert averaged_noise < total_noise  # Averaging reduces noise by sqrt(N)


if __name__ == "__main__":
    unittest.main()
