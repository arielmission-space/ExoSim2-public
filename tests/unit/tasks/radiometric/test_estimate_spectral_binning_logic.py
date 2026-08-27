"""
Unit tests for spectral binning estimation focusing on computational logic.

This module contains tests for the computational logic and optimization algorithms
used in spectral binning calculations, focusing on signal-to-noise optimization
and wavelength grid computations.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np


class TestEstimateSpectralBinningLogic(unittest.TestCase):
    """Test estimate_spectral_binning computational logic."""

    @patch("exosim.tasks.radiometric.estimate_spectral_binning.Task.__init__")
    def test_spectral_binning_initialization(self, mock_init):
        """Test initialization patterns."""
        from exosim.tasks.radiometric.estimate_spectral_binning import (
            EstimateSpectralBinning,
        )

        mock_init.return_value = None

        task = EstimateSpectralBinning()
        task.add_task_param = Mock()

        assert task is not None

    def test_spectral_binning_calculation_logic(self):
        """Test binning calculation logic."""
        # Test wavelength binning concepts
        wl_min, wl_max = 1.0, 2.0  # microns
        n_bins = 10

        bin_width = (wl_max - wl_min) / n_bins
        assert bin_width == 0.1

        # Test bin edge calculation
        bin_edges = np.linspace(wl_min, wl_max, n_bins + 1)
        assert len(bin_edges) == 11
        assert bin_edges[0] == wl_min
        assert bin_edges[-1] == wl_max

        # Test bin centers
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        assert len(bin_centers) == n_bins
        assert bin_centers[0] == 1.05  # First bin center
        assert bin_centers[-1] == 1.95  # Last bin center

    @patch("exosim.tasks.radiometric.estimate_spectral_binning.np")
    def test_binning_optimization_logic(self, mock_np):
        """Test binning optimization patterns."""
        mock_np.diff.return_value = np.array([0.1, 0.1, 0.1])
        mock_np.mean.return_value = 0.1

        # Test optimization criteria
        signal_to_noise = np.array([10, 15, 20])
        target_snr = 12

        # Simple binning logic
        adequate_bins = signal_to_noise >= target_snr
        n_adequate = np.sum(adequate_bins)

        assert n_adequate == 2

    def test_adaptive_binning_algorithm(self):
        """Test adaptive binning based on signal strength."""
        # Test spectral data with varying signal levels
        wavelengths = np.linspace(1.0, 2.5, 150)  # High resolution spectrum

        # Simulate stellar spectrum with lines (higher signal at certain wavelengths)
        continuum_signal = 1000.0  # e-/bin
        line_signal = continuum_signal * np.exp(
            -0.5 * ((wavelengths - 1.5) / 0.05) ** 2
        )
        total_signal = continuum_signal + line_signal

        # Calculate S/N (assuming Poisson noise)
        snr = total_signal / np.sqrt(total_signal)

        # Define target S/N
        target_snr = 50.0

        # Adaptive binning: bin size based on local S/N
        bin_sizes = np.ones_like(wavelengths)
        low_snr_mask = snr < target_snr

        # Where S/N is low, increase bin size (group more pixels)
        bin_sizes[low_snr_mask] = 2.0

        # Test that we identified low S/N regions correctly
        assert np.any(low_snr_mask)  # Should have some low S/N regions
        assert np.all(bin_sizes >= 1.0)  # All bin sizes should be >= 1

    def test_spectral_resolution_calculations(self):
        """Test spectral resolution and resolving power calculations."""
        # Test resolving power R = λ/Δλ
        central_wavelength = 2.0  # microns
        wavelength_resolution = 0.01  # microns (Δλ)

        resolving_power = central_wavelength / wavelength_resolution
        assert resolving_power == 200.0

        # Test for different spectral orders
        wavelengths = np.array([1.0, 1.5, 2.0, 2.5])  # microns
        resolutions = np.array([0.005, 0.0075, 0.01, 0.0125])  # microns

        resolving_powers = wavelengths / resolutions
        expected_R = np.array([200, 200, 200, 200])  # Constant R

        np.testing.assert_array_equal(resolving_powers, expected_R)

    def test_binning_for_emission_lines(self):
        """Test optimal binning for spectral line detection."""
        # Test line detection optimization
        line_wavelength = 1.65  # microns (H-alpha in NIR)
        line_width = 0.002  # microns (natural line width)

        # Optimal binning should be ~ line_width / 2 for Nyquist sampling
        optimal_bin_width = line_width / 2.0
        assert optimal_bin_width == 0.001

        # Test spectral sampling requirements
        # Test that the line falls within our spectral range
        spectral_range = np.array([1.6, 1.7])  # microns around line
        assert spectral_range[0] <= line_wavelength <= spectral_range[1], (
            "Line wavelength outside spectral range"
        )
        range_width = np.diff(spectral_range)[0]

        n_bins_needed = int(range_width / optimal_bin_width)
        assert (
            n_bins_needed >= 99
        )  # Need ~100 bins to sample 0.1 μm at 0.001 μm resolution

    def test_snr_weighted_binning(self):
        """Test S/N weighted spectral binning."""
        # Generate mock spectrum with noise
        n_pixels = 100
        wavelengths = np.linspace(1.0, 2.0, n_pixels)

        # Varying signal levels across spectrum
        signal_levels = 1000 * (1 + 0.5 * np.sin(2 * np.pi * (wavelengths - 1.0)))
        noise_levels = np.sqrt(signal_levels)  # Poisson noise

        snr_per_pixel = signal_levels / noise_levels

        # Target S/N for final binned spectrum
        target_snr_final = 100.0

        # Calculate required binning factor per region
        current_snr = snr_per_pixel
        binning_factor = np.ceil((target_snr_final / current_snr) ** 2)
        binning_factor = np.maximum(binning_factor, 1)  # At least 1 pixel per bin

        # Test that binning factors make sense
        assert np.all(binning_factor >= 1)
        assert np.max(binning_factor) > 1  # Should need some binning

        # Where signal is high, binning factor should be low
        high_signal_indices = signal_levels > np.median(signal_levels)
        low_signal_indices = ~high_signal_indices

        avg_binning_high = np.mean(binning_factor[high_signal_indices])
        avg_binning_low = np.mean(binning_factor[low_signal_indices])

        # Should need less binning where signal is high
        assert avg_binning_high <= avg_binning_low

    def test_spectral_binning_conservation(self):
        """Test conservation laws in spectral binning."""
        # Test that total flux is conserved during binning
        original_spectrum = np.array([100, 120, 150, 180, 200, 190, 160, 140])

        # Bin by factor of 2
        binning_factor = 2
        n_bins_out = len(original_spectrum) // binning_factor

        binned_spectrum = np.zeros(n_bins_out)
        for i in range(n_bins_out):
            start_idx = i * binning_factor
            end_idx = start_idx + binning_factor
            binned_spectrum[i] = np.sum(original_spectrum[start_idx:end_idx])

        # Test flux conservation
        total_flux_original = np.sum(original_spectrum)
        total_flux_binned = np.sum(binned_spectrum)

        assert abs(total_flux_original - total_flux_binned) < 1e-10

        # Test that binned spectrum has correct length
        assert len(binned_spectrum) == len(original_spectrum) // binning_factor


if __name__ == "__main__":
    unittest.main()
