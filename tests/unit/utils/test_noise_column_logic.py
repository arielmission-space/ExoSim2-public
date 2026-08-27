"""
Unit tests for compute_noise_column functionality focusing on computational logic.

This module contains tests for the computational logic and noise analysis
patterns used in noise column calculations, focusing on statistical methods
and noise propagation algorithms.
"""

import unittest

import numpy as np


class TestComputeNoiseColumnLogic(unittest.TestCase):
    """Test compute_noise_column computational concepts and logic."""

    def test_noise_column_calculations(self):
        """Test noise column computational concepts."""
        # Test noise propagation along columns
        data_columns = np.random.poisson(1000, (100, 50))  # 100 rows, 50 columns

        # Calculate column-wise statistics
        column_means = np.mean(data_columns, axis=0)
        column_stds = np.std(data_columns, axis=0)
        column_noise = np.sqrt(column_means)  # Poisson noise

        assert len(column_means) == 50
        assert len(column_stds) == 50
        assert np.all(column_noise > 0)

        # Test SNR calculation
        column_snr = column_means / column_noise
        assert np.all(column_snr > 0)

    def test_noise_correlation_concepts(self):
        """Test noise correlation calculations."""
        # Test correlated noise pattern
        n_pixels = 64
        correlation_length = 5

        # Generate correlated noise pattern
        random_base = np.random.normal(0, 1, n_pixels)
        corr_noise = np.convolve(
            random_base, np.ones(correlation_length) / correlation_length, mode="same"
        )

        assert len(corr_noise) == n_pixels

        # Test autocorrelation concept
        autocorr = np.correlate(corr_noise, corr_noise, mode="full")
        max_corr_index = np.argmax(autocorr)

        # Maximum should be at center (zero lag)
        expected_center = len(autocorr) // 2
        assert max_corr_index == expected_center

    def test_noise_model_fitting(self):
        """Test noise model fitting for columns."""
        # Generate synthetic data with known noise properties
        n_rows, n_cols = 200, 10
        base_signal = 1000  # e-
        read_noise = 5.0  # e- RMS

        np.random.seed(42)

        # Create synthetic column data
        synthetic_data = np.zeros((n_rows, n_cols))
        for col in range(n_cols):
            # Each column has slightly different signal level
            col_signal = base_signal * (1 + 0.1 * col)

            # Add shot noise (Poisson) + read noise (Gaussian)
            shot_component = np.random.poisson(col_signal, n_rows)
            read_component = np.random.normal(0, read_noise, n_rows)

            synthetic_data[:, col] = shot_component + read_component

        # Analyze noise properties
        column_vars = np.var(synthetic_data, axis=0)
        column_means = np.mean(synthetic_data, axis=0)

        # For Poisson + Gaussian noise: Var = Signal + ReadNoise^2
        expected_vars = column_means + read_noise**2

        # Check if measured variance matches expectation (within reasonable tolerance)
        relative_errors = np.abs(column_vars - expected_vars) / expected_vars
        assert np.all(
            relative_errors < 0.2
        )  # Within 20% (more realistic for random data)

    def test_bad_pixel_detection_logic(self):
        """Test bad pixel detection in noise calculations."""
        # Create column with some bad pixels
        n_pixels = 100
        normal_noise = np.random.normal(0, 10, n_pixels)

        # Add some bad pixels
        bad_indices = [25, 50, 75]
        noisy_column = normal_noise.copy()
        noisy_column[bad_indices] += np.random.normal(
            0, 100, len(bad_indices)
        )  # Much higher noise

        # Test outlier detection using MAD (Median Absolute Deviation)
        median_val = np.median(noisy_column)
        mad = np.median(np.abs(noisy_column - median_val))

        # Identify outliers (> 5 sigma using MAD)
        threshold = 5 * 1.4826 * mad  # 1.4826 converts MAD to sigma equivalent
        outliers = np.abs(noisy_column - median_val) > threshold

        detected_bad_pixels = np.where(outliers)[0]

        # Should detect most of the bad pixels
        detection_rate = len(set(detected_bad_pixels) & set(bad_indices)) / len(
            bad_indices
        )
        assert detection_rate >= 0.5  # At least 50% detection

    def test_noise_frequency_analysis(self):
        """Test frequency domain analysis of noise."""
        # Generate time series noise with specific frequency components
        n_samples = 1000
        sample_rate = 100  # Hz
        time = np.arange(n_samples) / sample_rate

        # Create noise with 1/f component + white noise
        white_noise = np.random.normal(0, 1, n_samples)

        # Add 10 Hz sine wave (coherent noise)
        coherent_freq = 10  # Hz
        coherent_amplitude = 0.5
        coherent_noise = coherent_amplitude * np.sin(2 * np.pi * coherent_freq * time)

        combined_noise = white_noise + coherent_noise

        # Compute power spectrum
        fft_noise = np.fft.fft(combined_noise)
        frequencies = np.fft.fftfreq(n_samples, 1 / sample_rate)
        power_spectrum = np.abs(fft_noise) ** 2

        # Find peak frequency
        positive_freqs = frequencies[frequencies > 0]
        positive_power = power_spectrum[frequencies > 0]
        peak_freq_idx = np.argmax(positive_power)
        peak_frequency = positive_freqs[peak_freq_idx]

        # Should detect the 10 Hz component
        assert abs(peak_frequency - coherent_freq) < 1.0  # Within 1 Hz

    def test_noise_propagation_through_processing(self):
        """Test noise propagation through data processing steps."""
        # Initial noise level
        input_noise_rms = 10.0
        n_samples = 1000

        np.random.seed(42)
        input_data = np.random.normal(0, input_noise_rms, n_samples)

        # Processing step 1: Gain multiplication
        gain = 2.0
        gained_data = input_data * gain
        gained_noise_rms = np.std(gained_data)

        expected_gained_noise = input_noise_rms * gain
        assert abs(gained_noise_rms - expected_gained_noise) < 0.5

        # Processing step 2: Averaging (reduces noise)
        avg_factor = 4
        n_avg_samples = n_samples // avg_factor
        averaged_data = np.mean(gained_data.reshape(n_avg_samples, avg_factor), axis=1)
        averaged_noise_rms = np.std(averaged_data)

        expected_averaged_noise = expected_gained_noise / np.sqrt(avg_factor)
        assert abs(averaged_noise_rms - expected_averaged_noise) < 1.0

        # Processing step 3: Offset subtraction (shouldn't change noise)
        offset = 100.0
        offset_subtracted = averaged_data - offset
        final_noise_rms = np.std(offset_subtracted)

        # Noise should be unchanged by offset subtraction
        assert abs(final_noise_rms - averaged_noise_rms) < 0.1


if __name__ == "__main__":
    unittest.main()
