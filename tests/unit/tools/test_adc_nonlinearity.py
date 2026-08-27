"""
Unit tests for ADC and pixel non-linearity calibration tools.

This module contains tests for:
- ADC gain estimation
- Pixel non-linearity characterization
- Non-linearity correction from calibration data
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.tools import (
    ADCGainEstimator,
    PixelsNonLinearity,
    PixelsNonLinearityFromCorrection,
)


class TestADCGain:
    """Test suite for ADC gain estimation tool."""

    def test_value(self):
        """Test ADC gain calculation with different bit depths."""
        params = {
            "channel": {
                "value": "test",
                "detector": {"ADC_num_bit": 16, "ADC_max_value": 120000},
            }
        }
        adc = ADCGainEstimator(params)
        res = adc.results["test"]
        np.testing.assert_allclose(res["gain factor"], 0.546125)
        np.testing.assert_equal(res["max adc value"], 65535)
        np.testing.assert_equal(res["integer dtype"], np.dtype("int16"))

        # Test with 8-bit ADC
        params = {
            "channel": {
                "value": "test",
                "detector": {"ADC_num_bit": 8, "ADC_max_value": 120000},
            }
        }
        adc = ADCGainEstimator(params)
        res = adc.results["test"]
        np.testing.assert_equal(res["integer dtype"], np.dtype("int8"))

    def test_missing_info(self):
        """Test ADC gain estimation with missing configuration."""
        params = {"channel": {"value": "test", "detector": {"well_depth": 120000}}}
        adc = ADCGainEstimator(params)
        res = adc.results["test"]
        np.testing.assert_equal(res["integer dtype"], np.dtype("int32"))

    def test_errors(self):
        """Test error handling for invalid ADC parameters."""
        # Test with invalid bit depth type
        params = {
            "channel": {
                "value": "test",
                "detector": {"ADC_num_bit": "16", "ADC_max_value": 120000},
            }
        }
        with pytest.raises(TypeError):
            ADCGainEstimator(params)

        # Test with unsupported bit depth
        params = {
            "channel": {
                "value": "test",
                "detector": {"ADC_num_bit": 64, "ADC_max_value": 120000},
            }
        }
        with pytest.raises(ValueError, match=r".*"):
            ADCGainEstimator(params)


class TestPixelNonLinearity:
    """Test suite for pixel non-linearity characterization."""

    def __init__(self):
        """Initialize test parameters for pixel non-linearity tests."""
        self.params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 200,
                    "spectral_pix": 200,
                    "well_depth": 25000 * u.ct,
                    "pnl_coeff_std": 0.05,
                },
            }
        }

    def test_values(self, skip_plot):
        """Test pixel non-linearity coefficient generation."""
        test_out = PixelsNonLinearity(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["coeff"]
        assert results is not None

    def test_map(self, skip_plot):
        """Test pixel non-linearity map generation and statistical properties."""
        test_out = PixelsNonLinearity(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["map"]
        coeff = test_out.results["test"]["coeff"]
        expected_std = self.params["channel"]["detector"]["pnl_coeff_std"]

        # Test statistical properties of generated maps
        for i, map_ in enumerate(results):
            mean = np.mean(map_)
            std = np.std(map_) / np.abs(mean)
            np.testing.assert_allclose(mean, coeff[i], rtol=5 * 1e-03)
            np.testing.assert_allclose(std, expected_std, rtol=5 * 1e-02)

        # Test with zero standard deviation (constant maps)
        self.params["channel"]["detector"].pop("pnl_coeff_std")
        test_out = PixelsNonLinearity(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["map"]
        coeff = test_out.results["test"]["coeff"]

        for i, map_ in enumerate(results):
            np.testing.assert_allclose(map_, np.ones(map_.shape) * coeff[i])


class TestPixelNonLinearityFromCorrection:
    """Test suite for pixel non-linearity from correction data."""

    def __init__(self):
        """Initialize test parameters for correction-based non-linearity tests."""
        self.params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 10,
                    "spectral_pix": 20,
                    "well_depth": 25000 * u.ct,
                    "pnl_coeff_a": 1.00117667e00,
                    "pnl_coeff_b": -5.41836850e-07,
                    "pnl_coeff_c": 4.57790820e-11,
                    "pnl_coeff_d": 7.66734616e-16,
                    "pnl_coeff_e": -2.32026578e-19,
                    "pnl_coeff_std": 0.005,
                    "pnl_correction_operator": "/",
                },
            }
        }

    def test_values(self, skip_plot):
        """Test coefficient computation from correction parameters."""
        test_out = PixelsNonLinearityFromCorrection(
            self.params, show_results=not skip_plot
        )
        results = test_out.results["test"]["coeff"]
        np.testing.assert_allclose(
            results,
            [
                1,
                -6.02340621e-07,
                1.42989267e-10,
                -9.63414109e-15,
                6.97856298e-20,
            ],
            rtol=1e-06,
        )

    def test_correction(self, skip_plot):
        """Test that correction properly inverts the non-linearity."""
        test_out = PixelsNonLinearityFromCorrection(
            self.params, show_results=not skip_plot
        )
        results = test_out.results["test"]["coeff"]
        p = np.polynomial.Polynomial(results)

        # Generate test counts and apply non-linearity
        Q = np.linspace(
            1, test_out.results["test"]["saturation"], 2**10
        )  # detector pixel counts in ADU
        Q_det = Q * p(Q)

        # Apply correction and verify we recover original values
        corr_coeff = [
            1.00117667e00,
            -5.41836850e-07,
            4.57790820e-11,
            7.66734616e-16,
            -2.32026578e-19,
        ]
        p_corr = np.polynomial.Polynomial(corr_coeff)
        Q_corr = Q_det / p_corr(Q)

        np.testing.assert_allclose(Q_corr, Q, rtol=5 * 1e-02)

    def test_missing_keys(self, skip_plot):
        """Test error handling for missing correction coefficients."""
        params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 10,
                    "spectral_pix": 20,
                    "well_depth": 25000 * u.ct,
                },
            }
        }
        with pytest.raises(KeyError):
            PixelsNonLinearityFromCorrection(params, show_results=not skip_plot)
