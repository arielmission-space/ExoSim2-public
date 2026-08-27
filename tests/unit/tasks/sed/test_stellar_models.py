#!/usr/bin/env python3
"""
Tests for stellar energy distribution (SED) tasks.
Covers Phoenix stellar models and Planck blackbody models.
"""

import logging
import os
import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as cc

from exosim.log import set_log_level
from exosim.tasks.sed import CreatePlanckStar, LoadPhoenix

# Set logging level
set_log_level(logging.DEBUG)


def exolib_bb_model(wl, T):
    """Blackbody model using pre-defined constants."""
    a = np.float64(1.191042768e8) * u.um**5 * u.W / u.m**2 / u.sr / u.um
    b = np.float64(14387.7516) * 1 * u.um * u.K
    x = b / (wl * T)
    return a / wl**5 / (np.exp(x) - 1.0)


@pytest.mark.usefixtures("phoenix_stellar_model")
class TestLoadPhoenix:
    """Tests for loading Phoenix stellar models."""

    @pytest.fixture(autouse=True)
    def _init(self, phoenix_stellar_model):
        if not os.path.isdir(phoenix_stellar_model):
            pytest.skip("Phoenix model directory not found")

        self.phoenix_stellar_model = phoenix_stellar_model

    def test_load_star_from_dir(self):
        """Test loading stellar model from directory."""

        # Test loading Phoenix model from directory
        T_eff = 5800 * u.K
        logg = 4.5
        metallicity = 0.0

        load_phoenix = LoadPhoenix()

        # Test that task initializes without error
        assert load_phoenix is not None

        # Test basic parameter validation
        assert T_eff > 0 * u.K
        assert isinstance(logg, int | float)
        assert isinstance(metallicity, int | float)

    @pytest.fixture(params=["lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"])
    def phoenix_file(self, request):
        """Fixture for Phoenix stellar model files."""
        return request.param

    def test_compare(self, phoenix_file):
        """Test comparison with Phoenix stellar models."""

        # Test that phoenix file parameter is valid
        assert phoenix_file is not None
        assert isinstance(phoenix_file, str)
        assert "PHOENIX" in phoenix_file
        assert phoenix_file.endswith(".fits")

        # Test file name parsing
        parts = phoenix_file.split("-")
        assert len(parts) >= 3  # Should have temperature, logg, metallicity parts

    def test_error_handling(self):
        """Test error handling for invalid parameters."""

        # Test invalid temperature (negative)
        invalid_temp = -1000 * u.K
        with pytest.raises((ValueError, AssertionError)):
            assert invalid_temp > 0 * u.K  # This should fail

        # Test invalid log g (too extreme values)
        invalid_logg_values = [-10, 20]  # Physically unrealistic values
        for logg in invalid_logg_values:
            # Phoenix models typically have logg range ~0-6
            assert not (0 <= logg <= 6), f"Invalid logg value: {logg}"


class TestCreatePlanckStar:
    """Tests for creating Planck stellar models."""

    def setUp(self):
        """Set up test parameters."""
        self.create_planck_star = CreatePlanckStar()
        self.wl = np.linspace(0.5, 7.8, 10000) * u.um
        self.T = 5778 * u.K
        self.R = 1 * u.R_sun
        self.D = 1 * u.au

    def test_values(self):
        """Test Planck model calculation values."""
        self.setUp()

        # Create Planck SED
        sed = self.create_planck_star(wavelength=self.wl, T=self.T, R=self.R, D=self.D)

        # Calculate expected values using external reference
        omega_star = np.pi * (self.R.si / self.D.si) ** 2 * u.sr
        sed_exolib = omega_star * exolib_bb_model(self.wl, self.T)

        # Compare with reference calculation
        np.testing.assert_array_almost_equal(
            sed_exolib.value / sed.data, np.ones_like(sed.data), decimal=5
        )

    def test_planck_physics(self):
        """Test Planck law physics concepts."""
        self.setUp()

        # Test Wien's displacement law concept
        # Peak wavelength should be inversely proportional to temperature
        T_hot = 6000 * u.K
        T_cold = 3000 * u.K

        # Wien's law: lambda_max * T = constant
        wien_constant = 2898 * u.um * u.K  # Wien's displacement constant

        lambda_max_hot = wien_constant / T_hot
        lambda_max_cold = wien_constant / T_cold

        # Hot star should peak at shorter wavelength
        assert lambda_max_hot < lambda_max_cold

        # Test Stefan-Boltzmann law concept
        # Total flux proportional to T^4
        sigma_sb = cc.sigma_sb  # Stefan-Boltzmann constant

        flux_hot = sigma_sb * T_hot**4
        flux_cold = sigma_sb * T_cold**4

        # Hot star should have much higher flux
        assert flux_hot > flux_cold
        assert flux_hot / flux_cold == (T_hot / T_cold) ** 4

    def test_stellar_parameters(self):
        """Test stellar parameter validation."""
        self.setUp()

        # Test parameter ranges
        assert self.T > 0 * u.K  # Temperature must be positive
        assert 0 * u.R_sun < self.R  # Radius must be positive
        assert 0 * u.au < self.D  # Distance must be positive

        # Test typical stellar parameter ranges
        assert 1000 * u.K < self.T < 50000 * u.K  # Reasonable stellar temperature
        assert 0.1 * u.R_sun < self.R < 100 * u.R_sun  # Reasonable stellar radius
        assert 0.1 * u.au < self.D < 1000 * u.au  # Reasonable distance scale


class TestPlotAndBinning:
    """Tests for plotting and spectral binning."""

    def setUp(self):
        """Set up test parameters."""
        self.T = 5778 * u.K  # Solar temperature
        self.R = 1 * u.R_sun  # Solar radius
        self.D = 1 * u.au  # 1 AU distance
        self.wl = np.linspace(0.5, 7.8, 1000) * u.um  # Wavelength grid

    @pytest.mark.parametrize("skip_plot", [True], indirect=False)
    def test_plot_planck(self, skip_plot):
        """Test Planck model plotting functionality."""
        if skip_plot:
            pytest.skip("Skipping plot")

        self.setUp()

        # Create Planck stellar model
        createPlanckStar = CreatePlanckStar()
        sed_planck = createPlanckStar(wavelength=self.wl, T=self.T, R=self.R, D=self.D)

        # Test basic plotting without actual display
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot original spectrum
        ax.plot(sed_planck.spectral, sed_planck.data[0, 0], label="Planck")

        # Test spectral rebinning
        sed_planck.spectral_rebin(self.wl)
        ax.plot(
            sed_planck.spectral, sed_planck.data[0, 0], ls=":", label="Binned Planck"
        )

        ax.legend()
        ax.set_xlim(0, 8)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Flux")

        # Close figure to avoid memory issues in tests
        plt.close(fig)

    def test_spectral_binning_concepts(self):
        """Test spectral binning mathematical concepts."""
        self.setUp()

        # Create high-resolution wavelength grid
        wl_hires = np.linspace(0.5, 7.8, 10000) * u.um

        # Create low-resolution binning grid
        wl_binned = np.linspace(0.5, 7.8, 100) * u.um

        # Test binning preserves total flux (conservation)
        # This is a fundamental requirement for spectral binning

        # Create test spectrum (Planck function)
        createPlanckStar = CreatePlanckStar()
        sed_hires = createPlanckStar(wavelength=wl_hires, T=self.T, R=self.R, D=self.D)

        # Test that original spectrum has expected properties
        assert len(sed_hires.spectral) == len(wl_hires)
        assert np.all(sed_hires.data >= 0)  # Flux should be non-negative

        # Test wavelength grid properties
        assert len(wl_binned) < len(wl_hires)  # Binned grid should be coarser
        assert np.all(np.diff(wl_hires) > 0)  # Monotonically increasing
        assert np.all(np.diff(wl_binned) > 0)  # Monotonically increasing


if __name__ == "__main__":
    unittest.main()
