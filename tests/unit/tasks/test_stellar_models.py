"""
Unit tests for stellar model loading and SED generation.

This module contains tests for loading stellar models from various sources,
including Phoenix stellar models, custom SED files, and Planck blackbody
models. It also includes tests for spectral plotting and binning functionality.
"""

import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as cc
from astropy.io import ascii

from exosim.log import set_log_level
from exosim.tasks.sed import (
    CreatePlanckStar,
    LoadCustom,
    LoadPhoenix,
)

# Set logging level
set_log_level(logging.DEBUG)


def exolib_bb_model(wl, T):
    """
    Blackbody model using pre-defined constants.

    This function implements a reference blackbody model for comparison
    with stellar SED models, using consistent physical constants.

    Parameters
    ----------
    wl : astropy.units.Quantity
        Wavelength array
    T : astropy.units.Quantity
        Blackbody temperature

    Returns
    -------
    astropy.units.Quantity
        Blackbody spectral radiance
    """
    a = np.float64(1.191042768e8) * u.um**5 * u.W / u.m**2 / u.sr / u.um
    b = np.float64(14387.7516) * 1 * u.um * u.K
    x = b / (wl * T)
    return a / wl**5 / (np.exp(x) - 1.0)


class TestLoadPhoenix:
    """Test suite for loading Phoenix stellar models."""

    @pytest.fixture(autouse=True)
    def _init(self, phoenix_stellar_model):
        """
        Initialize Phoenix model loading tests.

        This fixture sets up the LoadPhoenix task instance and validates
        that Phoenix model data is available for testing.

        Parameters
        ----------
        phoenix_stellar_model : str or None
            Path to Phoenix stellar model directory from fixture
        """
        if phoenix_stellar_model is None:
            pytest.skip("Phoenix model directory not available")

        self.loadPhoenix = LoadPhoenix()
        self.phoenix_stellar_model = phoenix_stellar_model

    def test_load_star_from_directory(self):
        """
        Test loading Phoenix stellar model from directory.

        This test verifies that Phoenix stellar models can be loaded from
        a model directory using stellar parameters (temperature, mass, radius,
        distance, metallicity, and surface gravity).
        """
        # Stellar parameters for GJ 1214 (M dwarf)
        D = 12.975 * u.pc  # Distance
        T = 3016 * u.K  # Effective temperature
        M = 0.15 * u.Msun  # Mass
        R = 0.218 * u.Rsun  # Radius
        z = 0.0  # Metallicity

        # Calculate surface gravity
        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)

        sed = self.loadPhoenix(
            path=self.phoenix_stellar_model, T=T, D=D, R=R, z=z, logg=logg
        )
        assert sed is not None, "Phoenix SED should be loaded successfully"

    def test_compare_directory_vs_file_loading(self, phoenix_file):
        """
        Test consistency between directory and file loading methods.

        This test verifies that loading a Phoenix model from a directory
        produces the same result as loading from a specific file path,
        ensuring consistency across different loading methods.
        """
        if phoenix_file is None:
            pytest.skip("Phoenix file not available")

        # Stellar parameters
        D = 12.975 * u.pc
        T = 3016 * u.K
        M = 0.15 * u.Msun
        R = 0.218 * u.Rsun
        z = 0.0

        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)

        # Load from directory
        sed_dir = self.loadPhoenix(
            path=self.phoenix_stellar_model, T=T, D=D, R=R, z=z, logg=logg
        )

        # Load from specific file
        sed_file = self.loadPhoenix(filename=phoenix_file, D=D, R=R)

        # Compare results
        np.testing.assert_array_equal(
            sed_dir.data,
            sed_file.data,
            "SED data should be identical between directory and file loading",
        )

    def test_error_handling(self):
        """
        Test error handling for invalid Phoenix model paths.

        This test verifies that appropriate errors are raised when
        attempting to load Phoenix models from invalid paths.
        """
        with pytest.raises(IOError, match=r".*"):
            self.loadPhoenix(
                path="invalid_path", T=3000 * u.K, D=1 * u.pc, R=1 * u.Rsun
            )


class TestLoadCustomSED:
    """Test suite for loading custom SED files."""

    def test_custom_sed_loading(self, project_root):
        """
        Test loading custom SED files.

        This test verifies that custom SED files in ECSV format can be
        loaded correctly, with proper unit handling and scaling based on
        stellar parameters.
        """
        # Look for custom SED file in common locations
        custom_paths = [
            os.path.join(project_root, "examples", "customsed.csv"),
            os.path.join(project_root, "test_data", "customsed.csv"),
            os.path.join(project_root, "tests", "test_data", "customsed.csv"),
            os.path.join(project_root, "data", "customsed.csv"),
        ]

        custom_file = None
        for path in custom_paths:
            if os.path.isfile(path):
                custom_file = path
                break
        loadCustom = LoadCustom()

        if custom_file is None:
            pytest.skip("Custom SED file not found in any expected location")

        # Stellar parameters
        D = 1 * u.au  # Distance
        R = 1 * u.Rsun  # Radius

        sed = loadCustom(filename=custom_file, D=D, R=R)

        # Load reference data for comparison
        ph = ascii.read(custom_file, format="ecsv")
        ph_sed = ph["Sed"].data * ph["Sed"].unit
        ph_sed *= np.pi * (R.to(u.m) / D.to(u.m)) ** 2 * u.sr

        # Validate units and data
        np.testing.assert_equal(
            sed.data_units, ph_sed.unit, "SED units should match reference"
        )
        np.testing.assert_array_equal(
            sed.data[0, 0], ph_sed.value, "SED values should match reference data"
        )


class TestPlanckStellarModels:
    """Test suite for Planck blackbody stellar models."""

    wl = np.linspace(0.5, 7.8, 10000) * u.um
    T = 5778 * u.K  # Solar temperature
    R = 1 * u.R_sun  # Solar radius
    D = 1 * u.au  # 1 AU distance

    def test_planck_model_accuracy(self):
        """
        Test accuracy of Planck stellar model against reference blackbody.

        This test verifies that the CreatePlanckStar task produces results
        consistent with a reference blackbody model, accounting for solid
        angle scaling based on stellar parameters.
        """
        createPlanckStar = CreatePlanckStar()
        sed = createPlanckStar(wavelength=self.wl, T=self.T, R=self.R, D=self.D)

        # Calculate expected solid angle
        omega_star = np.pi * (self.R.si / self.D.si) ** 2 * u.sr

        # Reference blackbody model
        sed_exolib = omega_star * exolib_bb_model(self.wl, self.T)

        # Compare with high precision
        np.testing.assert_array_almost_equal(
            sed_exolib.value / sed.data,
            np.ones_like(sed.data),
            decimal=5,
            err_msg="Planck model should match reference blackbody within 5 decimal places",
        )


class TestSpectralPlottingAndBinning:
    """Test suite for plotting and spectral binning functionality."""

    def setup_method(self, method):
        """
        Set up test parameters for plotting and binning tests.

        This method initializes stellar parameters and wavelength arrays
        for testing spectral plotting and binning functionality.
        """
        # Skip plotting tests if requested
        if hasattr(self, "skip_plot") and self.skip_plot:
            pytest.skip("Skipping plot tests")

        self.T = 5778 * u.K  # Solar temperature
        self.R = 1 * u.R_sun  # Solar radius
        self.D = 1 * u.au  # 1 AU distance
        self.wl = np.linspace(0.5, 7.8, 1000) * u.um

    @pytest.mark.skip(reason="Plotting test - enable manually if needed")
    def test_planck_plotting_and_binning(self):
        """
        Test Planck model plotting and spectral rebinning.

        This test verifies that Planck stellar models can be plotted
        and spectrally rebinned to different wavelength grids while
        maintaining data integrity.

        Note: This test is skipped by default to avoid GUI dependencies
        in automated testing environments.
        """
        createPlanckStar = CreatePlanckStar()
        sed_planck = createPlanckStar(wavelength=self.wl, T=self.T, R=self.R, D=self.D)

        # Create plot of original spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(sed_planck.spectral, sed_planck.data[0, 0], label="Original Planck")

        # Test spectral rebinning
        sed_planck.spectral_rebin(self.wl)
        plt.plot(
            sed_planck.spectral,
            sed_planck.data[0, 0],
            ls=":",
            linewidth=2,
            label="Binned Planck",
        )

        # Format plot
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Spectral Radiance")
        plt.title("Planck Stellar Model: Original vs Binned")
        plt.legend()
        plt.xlim(0, 8)
        plt.grid(True, alpha=0.3)
        plt.show()

        # Verify that rebinning completed without errors
        assert sed_planck.data is not None, "Rebinned SED data should not be None"
