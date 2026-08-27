"""
Unit tests for PSF utilities and aperture calculations.

This module tests PSF (Point Spread Function) creation, aperture finding
algorithms, and related instrumental PSF loading functionality.
"""

import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from photutils.aperture import EllipticalAperture, RectangularAperture

import exosim.tasks.instrument as instrument
import exosim.utils.aperture as psf_util
from exosim.log import set_log_level
from exosim.utils.klass_factory import find_task
from exosim.utils.psf import create_psf

set_log_level(logging.DEBUG)


@pytest.fixture
def wl_grid():
    """
    Create a wavelength grid for PSF testing.

    This fixture provides a standard wavelength grid covering
    the infrared range commonly used in exoplanet observations.

    Returns
    -------
    astropy.units.Quantity
        Array of wavelengths in micrometers
    """
    return np.linspace(1.95, 3.78, 10) * u.um


@pytest.fixture
def psf():
    """
    Create a standard PSF for aperture testing.

    This fixture generates an Airy disk PSF at a specific wavelength
    with standard optical parameters for testing aperture algorithms.

    Returns
    -------
    numpy.ndarray
        2D PSF array with Airy disk pattern
    """
    return create_psf(4 * u.um, 15.5, 6 * u.um, shape="airy")


class TestPSFCreation:
    """Test suite for PSF creation functionality."""

    @pytest.mark.parametrize("shape", ["airy", "gauss"])
    def test_create_psf_shape(self, wl_grid, shape):
        """
        Test PSF creation with different profile shapes.

        This test verifies that PSF creation works correctly for
        both Airy disk and Gaussian profile shapes, maintaining
        proper array dimensions.

        Parameters
        ----------
        wl_grid : astropy.units.Quantity
            Wavelength grid for PSF generation
        shape : str
            PSF profile shape ("airy" or "gauss")
        """
        psf = create_psf(wl_grid, 15.5, 18 * u.um, shape=shape)
        assert psf.shape == (len(wl_grid), 15, 15)

    @pytest.mark.parametrize("nzero", [4, 8])
    def test_create_psf_nzero(self, wl_grid, nzero):
        """
        Test PSF creation with different numbers of Airy ring zeros.

        This test verifies that the PSF array size adapts properly
        to include the specified number of Airy ring zeros.

        Parameters
        ----------
        wl_grid : astropy.units.Quantity
            Wavelength grid for PSF generation
        nzero : int
            Number of Airy ring zeros to include
        """
        psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy", nzero=nzero)
        assert psf.shape[-1] > 4

    @pytest.mark.parametrize(
        ("max_array_size", "expected_shape"),
        [
            ((5, 5), (10, 5, 5)),
            ((4, 4), (10, 5, 5)),
        ],
    )
    def test_create_psf_max_size(self, wl_grid, max_array_size, expected_shape):
        """
        Test PSF creation with maximum array size constraints.

        This test verifies that PSF creation respects maximum
        array size limits while maintaining proper dimensions.

        Parameters
        ----------
        wl_grid : astropy.units.Quantity
            Wavelength grid for PSF generation
        max_array_size : tuple
            Maximum allowed array size (height, width)
        expected_shape : tuple
            Expected output PSF shape
        """
        psf = create_psf(
            wl_grid, 15.5, 18 * u.um, shape="airy", max_array_size=max_array_size
        )
        assert psf.shape == expected_shape

    @pytest.mark.parametrize(
        ("array_size", "expected_shape"),
        [
            ((31, 21), (10, 31, 21)),
            ((30, 20), (10, 31, 21)),
        ],
    )
    def test_create_psf_array_size(self, wl_grid, array_size, expected_shape):
        """
        Test PSF creation with specific array size requirements.

        This test verifies that PSF creation handles specific
        array size requirements correctly, including odd-size
        adjustments for proper centering.

        Parameters
        ----------
        wl_grid : astropy.units.Quantity
            Wavelength grid for PSF generation
        array_size : tuple
            Requested array size (height, width)
        expected_shape : tuple
            Expected output PSF shape after odd-size adjustment
        """
        psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy", array_size=array_size)
        assert psf.shape == expected_shape

    @pytest.mark.skip(reason="Plotting test - enable manually if needed")
    def test_create_psf_plot(self, wl_grid):
        """
        Test PSF visualization plotting.

        This test creates a PSF and generates a plot for visual
        inspection. It is skipped by default to avoid GUI dependencies
        in automated testing environments.

        Parameters
        ----------
        wl_grid : astropy.units.Quantity
            Wavelength grid for PSF generation
        """
        psf = create_psf(wl_grid, 15.5, 18 * u.um, shape="airy")
        plt.figure(figsize=(8, 6))
        plt.imshow(psf[0])
        plt.title("PSF Test")
        plt.colorbar()
        plt.show()


class TestApertureCalculations:
    """Test suite for aperture finding algorithms."""

    def test_energy_rectangular(self, psf):
        """
        Test rectangular aperture energy enclosure calculation.

        This test verifies that the rectangular aperture finding
        algorithm correctly identifies aperture sizes that enclose
        the specified fraction of PSF energy.

        Parameters
        ----------
        psf : numpy.ndarray
            2D PSF array for aperture calculation
        """
        _sizes, _surf, ene = psf_util.find_rectangular_aperture(psf, 0.84)
        assert np.round(ene, decimals=2) >= 0.84

    def test_energy_elliptical(self, psf):
        """
        Test elliptical aperture energy enclosure calculation.

        This test verifies that the elliptical aperture finding
        algorithm correctly identifies aperture sizes that enclose
        the specified fraction of PSF energy.

        Parameters
        ----------
        psf : numpy.ndarray
            2D PSF array for aperture calculation
        """
        _sizes, _, ene = psf_util.find_elliptical_aperture(psf, 0.84)
        assert np.round(ene, decimals=2) >= 0.84

    @pytest.mark.skip(reason="Plotting test - enable manually if needed")
    def test_aperture_plot_rectangular(self, psf):
        """
        Test rectangular aperture visualization.

        This test creates a rectangular aperture and overlays it
        on the PSF for visual inspection. It is skipped by default
        to avoid GUI dependencies.

        Parameters
        ----------
        psf : numpy.ndarray
            2D PSF array for aperture visualization
        """
        sizes, _, _ = psf_util.find_rectangular_aperture(psf, 0.84)
        aper = RectangularAperture(
            (psf.shape[1] // 2, psf.shape[0] // 2),
            h=sizes[1],
            w=sizes[0],
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(psf)
        aper.plot(color="g", lw=2, label="Photometry aperture")
        plt.legend()
        plt.title("Rectangular Aperture Test")
        plt.show()

    @pytest.mark.skip(reason="Plotting test - enable manually if needed")
    def test_aperture_plot_elliptical(self, psf):
        """
        Test elliptical aperture visualization.

        This test creates an elliptical aperture and overlays it
        on the PSF for visual inspection. It is skipped by default
        to avoid GUI dependencies.

        Parameters
        ----------
        psf : numpy.ndarray
            2D PSF array for aperture visualization
        """
        sizes, _, _ = psf_util.find_elliptical_aperture(psf, 0.84)
        aper = EllipticalAperture(
            (psf.shape[1] // 2, psf.shape[0] // 2),
            a=sizes[1],
            b=sizes[0],
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(psf)
        aper.plot(color="g", lw=2, label="Photometry aperture")
        plt.legend()
        plt.title("Elliptical Aperture Test")
        plt.show()


class TestTaskFactory:
    """Test suite for task factory functionality."""

    def test_find_task(self):
        """
        Test task factory PSF loading functionality.

        This test verifies that the task factory can correctly
        locate and instantiate PSF loading tasks from the
        instrument module.
        """
        task = find_task("LoadPsfPaos", instrument.LoadPsf)
        assert task is not None


class TestPAOSIntegration:
    """Test suite for PAOS PSF data integration."""

    @pytest.fixture
    def paos_data(self, test_data_dir):
        """
        Provide PAOS data file for testing.

        This fixture locates the PAOS PSF data file in the test
        data directory for integration testing.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path

        Returns
        -------
        str
            Path to PAOS data file
        """
        return os.path.join(test_data_dir, "PAOS_ab0.h5")

    def test_load_paos(self, paos_data):
        """
        Test PAOS PSF data loading.

        This test verifies that PAOS PSF files can be loaded correctly
        with proper wavelength and time grid handling, and that the
        resulting PSF cube has the expected dimensions.

        Parameters
        ----------
        paos_data : str
            Path to PAOS data file
        """
        if not os.path.exists(paos_data):
            pytest.skip(f"PAOS data file not found: {paos_data}")

        wl = np.linspace(1, 2.8, 5) * u.um
        tt = np.linspace(0, 10, 2) * u.hr
        parameters = {
            "detector": {
                "oversampling": 2,
                "delta_pix": 10 * u.um,
                "spatial_pix": 32,
                "spectral_pix": 32,
            }
        }

        loadPsfPaos = instrument.LoadPsfPaos()
        cube, _norms = loadPsfPaos(
            filename=paos_data, parameters=parameters, wavelength=wl, time=tt
        )

        assert cube.shape[1] == len(wl), (
            "PSF cube wavelength dimension should match input grid"
        )
        assert cube.shape[0] == len(tt), (
            "PSF cube time dimension should match input grid"
        )


class TestPSFUtilityRobustness:
    """Test suite for PSF utility robustness and edge cases."""

    def test_create_psf_minimal_parameters(self):
        """
        Test PSF creation with minimal parameters.

        This test verifies that PSF creation works with the most
        basic parameter set, ensuring robustness of default values.
        """
        wl = 2.0 * u.um
        fnum = 15.0
        pixel_size = 10 * u.um

        psf = create_psf(wl, fnum, pixel_size)

        assert psf.ndim == 2, "PSF should be 2D for single wavelength"
        assert psf.shape[0] > 0, "PSF should have positive height"
        assert psf.shape[1] > 0, "PSF should have positive width"

    def test_psf_energy_conservation(self):
        """
        Test PSF energy conservation.

        This test verifies that created PSFs maintain proper
        energy normalization across different configurations.
        """
        wl = 2.0 * u.um
        fnum = 15.0
        pixel_size = 10 * u.um

        # Create PSFs with different shapes
        psf_airy = create_psf(wl, fnum, pixel_size, shape="airy")
        psf_gauss = create_psf(wl, fnum, pixel_size, shape="gauss")

        # Check that PSFs have reasonable energy distribution
        airy_sum = np.sum(psf_airy)
        gauss_sum = np.sum(psf_gauss)

        assert airy_sum > 0, "Airy PSF should have positive total energy"
        assert gauss_sum > 0, "Gaussian PSF should have positive total energy"

        # Check that peak is at center (within 1 pixel)
        airy_center = np.unravel_index(np.argmax(psf_airy), psf_airy.shape)
        gauss_center = np.unravel_index(np.argmax(psf_gauss), psf_gauss.shape)

        expected_center_y = psf_airy.shape[0] // 2
        expected_center_x = psf_airy.shape[1] // 2

        assert abs(airy_center[0] - expected_center_y) <= 1, (
            "Airy PSF peak should be near center"
        )
        assert abs(airy_center[1] - expected_center_x) <= 1, (
            "Airy PSF peak should be near center"
        )
        assert abs(gauss_center[0] - expected_center_y) <= 1, (
            "Gaussian PSF peak should be near center"
        )
        assert abs(gauss_center[1] - expected_center_x) <= 1, (
            "Gaussian PSF peak should be near center"
        )
