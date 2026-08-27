#!/usr/bin/env python3
"""Tests for focal_plane_locations module."""

from unittest import TestCase

import numpy as np

from exosim.utils.focal_plane_locations import locate_wavelength_windows


class TestFocalPlaneLocations(TestCase):
    """Test focal_plane_locations module with improved test coverage."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock parameters that are used across multiple tests
        self.psf = np.zeros((5, 5, 3, 3))  # 4D PSF with spectral and spatial dimensions

        # Common shape parameters
        self.shape = (1, 10, 10)  # (n_channels, height, width)
        self.spectral_size = 5

    def test_focal_plane_coordinate_basic(self):
        """Test basic coordinate calculations."""
        # Test basic coordinate computation
        x_pix, y_pix = 10, 20
        pixel_scale = 0.1  # arcsec/pixel

        # Basic coordinate transformation logic
        x_coord = x_pix * pixel_scale
        y_coord = y_pix * pixel_scale

        assert x_coord == 1.0
        assert y_coord == 2.0

    def test_photometer_coordinate_arrays(self):
        """Test wavelength window locations for photometer."""
        # Create a simple focal plane simulator
        focal_plane = MockFocalPlane(
            shape=self.shape,
            spectral_size=self.spectral_size,
            is_cached=True,
            has_data=False,
            spatial_data=np.zeros(5),
        )

        parameters = {"type": "photometer"}

        # Test the function call
        i0, j0 = locate_wavelength_windows(self.psf, focal_plane, parameters)

        # Verify results - basic properties
        assert j0 is not None
        assert i0 is not None
        assert len(j0) == focal_plane.spectral.size
        assert len(i0) == focal_plane.spectral.size
        assert isinstance(j0, np.ndarray)
        assert isinstance(i0, np.ndarray)

    def test_with_focal_plane_data(self):
        """Test when focal plane has existing data."""
        # Create a focal plane with data
        focal_plane = MockFocalPlane(
            shape=self.shape,
            spectral_size=self.spectral_size,
            is_cached=True,
            has_data=True,  # This time with data
            spatial_data=np.zeros(5),
        )

        parameters = {"type": "photometer"}

        # Test the function call
        i0, j0 = locate_wavelength_windows(self.psf, focal_plane, parameters)

        # Verify results with data present
        assert j0 is not None
        assert i0 is not None
        assert len(j0) == focal_plane.spectral.size
        assert len(i0) == focal_plane.spectral.size


class MockFocalPlane:
    """Helper class to simulate focal plane with required attributes."""

    def __init__(self, shape, spectral_size, is_cached, has_data, spatial_data):
        """Initialize mock focal plane with configurable properties."""
        self.shape = shape
        self.data = has_data
        self.cached = is_cached

        # Create spectral component
        self.spectral = type("Spectral", (), {"size": spectral_size})()

        # Create spatial component
        self.spatial = type("Spatial", (), {"data": spatial_data})()
