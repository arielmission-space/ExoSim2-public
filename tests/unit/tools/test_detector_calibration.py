"""
Unit tests for detector calibration tools.

This module contains tests for detector-specific calibration tools including:
- Quantum efficiency maps
- Dead pixel maps
- Dark current maps
"""

import os

import astropy.units as u
import numpy as np
import pytest

from exosim.tools import (
    DarkCurrentMap,
    DeadPixelsMap,
    QuantumEfficiencyMap,
)
from exosim.utils import RunConfig


@pytest.fixture
def skip_plot():
    """Fixture to control whether plotting tests are skipped."""
    return True  # Skip plotting tests by default


@pytest.fixture
def seed():
    """Fixture to provide a random seed for reproducible tests."""
    return 42


@pytest.mark.usefixtures("test_data_dir")
class TestQuantumEfficiencyMap:
    """Test suite for quantum efficiency map generation."""

    @pytest.fixture(autouse=True)
    def _init(self, test_data_dir):
        """Initialize test parameters for QE map tests."""
        self.params = {
            "time_grid": {
                "start_time": 0 * u.hr,
                "end_time": 1 * u.hr,
                "low_frequencies_resolution": 1 * u.hr,
            },
            "channel": {
                "value": "test",
                "detector": {
                    "qe_sigma": 0.5,
                    "spatial_pix": 10,
                    "spectral_pix": 20,
                },
            },
        }
        self.fname = os.path.join(test_data_dir, "qe_test.h5")

    def test_constant(self):
        """Test quantum efficiency map with constant values (zero sigma)."""
        self.params["channel"]["detector"]["qe_sigma"] = 0

        qe = QuantumEfficiencyMap(options_file=self.params, output=self.fname)
        q_map = qe.results["test"]

        constant_map = np.ones(
            (
                1,
                self.params["channel"]["detector"]["spatial_pix"],
                self.params["channel"]["detector"]["spectral_pix"],
            )
        )
        np.testing.assert_array_equal(q_map.data, constant_map)

        os.remove(self.fname)

    def test_no_output(self):
        """Test quantum efficiency map generation without file output."""
        self.params["channel"]["detector"]["qe_sigma"] = 0

        qe_map = QuantumEfficiencyMap(options_file=self.params)
        assert qe_map is not None

    def test_size(self):
        """Test that quantum efficiency map has correct dimensions."""
        qe = QuantumEfficiencyMap(options_file=self.params, output=self.fname)
        q_map = qe.results["test"]

        assert q_map.data.shape == (1, 10, 20)
        os.remove(self.fname)

    def test_value(self):
        """Test quantum efficiency map statistical properties."""
        self.params["channel"]["detector"]["qe_sigma"] = 0.1

        qe = QuantumEfficiencyMap(options_file=self.params, output=self.fname)
        q_map = qe.results["test"]

        np.testing.assert_almost_equal(np.std(q_map.data[0]), 0.1, decimal=1)
        np.testing.assert_almost_equal(np.mean(q_map.data[0]), 1, decimal=1)
        os.remove(self.fname)

    def test_time_variation(self):
        """Test quantum efficiency variation over time with aging effects."""
        params = {
            "time_grid": {
                "start_time": 0 * u.hr,
                "end_time": 5 * u.hr,
                "low_frequencies_resolution": 1 * u.hr,
            },
            "channel": {
                "value": "test",
                "detector": {
                    "qe_sigma": 0.1,
                    "qe_aging_factor": 0.05,
                    "qe_aging_time_scale": 5 * u.hr,
                    "spatial_pix": 10,
                    "spectral_pix": 20,
                },
            },
        }

        qe = QuantumEfficiencyMap(options_file=params, output=self.fname)
        q_map = qe.results["test"]

        np.testing.assert_almost_equal(np.std(q_map.data[0]), 0.1, decimal=1)
        np.testing.assert_almost_equal(np.mean(q_map.data[0]), 1, decimal=1)
        np.testing.assert_almost_equal(np.mean(q_map.data[-1]), 1 - 0.05, decimal=1)
        os.remove(self.fname)


class TestDeadPixelMap:
    """Test suite for dead pixel map generation."""

    def test_constant_value(self):
        """Test dead pixel map generation with constant parameters."""
        params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 10,
                    "spectral_pix": 20,
                    "dp_mean": 10,
                },
            }
        }
        test_out = DeadPixelsMap(params)
        assert test_out.results["test"]["spatial_coords"].size == 10
        assert test_out.results["test"]["spectral_coords"].size == 10

    def test_plot(self, skip_plot):
        """Test dead pixel map visualization (skipped if no plotting)."""
        if skip_plot:
            pytest.skip("This test only produces plots")
        params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 32,
                    "spectral_pix": 32,
                    "dp_mean": 10,
                },
            }
        }

        test_out = DeadPixelsMap(params)
        dead_coords = test_out.results["test"]

        y_size = params["channel"]["detector"]["spatial_pix"]
        x_size = params["channel"]["detector"]["spectral_pix"]
        dead_pixels_map = np.ones((x_size, y_size))

        # Mark dead pixels in the map
        for x, y in dead_coords["spectral_coords", "spatial_coords"]:
            dead_pixels_map[y, x] = 0

    def test_output(self, test_data_dir):
        """Test dead pixel map file output functionality."""
        params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 10,
                    "spectral_pix": 20,
                    "dp_mean": 10,
                },
            }
        }
        DeadPixelsMap(params, test_data_dir)

        fname = os.path.join(test_data_dir, "dp_map_test.csv")
        assert os.path.isfile(fname)
        os.remove(fname)

    def test_random_map(self, seed):
        """Test statistical properties of randomly generated dead pixel maps."""
        params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 1000,
                    "spectral_pix": 1000,
                    "dp_mean": 10,
                    "dp_sigma": 1,
                },
            }
        }
        test_x = []
        test_y = []

        # Generate multiple realizations to test statistical properties
        for i in range(1000):
            RunConfig.random_seed += i
            test_out = DeadPixelsMap(params)
            test_x += [test_out.results["test"]["spatial_coords"].size]
            test_y += [test_out.results["test"]["spectral_coords"].size]
        RunConfig.random_seed = seed

        np.testing.assert_allclose(np.mean(test_x), 10, 1)
        np.testing.assert_allclose(np.mean(test_y), 10, 1)
        np.testing.assert_allclose(np.std(test_x), 1, 0.1)
        np.testing.assert_allclose(np.std(test_y), 1, 0.1)


class TestDarkCurrentMap:
    """Test suite for dark current map generation."""

    def __init__(self):
        """Initialize test parameters for dark current map tests."""
        self.parameters = {
            "detector": {
                "dc_median": 1.0 * u.ct / u.s,
                "dc_sigma": 0.1 * u.ct / u.s,
                "spatial_pix": 64,
                "spectral_pix": 64,
                "oversampling": 3,
                "dc_aging_factor": 0.1,
                "dc_aging_time_scale": 10.0,
            }
        }

        self.times = np.arange(start=0.0, stop=1.0, step=0.25) * u.s

    def test_values(self):
        """Test dark current map generation and statistical properties."""
        darkCurrentMap = DarkCurrentMap()
        dc_map = darkCurrentMap(parameters=self.parameters, time=self.times)

        # Test the map shape
        assert dc_map.data.shape == (
            self.times.size,
            self.parameters["detector"]["spatial_pix"]
            * self.parameters["detector"]["oversampling"],
            self.parameters["detector"]["spectral_pix"]
            * self.parameters["detector"]["oversampling"],
        )

        # Test the dictionary dc_mean key
        assert "dc_mean" in self.parameters["detector"]

        # Test the statistical distribution
        np.testing.assert_allclose(
            np.median(dc_map.data[0, :, :]),
            self.parameters["detector"]["dc_median"].value,
            atol=0.1,
        )
        np.testing.assert_allclose(
            np.mean(dc_map.data[0, :, :]),
            self.parameters["detector"]["dc_mean"].value,
            atol=0.1,
        )
        np.testing.assert_allclose(
            np.std(dc_map.data[0, :, :]),
            self.parameters["detector"]["dc_sigma"].value,
            atol=0.01,
        )

    def test_compute_dc_mean(self):
        """Test computation of dark current mean from median and sigma."""
        from copy import deepcopy as dc

        detector = dc(self.parameters["detector"])

        darkCurrentMap = DarkCurrentMap()

        mu, sigma = 1.0, 0.1
        median = np.exp(mu)
        var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        std = np.sqrt(var)

        detector.update(
            {"dc_median": median * u.ct / u.s, "dc_sigma": std * u.ct / u.s}
        )
        darkCurrentMap.compute_dc_mean(detector=detector)

        dc_mean = detector["dc_mean"].value
        mean = np.exp(mu + sigma**2 / 2)

        assert dc_mean == mean
