"""
Unit tests for sub-exposure tasks and jitter modeling.

This module tests sub-exposure functionality including pointing jitter,
channel jitter estimation, foreground addition, and quantum efficiency
map application.
"""

import logging
import os

import astropy.units as u
import numpy as np
import pytest

from exosim.log import set_log_level
from exosim.models.signal import Counts, CountsPerSecond, Dimensionless
from exosim.output import SetOutput
from exosim.tasks.subexposures import (
    AddForegrounds,
    ApplyQeMap,
    EstimateChJitter,
    EstimatePointingJitter,
)

set_log_level(logging.DEBUG)


@pytest.fixture
def output_file(test_data_dir):
    """
    Create temporary output file for sub-exposure testing.

    This fixture creates a temporary HDF5 output file for testing
    sub-exposure operations and cleans up after the test.

    Parameters
    ----------
    test_data_dir : pathlib.Path
        Test data directory path

    Yields
    ------
    str
        Path to temporary output file
    """
    fname = os.path.join(test_data_dir, "output_test.h5")
    yield fname
    if os.path.exists(fname):
        os.remove(fname)


class TestPointingJitter:
    """Test suite for pointing jitter estimation."""

    def test_pointing_jitter_units(self):
        """
        Test pointing jitter output units.

        This test verifies that the pointing jitter estimation task
        produces outputs with correct angular units (degrees) for
        both spatial and spectral jitter components.
        """
        parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            },
            "jitter": {
                "jitter_task": "EstimatePointingJitter",
                "spatial": 0.2 * u.arcsec,
                "spectral": 0.4 * u.arcsec,
                "frequency_resolution": 0.01 * u.s,
            },
        }

        estimatePointingJitter = EstimatePointingJitter()
        jitter_spa, jitter_spe, _jitter_time = estimatePointingJitter(
            parameters=parameters
        )

        assert jitter_spa.unit == "deg", "Spatial jitter should be in degrees"
        assert jitter_spe.unit == "deg", "Spectral jitter should be in degrees"

    def test_pointing_jitter_statistics(self):
        """
        Test pointing jitter statistical properties.

        This test verifies that the generated pointing jitter has
        the expected statistical properties (standard deviation)
        matching the input parameters.
        """
        parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            },
            "jitter": {
                "jitter_task": "EstimatePointingJitter",
                "spatial": 0.2 * u.arcsec,
                "spectral": 0.4 * u.arcsec,
                "frequency_resolution": 0.01 * u.s,
            },
        }

        estimatePointingJitter = EstimatePointingJitter()
        jitter_spa, jitter_spe, _jitter_time = estimatePointingJitter(
            parameters=parameters
        )

        # Check standard deviations match input parameters within tolerance
        np.testing.assert_allclose(
            jitter_spa.to(u.arcsec).std().value,
            parameters["jitter"]["spatial"].to(u.arcsec).value,
            atol=0.002,
            err_msg="Spatial jitter standard deviation should match input",
        )
        np.testing.assert_allclose(
            jitter_spe.to(u.arcsec).std().value,
            parameters["jitter"]["spectral"].to(u.arcsec).value,
            atol=0.002,
            err_msg="Spectral jitter standard deviation should match input",
        )


class TestChannelJitter:
    """Test suite for channel jitter estimation."""

    def test_channel_jitter_statistics(self):
        """
        Test channel jitter pixel-level statistics.

        This test verifies that channel jitter estimation correctly
        converts pointing jitter to pixel-level jitter using the
        detector plate scale and produces expected statistics.
        """
        # Set up input jitter time series
        jitter_time = np.arange(0, 60, 0.005) * u.s
        np.random.seed(42)  # For reproducible results
        jitter_spa = np.random.normal(0, 0.01, jitter_time.size) * u.arcsec
        jitter_spe = np.random.normal(0, 0.05, jitter_time.size) * u.arcsec

        parameters = {
            "detector": {
                "plate_scale": {
                    "spatial": 0.01 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
                "delta_pix": 0,
                "oversampling": 1,
            },
            "readout": {"readout_frequency": 200 * u.Hz},
        }

        estimateChJitter = EstimateChJitter()
        jitter_spe_out, jitter_spa_out, jit_y, jit_x, _new_jit_time = estimateChJitter(
            parameters=parameters,
            pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
        )

        # Check pixel-level jitter statistics (approximately 1 pixel std)
        np.testing.assert_allclose(
            np.std(jit_y),
            1,
            atol=0.1,
            err_msg="Y-direction pixel jitter should have std ≈ 1 pixel",
        )
        np.testing.assert_allclose(
            np.std(jit_x),
            1,
            atol=0.1,
            err_msg="X-direction pixel jitter should have std ≈ 1 pixel",
        )

        # Check that angular jitter is preserved
        np.testing.assert_allclose(
            np.std(jitter_spe_out),
            0.05 * u.arcsec,
            atol=0.01,
            err_msg="Spectral jitter magnitude should be preserved",
        )
        np.testing.assert_allclose(
            np.std(jitter_spa_out),
            0.01 * u.arcsec,
            atol=0.01,
            err_msg="Spatial jitter magnitude should be preserved",
        )


class TestForegroundOperations:
    """Test suite for foreground addition operations."""

    def test_add_foregrounds(self, output_file):
        """
        Test foreground addition to sub-exposures.

        This test verifies that foreground signals are correctly
        added to sub-exposure data, with proper handling of
        integration time scaling and signal units.

        Parameters
        ----------
        output_file : str
            Path to temporary output file
        """
        # Create test foreground signal (CountsPerSecond)
        frg = CountsPerSecond(
            spectral=np.arange(0, 10),
            data=np.ones((10, 10)) * 2,  # 2 counts/sec per pixel
            metadata={"oversampling": 1},
        )

        integration_time = np.ones(10) * u.s  # 1 second integration
        data = np.ones((10, 10, 10))  # 1 count per pixel baseline

        # Set up cached output for testing
        output = SetOutput(output_file)
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                time=np.arange(0, 10) * u.hr,
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            # Apply foreground addition
            addForegrounds = AddForegrounds()
            result = addForegrounds(
                subexposures=input_signal,
                frg_focal_plane=frg,
                integration_time=integration_time,
            )

            # Verify foreground addition: 1 (baseline) + 2 (foreground) = 3
            np.testing.assert_array_equal(
                result.dataset[0],
                np.ones((10, 10), dtype=np.float64) * 3,
                err_msg="First sub-exposure should have baseline + foreground counts",
            )

            # Verify total counts
            expected_total = 3 * 10 * 10 * 10  # 3 counts x 10x10 pixels x 10 time steps
            assert np.sum(result.dataset) == expected_total, (
                "Total counts should equal baseline + foreground for all pixels and times"
            )


class TestQuantumEfficiencyMapping:
    """Test suite for quantum efficiency map application."""

    def test_apply_quantum_efficiency_map(self, output_file):
        """
        Test quantum efficiency map application.

        This test verifies that quantum efficiency maps are correctly
        applied to sub-exposure data, scaling the signal appropriately
        while maintaining proper data types and caching.

        Parameters
        ----------
        output_file : str
            Path to temporary output file
        """
        # Create test QE map (50% efficiency everywhere)
        qe_map = Dimensionless(
            spectral=np.arange(0, 10),
            data=np.ones((1, 10, 10)) * 0.5,  # 50% quantum efficiency
        )

        data = np.ones((10, 10, 10)) * 4  # 4 counts per pixel baseline

        # Set up cached output for testing
        output = SetOutput(output_file)
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                time=np.arange(0, 10) * u.hr,
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            # Apply quantum efficiency map
            applyQeMap = ApplyQeMap()
            result = applyQeMap(subexposures=input_signal, qe_map=qe_map)

            # Verify QE scaling: 4 (baseline) x 0.5 (QE) = 2
            np.testing.assert_array_equal(
                result.dataset[0],
                np.ones((10, 10), dtype=np.float64) * 2,
                err_msg="First sub-exposure should be scaled by quantum efficiency",
            )

            # Verify total counts after QE application
            expected_total = 2 * 10 * 10 * 10  # 2 counts x 10x10 pixels x 10 time steps
            assert np.sum(result.dataset) == expected_total, (
                "Total counts should be scaled by quantum efficiency"
            )


class TestSubExposureIntegration:
    """Test suite for sub-exposure system integration."""

    def test_jitter_time_series_consistency(self):
        """
        Test consistency of jitter time series between tasks.

        This test verifies that jitter time series are properly
        handled and maintain consistency when passed between
        different sub-exposure processing tasks.
        """
        # Create pointing jitter
        parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 1.0 * u.hr,
                "low_frequencies_resolution": 1.0 * u.hr,
            },
            "jitter": {
                "jitter_task": "EstimatePointingJitter",
                "spatial": 0.1 * u.arcsec,
                "spectral": 0.2 * u.arcsec,
                "frequency_resolution": 0.001 * u.s,
            },
        }

        estimatePointingJitter = EstimatePointingJitter()
        jitter_spa, jitter_spe, jitter_time = estimatePointingJitter(
            parameters=parameters
        )

        # Verify time series properties
        assert len(jitter_spa) == len(jitter_spe), (
            "Spatial and spectral jitter should have same length"
        )
        assert len(jitter_spa) == len(jitter_time), (
            "Jitter arrays should match time array length"
        )

        # Verify time series is monotonically increasing
        assert np.all(np.diff(jitter_time.value) > 0), (
            "Time series should be monotonically increasing"
        )

    def test_signal_data_types(self, output_file):
        """
        Test signal data type preservation.

        This test verifies that sub-exposure processing tasks
        properly preserve data types and signal properties
        throughout the processing chain.

        Parameters
        ----------
        output_file : str
            Path to temporary output file
        """
        # Create test signals with specific data types
        data = np.ones((5, 8, 8), dtype=np.float32) * 10

        output = SetOutput(output_file)
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 8),
                time=np.arange(0, 5) * u.hr,
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="TestSubExposures",
                output_path=None,
                dtype=np.float32,
            )

            # Test that signal properties are preserved
            assert input_signal.data.dtype == np.float32, (
                "Data type should be preserved"
            )
            assert input_signal.shape == (5, 8, 8), "Shape should be preserved"

            # Test foreground addition preserves properties
            frg = CountsPerSecond(
                spectral=np.arange(0, 8),
                data=np.ones((8, 8)) * 1.5,
                metadata={"oversampling": 1},
            )
            integration_time = np.ones(5) * u.s

            addForegrounds = AddForegrounds()
            result = addForegrounds(
                subexposures=input_signal,
                frg_focal_plane=frg,
                integration_time=integration_time,
            )

            # Verify data type consistency after processing
            assert isinstance(result, Counts), "Result should be Counts signal"
            assert result.shape == input_signal.shape, (
                "Shape should be preserved after processing"
            )
