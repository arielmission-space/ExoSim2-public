"""
Unit tests for detector tasks in ExoSim2.

This module tests all detector-related tasks including dark current addition,
shot noise modeling, pixel non-linearity, saturation, readout noise,
cosmic ray simulation, and analog-to-digital conversion.
"""

import logging
import os
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.io import ascii
from astropy.table import Table

from exosim.log import set_log_level
from exosim.models.signal import Counts
from exosim.output import SetOutput
from exosim.tasks.detector import (
    AccumulateSubExposures,
    AddConstantDarkCurrent,
    AddCosmicRays,
    AddDarkCurrentMapNumpy,
    AddGainDrift,
    AddKTC,
    AddNormalReadNoise,
    AddReadNoiseMapNumpy,
    AddShotNoise,
    AnalogToDigital,
    ApplyDeadPixelMapNumpy,
    ApplyDeadPixelsMap,
    ApplyPixelsNonLinearity,
    ApplySimpleSaturation,
    MergeGroups,
)

set_log_level(logging.DEBUG)


class TestDarkCurrentOperations:
    """Test suite for dark current addition operations."""

    def test_constant_dark_current_values(self, test_data_dir):
        """
        Test constant dark current addition to sub-exposures.

        This test verifies that constant dark current is correctly added
        to detector sub-exposures with proper scaling by integration time.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = test_data_dir / "output_test.h5"
        output = SetOutput(str(fname))

        data = np.zeros((10, 10, 10))
        integration_times = np.ones(data.shape[0]) * u.s

        parameters = {"detector": {"dc_mean": 5 * u.ct / u.s}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addConstantDarkCurrent = AddConstantDarkCurrent()
            addConstantDarkCurrent(
                subexposures=input_signal,
                parameters=parameters,
                integration_times=integration_times,
            )

            np.testing.assert_array_equal(
                input_signal.dataset[0], np.ones((10, 10), dtype=np.float64) * 5
            )

        os.remove(fname)

    def test_dark_current_map_values(self, test_data_dir):
        """
        Test dark current map application to sub-exposures.

        This test verifies that spatially-varying dark current maps
        are correctly loaded and applied to detector data.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.zeros((10, 10, 10))
        integration_times = np.ones(data.shape[0]) * u.s

        dc = np.ones(data[0].shape) * 5
        dc_map_fname = os.path.join(test_data_dir, "dc_map.npy")
        np.save(dc_map_fname, dc)

        parameters = {"detector": {"dc_map_filename": dc_map_fname}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addDarkCurrentMapNumpy = AddDarkCurrentMapNumpy()
            addDarkCurrentMapNumpy(
                subexposures=input_signal,
                parameters=parameters,
                integration_times=integration_times,
            )

            np.testing.assert_array_equal(
                input_signal.dataset[0], np.ones((10, 10), dtype=np.float64) * 5
            )

        os.remove(fname)
        os.remove(dc_map_fname)

    def test_dark_current_map_error_handling(self, test_data_dir):
        """
        Test error handling for mismatched dark current map dimensions.

        This test verifies that appropriate errors are raised when
        dark current maps don't match detector dimensions.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 100, 100)) * 100
        integration_times = np.ones(data.shape[0]) * u.s

        # Create mismatched dimensions
        dc = np.ones((10, 20)) * 5
        dc_map_fname = os.path.join(test_data_dir, "dc_map.npy")
        np.save(dc_map_fname, dc)

        parameters = {"detector": {"dc_map_filename": dc_map_fname}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )
            addDarkCurrentMapNumpy = AddDarkCurrentMapNumpy()
            with pytest.raises(
                IOError, match="Map dimensions do not match signal shape"
            ):
                addDarkCurrentMapNumpy(
                    subexposures=input_signal,
                    parameters=parameters,
                    integration_times=integration_times,
                )

        os.remove(fname)
        os.remove(dc_map_fname)


class TestPixelNonLinearityOperations:
    """Test suite for pixel non-linearity operations."""

    def test_apply_nonlinearity_correction(self, test_data_dir):
        """
        Test application of pixel non-linearity correction.

        This test verifies that pixel non-linearity corrections are
        properly applied to detector counts, reducing signal at high levels.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        from exosim.tools import PixelsNonLinearity

        params = {
            "channel": {
                "value": "test",
                "detector": {
                    "spatial_pix": 10,
                    "spectral_pix": 10,
                    "well_depth": 10000 * u.ct,
                },
            }
        }

        pnl_dict = PixelsNonLinearity(params, show_results=False)

        data = np.ones((1, 10, 10)) * 10000

        fname = os.path.join(test_data_dir, "output_test_npl.h5")
        output = SetOutput(fname)
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                time=[0] * u.hr,
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            applyPixelsNonLinearity = ApplyPixelsNonLinearity()
            applyPixelsNonLinearity(
                subexposures=input_signal, parameters=pnl_dict.results["test"]
            )

            # Verify 5% reduction in signal due to non-linearity
            np.testing.assert_allclose(
                input_signal.dataset[0], np.ones((10, 10)) * 10000 * 0.95, rtol=1e-3
            )
            os.remove(fname)


class TestNoiseOperations:
    """Test suite for noise addition operations."""

    def test_shot_noise_statistics(self, test_data_dir):
        """
        Test shot noise addition with proper Poisson statistics.

        This test verifies that shot noise follows Poisson statistics
        with variance equal to the mean signal level.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((10, 10, 100)) * 10000

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addShotNoise = AddShotNoise()
            addShotNoise(subexposures=input_signal)

            # Test Poisson statistics: mean ≈ 10000, std ≈ sqrt(10000) = 100
            np.testing.assert_allclose(np.mean(input_signal.dataset), 10000, rtol=0.001)
            np.testing.assert_allclose(np.std(input_signal.dataset), 100, rtol=0.1)

        os.remove(fname)

    def test_read_noise_statistics(self, test_data_dir):
        """
        Test read noise addition with proper Gaussian statistics.

        This test verifies that read noise follows Gaussian distribution
        with specified standard deviation.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 100, 100)) * 100

        parameters = {"detector": {"read_noise_sigma": 1 * u.ct}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addNormalReadNoise = AddNormalReadNoise()
            addNormalReadNoise(subexposures=input_signal, parameters=parameters)

            # Test Gaussian statistics: mean ≈ 100, std ≈ 1
            np.testing.assert_allclose(np.median(input_signal.dataset), 100, atol=0.5)
            np.testing.assert_allclose(np.std(input_signal.dataset), 1, atol=0.1)

            # Test frame-by-frame statistics
            np.testing.assert_allclose(
                np.median(input_signal.dataset[0]), 100, atol=0.5
            )
            np.testing.assert_allclose(np.std(input_signal.dataset[0]), 1, atol=0.1)

            # Verify independence between frames
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    input_signal.dataset[0],
                    input_signal.dataset[1],
                )

        os.remove(fname)

    def test_read_noise_map_application(self, test_data_dir):
        """
        Test read noise map application with spatial variations.

        This test verifies that spatially-varying read noise maps
        are correctly applied to detector data.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 64, 64)) * 100

        read_map_fname = os.path.join(test_data_dir, "read_map.npy")
        read = np.ones(data[0].shape)
        np.save(read_map_fname, read)

        parameters = {"detector": {"read_noise_filename": read_map_fname}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addReadNoiseMapNumpy = AddReadNoiseMapNumpy()
            addReadNoiseMapNumpy(subexposures=input_signal, parameters=parameters)

            # Test statistical properties
            np.testing.assert_allclose(np.median(input_signal.dataset), 100, atol=0.5)
            np.testing.assert_allclose(np.std(input_signal.dataset), 1, atol=0.1)

            # Test frame independence
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    input_signal.dataset[0],
                    input_signal.dataset[1],
                )

        os.remove(fname)
        os.remove(read_map_fname)

    def test_read_noise_map_error_handling(self, test_data_dir):
        """
        Test error handling for mismatched read noise map dimensions.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 100, 100)) * 100

        read_map_fname = os.path.join(test_data_dir, "read_map.npy")
        read = np.ones((20, 10))  # Wrong dimensions
        np.save(read_map_fname, read)

        parameters = {"detector": {"read_noise_filename": read_map_fname}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )
            addReadNoiseMapNumpy = AddReadNoiseMapNumpy()
            with pytest.raises(
                IOError, match="Map dimensions do not match signal shape"
            ):
                addReadNoiseMapNumpy(subexposures=input_signal, parameters=parameters)

        os.remove(fname)
        os.remove(read_map_fname)

    def test_ktc_noise_behavior(self, test_data_dir):
        """
        Test kTC noise addition with proper correlation behavior.

        This test verifies that kTC noise is correctly added with
        proper correlation within ramps and independence between ramps.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.zeros((1000, 100, 100))
        state_machine = np.arange(0, 250).astype(int)
        state_machine = np.repeat(state_machine, 4)

        parameters = {"detector": {"ktc_sigma": 1 * u.ct}}
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addReadNoise = AddKTC()
            addReadNoise(
                subexposures=input_signal,
                state_machine=state_machine,
                parameters=parameters,
            )

            # Test overall statistics
            np.testing.assert_allclose(np.median(input_signal.dataset), 0, atol=0.5)
            np.testing.assert_allclose(np.std(input_signal.dataset), 1, atol=0.2)

            # Test frame statistics
            np.testing.assert_allclose(np.median(input_signal.dataset[0]), 0, atol=0.1)
            np.testing.assert_allclose(np.std(input_signal.dataset[0]), 1, atol=0.1)

            # Test correlation within same ramp
            np.testing.assert_equal(input_signal.dataset[0], input_signal.dataset[1])
            np.testing.assert_equal(input_signal.dataset[4], input_signal.dataset[5])

            # Test independence between different ramps
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    input_signal.dataset[0],
                    input_signal.dataset[5],
                )

        os.remove(fname)


class TestDetectorEffectsOperations:
    """Test suite for detector effects and calibration operations."""

    def test_sub_exposure_accumulation(self, test_data_dir):
        """
        Test sub-exposure accumulation based on state machine.

        This test verifies that sub-exposures are correctly accumulated
        according to the detector state machine configuration.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((4, 10, 100)).astype(np.float64)
        state_machine = np.array([0, 0, 1, 1]).astype(int)

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            accumulateSubExposures = AccumulateSubExposures()
            accumulateSubExposures(
                subexposures=input_signal, state_machine=state_machine
            )

            # Verify accumulation: frames with same state are added together
            test_data = deepcopy(data)
            test_data[1] += test_data[0]
            test_data[3] += test_data[2]

            np.testing.assert_array_equal(input_signal.dataset[0:4], test_data)

        os.remove(fname)

    def test_simple_saturation(self, test_data_dir):
        """
        Test simple saturation clipping at well depth.

        This test verifies that pixel values exceeding the well depth
        are correctly clipped to the saturation level.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        parameters = {"detector": {"well_depth": 1000}}

        data = np.ones((10, 10, 10)) * 10
        data[0, 0, 0] = 1001  # Above well depth
        data[1, 1, 1] = 1100  # Well above well depth
        data[1, 1, 2] = 999  # Below well depth
        data[1, 1, 3] = 1000  # Exactly at well depth

        fname = os.path.join(test_data_dir, "output_test_sat.h5")
        output = SetOutput(fname)

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            applySimpleSaturation = ApplySimpleSaturation()
            applySimpleSaturation(subexposures=input_signal, parameters=parameters)

            # Verify saturation clipping
            assert input_signal.dataset[0, 0, 0] == 1000.0
            assert input_signal.dataset[1, 1, 1] == 1000.0
            assert input_signal.dataset[1, 1, 3] == 1000.0
            assert input_signal.dataset[1, 1, 2] == 999  # Below threshold unchanged

        os.remove(fname)

    def test_gain_drift_effects(self, test_data_dir):
        """
        Test gain drift modeling over time and wavelength.

        This test verifies that gain drift is correctly applied
        as a function of time and wavelength with proper amplitude.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        fname = os.path.join(test_data_dir, "output_test_gain.h5")
        output = SetOutput(fname)

        data = np.ones((100, 100, 100)) * 100
        time = np.arange(0, 100)

        parameters = {
            "detector": {
                "gain_drift": True,
                "gain_drift_task": AddGainDrift,
                "gain_drift_amplitude": 1e-2,
                "gain_coeff_order_t": 1,
                "gain_coeff_t_min": 1.0,
                "gain_coeff_t_max": 1.01,
                "gain_coeff_order_w": 1,
                "gain_coeff_w_min": 1.0,
                "gain_coeff_w_max": 1.01,
            }
        }
        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                time=time,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
                metadata={"integration_times": np.ones(100) * 1},
            )

            addGainDrift = AddGainDrift()
            addGainDrift(subexposures=input_signal, parameters=parameters)

            # Test gain drift amplitude
            signal_range = (
                np.max(input_signal.dataset) - np.min(input_signal.dataset)
            ) / np.min(input_signal.dataset)
            np.testing.assert_almost_equal(signal_range, 1e-2, decimal=4)

        os.remove(fname)


class TestDeadPixelOperations:
    """Test suite for dead pixel map operations."""

    def test_dead_pixel_table_map(self, test_data_dir):
        """
        Test dead pixel map application using coordinate tables.

        This test verifies that dead pixels specified in coordinate
        tables are correctly identified and set to zero.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        tab = Table()
        tab["spatial_coords"] = [0, 2, 4]
        tab["spectral_coords"] = [1, 3, 5]

        map_fname = os.path.join(test_data_dir, "dp_test.h5")

        ascii.write(tab, map_fname, format="ecsv", overwrite=True, delimiter=",")

        parameters = {"detector": {"dp_map": map_fname}}

        data = np.ones((10, 10, 10)) * 10

        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            applyDeadPixelsMap = ApplyDeadPixelsMap()
            applyDeadPixelsMap(subexposures=input_signal, parameters=parameters)

            # Verify dead pixels are set to zero
            assert input_signal.dataset[0, 0, 1] == 0.0
            assert input_signal.dataset[0, 2, 3] == 0.0
            assert input_signal.dataset[0, 4, 5] == 0.0
            # Verify live pixels are unchanged
            assert input_signal.dataset[0, 0, 3] == 10
            assert input_signal.dataset[0, 2, 1] == 10

        os.remove(fname)
        os.remove(map_fname)

    def test_dead_pixel_numpy_map(self, test_data_dir):
        """
        Test dead pixel map application using numpy arrays.

        This test verifies that dead pixels specified in numpy
        boolean maps are correctly applied to detector data.

        Parameters
        ----------
        test_data_dir : pathlib.Path
            Test data directory path
        """
        test_array = np.zeros((10, 10))
        test_array[0, 1] = 1
        test_array[2, 3] = 1
        test_array[4, 5] = 1

        map_fname = os.path.join(test_data_dir, "dp_test.npy")
        np.save(map_fname, test_array)

        parameters = {"detector": {"dp_map_filename": map_fname}}

        data = np.ones((10, 10, 10)) * 10

        fname = os.path.join(test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            applyDeadPixelMapNumpy = ApplyDeadPixelMapNumpy()
            applyDeadPixelMapNumpy(subexposures=input_signal, parameters=parameters)

            # Verify dead pixels are zeroed
            assert input_signal.dataset[0, 0, 1] == 0.0
            assert input_signal.dataset[0, 2, 3] == 0.0
            assert input_signal.dataset[0, 4, 5] == 0.0
            # Verify live pixels are preserved
            assert input_signal.dataset[0, 0, 3] == 10
            assert input_signal.dataset[0, 2, 1] == 10

        os.remove(fname)
        os.remove(map_fname)


class TestAnalogToDigitalConversion:
    """Test suite for analog-to-digital conversion operations."""

    @pytest.fixture(autouse=True)
    def _inject_test_data_dir(self, test_data_dir):
        """Inject test data directory for all test methods."""
        self.test_data_dir = test_data_dir

    def produce_ndrs(self, dtype, nbits):
        """
        Produce NDRs with specified data type and bit depth.

        Parameters
        ----------
        dtype : numpy.dtype
            Target data type for conversion
        nbits : int
            Number of bits for ADC conversion
        """
        fname = os.path.join(self.test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1, 10, 10)).astype(np.float64)

        parameters = {"detector": {"ADC_num_bit": nbits, "ADC_gain": 0.5}}

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            analogToDigital = AnalogToDigital()
            ndrs = analogToDigital(
                subexposures=input_signal,
                output=input_signal.output,
                parameters=parameters,
            )

            test_data = (np.ones((10, 10)) * parameters["detector"]["ADC_gain"]).astype(
                dtype
            )

            np.testing.assert_array_equal(ndrs.dataset[0], test_data)

        os.remove(fname)

    def test_adc_32_bit_conversion(self):
        """Test 32-bit ADC conversion."""
        self.produce_ndrs(np.int32, 32)

    def test_adc_16_bit_conversion(self):
        """Test 16-bit ADC conversion."""
        self.produce_ndrs(np.int16, 16)

    def test_adc_8_bit_conversion(self):
        """Test 8-bit ADC conversion."""
        self.produce_ndrs(np.int8, 8)

    def test_adc_default_behavior(self):
        """Test ADC conversion with default bit depth."""
        fname = os.path.join(self.test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1, 10, 10)).astype(np.float64)

        parameters = {"detector": {"ADC_gain": 0.5}}

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            analogToDigital = AnalogToDigital()
            ndrs = analogToDigital(
                subexposures=input_signal,
                output=input_signal.output,
                parameters=parameters,
            )

            test_data = (np.ones((10, 10)) * parameters["detector"]["ADC_gain"]).astype(
                "int32"
            )

            np.testing.assert_array_equal(ndrs.dataset[0], test_data)

        os.remove(fname)

    def test_adc_float_bit_depth(self):
        """Test ADC conversion with float bit depth."""
        self.produce_ndrs(np.int8, 16.0)

    def test_adc_non_standard_bits(self):
        """Test ADC conversion with non-standard bit depth."""
        self.produce_ndrs(np.int16, 12)

    def test_adc_invalid_float_bits(self):
        """Test error handling for invalid float bit depth."""
        with pytest.raises(TypeError):
            self.produce_ndrs(np.int16, 12.5)

    def test_adc_excessive_bits(self):
        """Test error handling for excessive bit depth."""
        with pytest.raises(
            ValueError, match=r"ADC bit depth .* exceeds maximum allowed"
        ):
            self.produce_ndrs(np.int32, 34)


class TestGroupOperations:
    """Test suite for group merging operations."""

    @pytest.fixture(autouse=True)
    def _inject_test_data_dir(self, test_data_dir):
        """Inject test data directory for all test methods."""
        self.test_data_dir = test_data_dir

    def test_group_merging(self):
        """
        Test merging of detector groups.

        This test verifies that detector groups are correctly merged
        according to the specified number of groups and NDRs.
        """
        fname = os.path.join(self.test_data_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((6, 10, 100)).astype(np.float64)
        data[1] += 1
        data[2] += 2
        data[4] += 1
        data[5] += 2

        n_groups = 2
        n_ndrs = 3

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            mergeGroups = MergeGroups()
            output_signal = mergeGroups(
                subexposures=input_signal,
                n_groups=n_groups,
                n_ndrs=n_ndrs,
                output=out,
            )

            # Verify merged group structure
            test_data = np.ones((2, 10, 100)).astype(np.float64) * 2

            np.testing.assert_array_equal(output_signal.dataset[0:2], test_data)

        os.remove(fname)


class TestCosmicRayOperations:
    """Test suite for cosmic ray simulation operations."""

    @pytest.fixture(autouse=True)
    def _inject_test_data_dir(self, test_data_dir):
        """Inject test data directory for all test methods."""
        self.test_data_dir = test_data_dir

    def test_cosmic_ray_interaction_counts(self):
        """
        Test cosmic ray interaction rate and detection.

        This test verifies that cosmic rays are generated at the
        expected rate and produce the correct number of interactions.
        """
        fname = os.path.join(self.test_data_dir, "output_test_cr.h5")
        output = SetOutput(fname)

        data = np.zeros((10, 100, 100))
        parameters = {
            "detector": {
                "spatial_pix": 100,
                "spectral_pix": 100,
                "well_depth": 10000,
                "delta_pix": 1 * u.cm,
                "cosmic_rays_rate": 1 / 100 / 100 * u.ct / u.cm**2 / u.s,
                "saturation_rate": 1,
            },
        }
        integration_times = np.ones(10)
        integration_times[1] *= 2  # Double integration time for one frame

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addCosmicRays = AddCosmicRays()
            addCosmicRays(
                subexposures=input_signal,
                parameters=parameters,
                integration_times=integration_times,
            )

            # Count cosmic ray events (saturated pixels)
            total_detected_events = sum(
                len(np.where(input_signal.dataset[t] == 10000)[0])
                for t in range(input_signal.dataset.shape[0])
            )
            expected_total_events = 11  # 10 frames + 1 extra for doubled integration
            assert total_detected_events == expected_total_events, (
                f"Expected {expected_total_events} events, but detected {total_detected_events}"
            )

        os.remove(fname)

    def shape_test(self, shape, n_pix):
        """
        Test cosmic ray interaction shapes.

        Parameters
        ----------
        shape : str
            Cosmic ray interaction shape
        n_pix : int
            Expected number of affected pixels
        """
        fname = self.test_data_dir / "output_test_cr.h5"
        output = SetOutput(str(fname))

        data = np.zeros((1, 100, 100))
        parameters = {
            "detector": {
                "spatial_pix": 100,
                "spectral_pix": 100,
                "well_depth": 10000,
                "delta_pix": 1 * u.cm,
                "cosmic_rays_rate": 1 / 100 / 100 * u.ct / u.cm**2 / u.s,
                "saturation_rate": 1,
                "interaction_shapes": {shape: 1},
            },
        }
        integration_times = [1]

        with output.use(cache=True) as out:
            input_signal = Counts(
                spectral=np.arange(0, 10),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            addCosmicRays = AddCosmicRays()
            addCosmicRays(
                subexposures=input_signal,
                parameters=parameters,
                integration_times=integration_times,
            )

            n_sat = np.where(input_signal.dataset[0] == 10000)[0]
            assert len(n_sat) <= n_pix
        os.remove(fname)

    def test_cosmic_ray_shapes_single(self):
        """Test single pixel cosmic ray interactions."""
        self.shape_test("single", 1)

    def test_cosmic_ray_shapes_line_horizontal(self):
        """Test horizontal line cosmic ray interactions."""
        self.shape_test("line_h", 2)

    def test_cosmic_ray_shapes_line_vertical(self):
        """Test vertical line cosmic ray interactions."""
        self.shape_test("line_v", 2)

    def test_cosmic_ray_shapes_quad(self):
        """Test quad pixel cosmic ray interactions."""
        self.shape_test("quad", 4)

    def test_cosmic_ray_shapes_cross(self):
        """Test cross-shaped cosmic ray interactions."""
        self.shape_test("cross", 5)

    def test_cosmic_ray_shapes_rect_horizontal(self):
        """Test horizontal rectangle cosmic ray interactions."""
        self.shape_test("rect_h", 6)

    def test_cosmic_ray_shapes_rect_vertical(self):
        """Test vertical rectangle cosmic ray interactions."""
        self.shape_test("rect_v", 6)
