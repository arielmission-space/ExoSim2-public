import logging
import os
import unittest
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from inputs import regression_dir
from inputs import skip_plot
from inputs import test_dir
from tqdm.auto import tqdm

from exosim.log import setLogLevel
from exosim.models.signal import Counts
from exosim.output import SetOutput
from exosim.tasks.detector import AccumulateSubExposures
from exosim.tasks.detector import AddConstantDarkCurrent
from exosim.tasks.detector import AddCosmicRays
from exosim.tasks.detector import AddDarkCurrentMapNumpy
from exosim.tasks.detector import AddGainDrift
from exosim.tasks.detector import AddKTC
from exosim.tasks.detector import AddNormalReadNoise
from exosim.tasks.detector import AddReadNoiseMapNumpy
from exosim.tasks.detector import AddShotNoise
from exosim.tasks.detector import AnalogToDigital
from exosim.tasks.detector import ApplyDeadPixelMapNumpy
from exosim.tasks.detector import ApplyDeadPixelsMap
from exosim.tasks.detector import ApplyPixelsNonLinearity
from exosim.tasks.detector import ApplySimpleSaturation
from exosim.tasks.detector import LoadPixelsNonLinearityMap
from exosim.tasks.detector import MergeGroups

setLogLevel(logging.DEBUG)


class AddConstantDarkCurrentTest(unittest.TestCase):
    def test_values(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.zeros((10, 10, 10))

        integration_times = np.ones(data.shape[0]) * u.s

        parameters = {"detector": {"dc_mean": 5 * u.ct / u.s}}
        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input,
                parameters=parameters,
                integration_times=integration_times,
            )

            np.testing.assert_array_equal(
                input.dataset[0], np.ones((10, 10), dtype=np.float64) * 5
            )

        os.remove(fname)


class DarkCurrentMapTest(unittest.TestCase):
    def test_value(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.zeros((10, 10, 10))
        integration_times = np.ones(data.shape[0]) * u.s

        dc = np.ones(data[0].shape) * 5
        np.save(os.path.join(test_dir, "dc_map.npy"), dc)

        parameters = {
            "detector": {
                "dc_map_filename": os.path.join(test_dir, "dc_map.npy")
            }
        }
        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input,
                parameters=parameters,
                integration_times=integration_times,
            )

            np.testing.assert_array_equal(
                input.dataset[0], np.ones((10, 10), dtype=np.float64) * 5
            )

        os.remove(fname)

    def test_err(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 100, 100)) * 100
        integration_times = np.ones(data.shape[0]) * u.s

        dc = np.ones((10, 20)) * 5
        np.save(os.path.join(test_dir, "dc_map.npy"), dc)

        parameters = {
            "detector": {
                "dc_map_filename": os.path.join(test_dir, "dc_map.npy")
            }
        }
        with output.use(cache=True) as out:
            input = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )
            with self.assertRaises(IOError):
                addDarkCurrentMapNumpy = AddDarkCurrentMapNumpy()
                addDarkCurrentMapNumpy(
                    subexposures=input,
                    parameters=parameters,
                    integration_times=integration_times,
                )

        os.remove(fname)


class LoadPixelsNonLinearityMapTest(unittest.TestCase):
    filename = os.path.join(regression_dir, "data/payload/pnl_map.h5")

    parameters = {
        "value": "Photometer",
        "detector": {
            "pnl_map_task": LoadPixelsNonLinearityMap,
            "pnl_filename": filename,
        },
    }

    pnlMap = LoadPixelsNonLinearityMap()

    def test_load(self):
        pnl_map = self.pnlMap(parameters=self.parameters)

    @unittest.skipIf(skip_plot, "This test only produces plots")
    def test_plots(self):
        import matplotlib.pyplot as plt

        pnl_map = self.pnlMap(parameters=self.parameters)
        map = pnl_map["map"]

        Q = np.linspace(1, pnl_map["saturation"] * 1.2, 2**10)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle("detector non-linearity map loaded")
        ax.axvline(
            pnl_map["saturation"],
            c="k",
            label="real saturation (5% from linear): {}".format(
                int(np.ceil(pnl_map["saturation"]))
            ),
            ls="-",
        )

        coeffs = map.T.reshape(map.shape[1] * map.shape[2], map.shape[0])
        for cs in tqdm(coeffs, total=coeffs.shape[0], desc="preparing plot"):
            p = np.polynomial.Polynomial(cs)
            plt.plot(Q, Q * p(Q), c="g", lw=0.5, alpha=0.1)

        ax.plot(Q, Q, "k", ls=":", label="Linear pixel count")

        ax.set_xlabel("$Q$ [adu]")
        ax.set_ylabel("$Q_{det}$ [adu]")
        ax.legend()
        ax.grid()
        plt.show()


class ApplyPixelsNonLinearityMapTest(unittest.TestCase):
    def test_value(self):
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

        fname = os.path.join(test_dir, "output_test_npl.h5")
        output = SetOutput(fname)
        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input, parameters=pnl_dict.results["test"]
            )

            np.testing.assert_allclose(
                input.dataset[0], np.ones((10, 10)) * 10000 * 0.95, rtol=1e-3
            )
            os.remove(fname)


class AddShotNoiseTest(unittest.TestCase):
    def test_values(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((10, 10, 100)) * 10000

        with output.use(cache=True) as out:
            input = Counts(
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
            addShotNoise(
                subexposures=input,
            )

            np.testing.assert_allclose(
                np.mean(input.dataset), 10000, rtol=0.001
            )
            np.testing.assert_allclose(np.std(input.dataset), 100, rtol=0.1)

        os.remove(fname)


class AccumulateTest(unittest.TestCase):
    def test_values(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((4, 10, 100)).astype(np.float64)

        state_machine = np.array([0, 0, 1, 1]).astype(int)

        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input, state_machine=state_machine
            )

            test_data = deepcopy(data)
            test_data[1] += test_data[0]
            test_data[3] += test_data[2]

            np.testing.assert_array_equal(input.dataset[0:4], test_data)

        os.remove(fname)


class AddKTCTest(unittest.TestCase):
    def test_bias(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.zeros((1000, 100, 100))
        state_machine = np.arange(0, 250).astype(int)
        state_machine = np.repeat(state_machine, 4)

        parameters = {"detector": {"ktc_sigma": 1 * u.ct}}
        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input,
                state_machine=state_machine,
                parameters=parameters,
            )

            # testing the distribution
            np.testing.assert_allclose(np.median(input.dataset), 0, atol=0.5)
            np.testing.assert_allclose(np.std(input.dataset), 1, atol=0.2)

            # testing the random values in the same frame
            np.testing.assert_allclose(
                np.median(input.dataset[0]), 0, atol=0.1
            )
            np.testing.assert_allclose(np.std(input.dataset[0]), 1, atol=0.1)

            # testing equals on the same ramp
            np.testing.assert_equal(input.dataset[0], input.dataset[1])
            np.testing.assert_equal(input.dataset[4], input.dataset[5])

            # testing different on the different ramps
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                input.dataset[0],
                input.dataset[5],
            )

        os.remove(fname)


class DeadPixelTest(unittest.TestCase):
    def test_map(self):
        tab = Table()
        tab["spatial_coords"] = [0, 2, 4]
        tab["spectral_coords"] = [1, 3, 5]

        map_fname = os.path.join(test_dir, "dp_test.h5")

        ascii.write(
            tab, map_fname, format="ecsv", overwrite=True, delimiter=","
        )

        parameters = {"detector": {"dp_map": map_fname}}

        data = np.ones((10, 10, 10)) * 10

        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        with output.use(cache=True) as out:
            input = Counts(
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
            applyDeadPixelsMap(subexposures=input, parameters=parameters)

            self.assertEqual(input.dataset[0, 0, 1], 0.0)
            self.assertEqual(input.dataset[0, 2, 3], 0.0)
            self.assertEqual(input.dataset[0, 4, 5], 0.0)
            self.assertEqual(input.dataset[0, 0, 3], 10)
            self.assertEqual(input.dataset[0, 2, 1], 10)

        os.remove(fname)
        os.remove(map_fname)


class GainDriftTest(unittest.TestCase):
    def test_value(self):
        fname = os.path.join(test_dir, "output_test_gain.h5")
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
            input = Counts(
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
            addGainDrift(subexposures=input, parameters=parameters)

            # testing the range
            range = (np.max(input.dataset) - np.min(input.dataset)) / np.min(
                input.dataset
            )
            np.testing.assert_almost_equal(range, 1e-2, decimal=4)

        os.remove(fname)


class DeadPixelNumpyTest(unittest.TestCase):
    def test_map(self):
        test_array = np.zeros((10, 10))
        test_array[0, 1] = 1
        test_array[2, 3] = 1
        test_array[4, 5] = 1

        map_fname = os.path.join(test_dir, "dp_test.npy")
        np.save(map_fname, test_array)

        parameters = {"detector": {"dp_map_filename": map_fname}}

        data = np.ones((10, 10, 10)) * 10

        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        with output.use(cache=True) as out:
            input = Counts(
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
            applyDeadPixelMapNumpy(subexposures=input, parameters=parameters)

            self.assertEqual(input.dataset[0, 0, 1], 0.0)
            self.assertEqual(input.dataset[0, 2, 3], 0.0)
            self.assertEqual(input.dataset[0, 4, 5], 0.0)
            self.assertEqual(input.dataset[0, 0, 3], 10)
            self.assertEqual(input.dataset[0, 2, 1], 10)

        os.remove(fname)
        os.remove(map_fname)


class ReadNoiseTest(unittest.TestCase):
    def test_value(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 100, 100)) * 100

        parameters = {"detector": {"read_noise_sigma": 1 * u.ct}}
        with output.use(cache=True) as out:
            input = Counts(
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
            addNormalReadNoise(subexposures=input, parameters=parameters)

            # testing the distribution
            np.testing.assert_allclose(np.median(input.dataset), 100, atol=0.5)
            np.testing.assert_allclose(np.std(input.dataset), 1, atol=0.1)

            # testing the random values in the same frame
            np.testing.assert_allclose(
                np.median(input.dataset[0]), 100, atol=0.5
            )
            np.testing.assert_allclose(np.std(input.dataset[0]), 1, atol=0.1)

            # testing different on the different rams
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                input.dataset[0],
                input.dataset[1],
            )

        os.remove(fname)


class ReadNoiseMapTest(unittest.TestCase):
    def test_value(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 64, 64)) * 100

        read = np.ones(data[0].shape)
        np.save(os.path.join(test_dir, "read_map.npy"), read)

        parameters = {
            "detector": {
                "read_noise_filename": os.path.join(test_dir, "read_map.npy")
            }
        }
        with output.use(cache=True) as out:
            input = Counts(
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
            addReadNoiseMapNumpy(subexposures=input, parameters=parameters)

            # testing the distribution
            np.testing.assert_allclose(np.median(input.dataset), 100, atol=0.5)
            np.testing.assert_allclose(np.std(input.dataset), 1, atol=0.1)

            # testing the random values in the same frame
            np.testing.assert_allclose(
                np.median(input.dataset[0]), 100, atol=0.5
            )
            np.testing.assert_allclose(np.std(input.dataset[0]), 1, atol=0.1)

            # testing different on the different rams
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                input.dataset[0],
                input.dataset[1],
            )

        os.remove(fname)

    def test_err(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1000, 100, 100)) * 100

        read = np.ones((20, 10))
        np.save(os.path.join(test_dir, "read_map.npy"), read)

        parameters = {
            "detector": {
                "read_noise_filename": os.path.join(test_dir, "read_map.npy")
            }
        }
        with output.use(cache=True) as out:
            input = Counts(
                spectral=np.arange(0, 100),
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )
            with self.assertRaises(IOError):
                addReadNoiseMapNumpy = AddReadNoiseMapNumpy()
                addReadNoiseMapNumpy(subexposures=input, parameters=parameters)

        os.remove(fname)


class SimpleSaturationTest(unittest.TestCase):
    def test_sat(self):
        parameters = {"detector": {"well_depth": 1000}}

        data = np.ones((10, 10, 10)) * 10
        data[0, 0, 0] = 1001
        data[1, 1, 1] = 1100
        data[1, 1, 2] = 999
        data[1, 1, 3] = 1000

        fname = os.path.join(test_dir, "output_test_sat.h5")
        output = SetOutput(fname)

        with output.use(cache=True) as out:
            input = Counts(
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
            applySimpleSaturation(subexposures=input, parameters=parameters)

            self.assertEqual(input.dataset[0, 0, 0], 1000.0)
            self.assertEqual(input.dataset[1, 1, 1], 1000.0)
            self.assertEqual(input.dataset[1, 1, 3], 1000.0)
            self.assertEqual(input.dataset[1, 1, 2], 999)

        os.remove(fname)


class AnalogToDigitalTest(unittest.TestCase):
    def produce_ndrs(self, dtype, nbits):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1, 10, 10)).astype(np.float64)

        parameters = {"detector": {"ADC_num_bit": nbits, "ADC_gain": 0.5}}

        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input, output=input.output, parameters=parameters
            )

            test_data = (
                np.ones((10, 10)) * parameters["detector"]["ADC_gain"]
            ).astype(dtype)

            np.testing.assert_array_equal(ndrs.dataset[0], test_data)

        os.remove(fname)

    def test_values_32(self):
        self.produce_ndrs(np.int32, 32)

    def test_values_16(self):
        self.produce_ndrs(np.int16, 16)

    def test_values_8(self):
        self.produce_ndrs(np.int8, 8)

    def test_no_values(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((1, 10, 10)).astype(np.float64)

        parameters = {"detector": {"ADC_gain": 0.5}}

        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input, output=input.output, parameters=parameters
            )

            test_data = (
                np.ones((10, 10)) * parameters["detector"]["ADC_gain"]
            ).astype("int32")

            np.testing.assert_array_equal(ndrs.dataset[0], test_data)

        os.remove(fname)

    def test_values_float(self):
        self.produce_ndrs(np.int8, 16.0)

    def test_values_weird(self):
        self.produce_ndrs(np.int16, 12)

    def test_values_wrong_float(self):
        with self.assertRaises(TypeError):
            self.produce_ndrs(np.int16, 12.5)

    def test_values_too_big(self):
        with self.assertRaises(ValueError):
            self.produce_ndrs(np.int32, 34)


class MergeTest(unittest.TestCase):
    def test_values(self):
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)

        data = np.ones((6, 10, 100)).astype(np.float64)
        data[1] += 1
        data[2] += 2
        data[4] += 1
        data[5] += 2

        n_groups = 2
        n_ndrs = 3

        with output.use(cache=True) as out:
            input = Counts(
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
            output = mergeGroups(
                subexposures=input,
                n_groups=n_groups,
                n_ndrs=n_ndrs,
                output=out,
            )

            test_data = np.ones((2, 10, 100)).astype(np.float64) * 2

            np.testing.assert_array_equal(output.dataset[0:2], test_data)

        os.remove(fname)


class CosmicRaysTest(unittest.TestCase):
    def test_interactions_counts(self):
        fname = os.path.join(test_dir, "output_test_cr.h5")
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
        integration_times[1] = 2

        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input,
                parameters=parameters,
                integration_times=integration_times,
            )

            # import matplotlib.pyplot as plt

            # plt.imshow(input.dataset[0])
            # plt.show()
            for t in range(input.dataset.shape[0]):
                n_sat = np.where(input.dataset[t] == 10000)[0]
                if t == 1:
                    self.assertEqual(
                        len(n_sat),
                        2,
                    )
                else:
                    self.assertEqual(
                        len(n_sat),
                        1,
                    )
        os.remove(fname)

    def shape_test(self, shape, n_pix):
        fname = os.path.join(test_dir, "output_test_cr.h5")
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
                "interaction_shapes": {shape: 1},
            },
        }
        integration_times = np.ones(10)

        with output.use(cache=True) as out:
            input = Counts(
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
                subexposures=input,
                parameters=parameters,
                integration_times=integration_times,
            )

            for t in range(input.dataset.shape[0]):
                n_sat = np.where(input.dataset[t] == 10000)[0]
                self.assertLessEqual(len(n_sat), n_pix)
        os.remove(fname)

    def test_shapes_single(self):
        self.shape_test("single", 1)

    def test_shapes_line_h(self):
        self.shape_test("line_h", 2)

    def test_shapes_line_v(self):
        self.shape_test("line_v", 2)

    def test_shapes_quad(self):
        self.shape_test("quad", 4)

    def test_shapes_cross(self):
        self.shape_test("cross", 5)

    def test_shapes_rect_h(self):
        self.shape_test("rect_h", 6)

    def test_shapes_rect_v(self):
        self.shape_test("rect_v", 6)
