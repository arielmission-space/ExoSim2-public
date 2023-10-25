import logging
import os
import unittest

import astropy.units as u
import numpy as np
from inputs import regression_dir
from inputs import skip_plot
from inputs import test_dir

from exosim.log import setLogLevel
from exosim.models.signal import Counts
from exosim.models.signal import CountsPerSecond
from exosim.models.signal import Dimensionless
from exosim.output import SetOutput
from exosim.tasks.subexposures import AddForegrounds
from exosim.tasks.subexposures import ApplyQeMap
from exosim.tasks.subexposures import ComputeReadingScheme
from exosim.tasks.subexposures import EstimateChJitter
from exosim.tasks.subexposures import EstimatePointingJitter
from exosim.tasks.subexposures import LoadQeMap

setLogLevel(logging.DEBUG)


class PointingJitterTest(unittest.TestCase):
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
    jitter_spa, jitter_spe, jitter_time = estimatePointingJitter(
        parameters=parameters,
    )

    # plt.plot(jitter_spe, label='spectral', alpha=0.75)
    # plt.plot(jitter_spa, label='spatial', alpha=0.75)
    # plt.xlabel('time steps')
    # plt.ylabel('amplitude [deg]')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig('random_jitter.png')
    #
    # plt.clf()
    # n, bins, patches = plt.hist(jitter_spe.value, 50, density=True, alpha=0.75, label='spectral')
    # n, bins, patches = plt.hist(jitter_spa.value, 50, density=True, alpha=0.75, label='spatial')
    # plt.xlabel('amplitude [deg]')
    # plt.ylabel('Number of occurrence')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('random_histo_jitter.png')

    def test_units(self):
        self.assertEqual(self.jitter_spa.unit, "deg")
        self.assertEqual(self.jitter_spe.unit, "deg")

    def test_std(self):
        np.testing.assert_allclose(
            self.jitter_spa.to(u.arcsec).std().value,
            self.parameters["jitter"]["spatial"].to(u.arcsec).value,
            atol=0.002,
        )
        np.testing.assert_allclose(
            self.jitter_spe.to(u.arcsec).std().value,
            self.parameters["jitter"]["spectral"].to(u.arcsec).value,
            atol=0.002,
        )

    def test_average(self):
        np.testing.assert_allclose(
            np.median(self.jitter_spa.to(u.arcsec)).value, 0.0, atol=0.002
        )
        np.testing.assert_allclose(
            np.median(self.jitter_spe.to(u.arcsec)).value, 0.0, atol=0.002
        )

    def test_time(self):
        step = self.jitter_time[1] - self.jitter_time[0]
        self.assertEqual(
            step, self.parameters["jitter"]["frequency_resolution"]
        )


class ChJitterTest(unittest.TestCase):
    jitter_time = np.arange(0, 60, 0.005) * u.s
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

    def test_std(self):
        estimateChJitter = EstimateChJitter()
        jitter_spe, jitter_spa, jit_y, jit_x, new_jit_time = estimateChJitter(
            parameters=self.parameters,
            pointing_jitter=(
                self.jitter_spa,
                self.jitter_spe,
                self.jitter_time,
            ),
        )
        np.testing.assert_allclose(np.std(jit_y), 1, atol=0.1)
        np.testing.assert_allclose(np.std(jit_x), 1, atol=0.1)
        np.testing.assert_allclose(
            np.std(jitter_spe), 0.05 * u.arcsec, atol=0.01
        )
        np.testing.assert_allclose(
            np.std(jitter_spa), 0.01 * u.arcsec, atol=0.01
        )


class ComputeReadingSchemeTest(unittest.TestCase):
    osf = 3
    npix = 64
    x, y = np.meshgrid(
        np.arange(-npix * osf // 2, npix * osf // 2, 1),
        np.arange(-npix * osf // 2, npix * osf // 2, 1),
    )
    FWHM = 3 * osf
    sigma, mu = FWHM / 2.355, 0.0
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
    focal = g * 49 * u.ct / u.s
    focal = focal[np.newaxis, ...]
    focal = np.repeat(focal, 1, axis=0)

    frg = np.ones_like(focal)

    main_parameters = {"time_grid": {"end_time": 40 * u.s}}
    parameters = {
        "readout": {
            "readout_frequency": 100 * u.Hz,
            "n_NRDs_per_group": 1,
            "n_groups": 2,
            "n_sim_clocks_Ground": 20,
            "n_sim_clocks_NDR": 1,
            "n_sim_clocks_first_NDR": 20,
            "n_sim_clocks_Reset": 10,
            "n_sim_clocks_groups": 1950,
        }
    }

    def test_values(self):
        self.parameters["readout"]["readout_frequency"] = 100 * u.Hz
        self.parameters["readout"].pop("n_exposures", None)
        computeReadingScheme = ComputeReadingScheme()
        (
            clock,
            base_mask,
            frame_sequence,
            number_of_exposures,
        ) = computeReadingScheme(
            parameters=self.parameters,
            main_parameters=self.main_parameters,
            readout_oversampling=1,
        )
        self.assertTrue(clock == 0.01 * u.s)
        np.testing.assert_array_equal(base_mask, [0, 1, 1, 0])
        np.testing.assert_array_equal(frame_sequence, [20, 20, 1950, 10] * 2)
        self.assertTrue(number_of_exposures == 2)

    def test_oversampling(self):
        self.parameters["readout"]["readout_frequency"] = 100 * u.Hz
        self.parameters["readout"].pop("n_exposures", None)
        computeReadingScheme = ComputeReadingScheme()
        (
            clock,
            base_mask,
            frame_sequence,
            number_of_exposures,
        ) = computeReadingScheme(
            parameters=self.parameters,
            main_parameters=self.main_parameters,
            readout_oversampling=2,
        )
        self.assertTrue(clock == 0.005 * u.s)
        np.testing.assert_array_equal(base_mask, [0, 1, 1, 0])
        np.testing.assert_array_equal(frame_sequence, [40, 40, 3900, 20] * 2)
        self.assertTrue(number_of_exposures == 2)

    def test_values_freq_in_sec(self):
        self.parameters["readout"]["readout_frequency"] = 0.01 * u.s
        self.parameters["readout"].pop("n_exposures", None)

        computeReadingScheme = ComputeReadingScheme()
        (
            clock,
            base_mask,
            frame_sequence,
            number_of_exposures,
        ) = computeReadingScheme(
            parameters=self.parameters,
            main_parameters=self.main_parameters,
            readout_oversampling=1,
        )
        self.assertTrue(clock == 0.01 * u.s)
        np.testing.assert_array_equal(base_mask, [0, 1, 1, 0])
        np.testing.assert_array_equal(frame_sequence, [20, 20, 1950, 10] * 2)
        self.assertTrue(number_of_exposures == 2)

    def test_exposure(self):
        self.parameters["readout"]["n_exposures"] = 1

        computeReadingScheme = ComputeReadingScheme()
        (
            clock,
            base_mask,
            frame_sequence,
            number_of_exposures,
        ) = computeReadingScheme(
            parameters=self.parameters,
            main_parameters=self.main_parameters,
            readout_oversampling=1,
        )
        self.assertTrue(clock == 0.01 * u.s)
        np.testing.assert_array_equal(base_mask, [0, 1, 1, 0])
        np.testing.assert_array_equal(frame_sequence, [20, 20, 1950, 10] * 1)
        self.assertTrue(number_of_exposures == 1)


class AddForegroundTest(unittest.TestCase):
    def test_value(self):
        frg = CountsPerSecond(
            spectral=np.arange(0, 10),
            data=np.ones((10, 10)) * 2,
            metadata={"oversampling": 1},
        )
        integration_time = np.ones(10) * u.s
        data = np.ones((10, 10, 10))

        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)
        with output.use(cache=True) as out:
            input = Counts(
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

            addForegrounds = AddForegrounds()
            input = addForegrounds(
                subexposures=input,
                frg_focal_plane=frg,
                integration_time=integration_time,
            )

            np.testing.assert_array_equal(
                input.dataset[0], np.ones((10, 10), dtype=np.float64) * 3
            )

            self.assertEqual(np.sum(input.dataset), 3 * 10 * 10 * 10)
            os.remove(fname)

    def test_value_rebin(self):
        frg = CountsPerSecond(
            spectral=np.arange(0, 30),
            data=np.ones((30, 30)),
            metadata={"oversampling": 3},
        )
        integration_time = np.ones(10) * u.s
        data = np.ones((10, 10, 10))

        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)
        with output.use(cache=True) as out:
            input = Counts(
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

            addForegrounds = AddForegrounds()
            input = addForegrounds(
                subexposures=input,
                frg_focal_plane=frg,
                integration_time=integration_time,
            )

            np.testing.assert_array_equal(
                input.dataset[0], np.ones((10, 10), dtype=np.float64) + 3 * 3
            )

            os.remove(fname)


class LoadQeMapTest(unittest.TestCase):
    filename = os.path.join(regression_dir, "data/payload/qe_map.h5")

    parameters = {
        "value": "Photometer",
        "detector": {"qe_map_task": LoadQeMap, "qe_map_filename": filename},
    }

    loadQeMap = LoadQeMap()

    def test_single_time_step(self):
        time = [0] * u.hr
        qe_map = self.loadQeMap(parameters=self.parameters, time=time)

    def test_more_time_steps(self):
        time = [0, 2, 3, 5, 9] * u.hr
        qe_map = self.loadQeMap(parameters=self.parameters, time=time)

    @unittest.skipIf(skip_plot, "This test only produces plots")
    def test_plots(self):
        import matplotlib.pyplot as plt

        time = [0] * u.hr

        parameters = {
            "value": "Photometer",
            "detector": {
                "qe_map_task": LoadQeMap,
                "qe_map_filename": self.filename,
            },
        }
        qe_map = self.loadQeMap(parameters=parameters, time=time)
        plt.imshow(qe_map.data)

        parameters = {
            "value": "Spectrometer",
            "detector": {
                "qe_map_task": LoadQeMap,
                "qe_map_filename": self.filename,
            },
        }
        qe_map = self.loadQeMap(parameters=parameters, time=time)
        plt.imshow(qe_map.data)


class ApplyQeMapTest(unittest.TestCase):
    def test_value(self):
        qe_map = Dimensionless(
            spectral=np.arange(0, 10), data=np.ones((1, 10, 10)) * 2
        )

        data = np.ones((10, 10, 10))
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)
        with output.use(cache=True) as out:
            input = Counts(
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

            applyQeMap = ApplyQeMap()
            input = applyQeMap(subexposures=input, qe_map=qe_map)

            np.testing.assert_array_equal(
                input.dataset[0], np.ones((10, 10), dtype=np.float64) * 2
            )

            self.assertEqual(np.sum(input.dataset), 2 * 10 * 10 * 10)
            os.remove(fname)

    def test_weird_value(self):
        qe_data = np.ones((20000, 10, 10))
        qe_data[0] *= 2
        qe_data[2] *= np.reshape(np.arange(0, 100), (10, 10))
        qe_map = Dimensionless(
            spectral=np.arange(0, 10),
            data=qe_data,
            time=np.linspace(0, 10000, 20000) * u.hr,
        )

        data = np.ones((10000, 10, 10))
        fname = os.path.join(test_dir, "output_test.h5")
        output = SetOutput(fname)
        with output.use(cache=True) as out:
            input = Counts(
                spectral=np.arange(0, 10),
                time=np.arange(0, 10000) * u.hr,
                data=data,
                shape=data.shape,
                cached=True,
                output=out,
                dataset_name="SubExposures",
                output_path=None,
                dtype=np.float64,
            )

            applyQeMap = ApplyQeMap()
            input = applyQeMap(subexposures=input, qe_map=qe_map)

            np.testing.assert_array_equal(
                input.dataset[0], np.ones((10, 10), dtype=np.float64) * 2
            )
            np.testing.assert_array_equal(
                input.dataset[1],
                np.reshape(np.arange(0, 100, dtype=np.float64), (10, 10)),
            )

            np.testing.assert_array_equal(
                input.dataset[3], np.ones((10, 10), dtype=np.float64)
            )
            np.testing.assert_array_equal(
                input.dataset[-1], np.ones((10, 10), dtype=np.float64)
            )
            os.remove(fname)
