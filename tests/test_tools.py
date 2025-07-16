import glob
import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from exosim.log import setLogLevel
from exosim.tools import (
    ADCGainEstimator,
    DarkCurrentMap,
    DeadPixelsMap,
    PixelsNonLinearity,
    PixelsNonLinearityFromCorrection,
    QuantumEfficiencyMap,
    ReadoutSchemeCalculator,
)
from exosim.tools.exosimTool import ExoSimTool
from exosim.utils import RunConfig

setLogLevel(logging.DEBUG)

class TestExoSimTools:

    def test_attr(self, tools_file):
        exotools = ExoSimTool(tools_file)
        assert hasattr(exotools, "options")


class TestReadoutSchemeCalculator:

    def test_read_out(self, test_data_dir, tools_file):
        f_list = glob.glob(os.path.join(test_data_dir, "test_data-*fp.h5"))
        if not f_list:
            pytest.skip("missing_file")
        ReadoutSchemeCalculator(tools_file, f_list[0])


@pytest.mark.usefixtures("test_data_dir")
class TestQuantumEfficiencyMap:

    @pytest.fixture(autouse=True)
    def _init(self, test_data_dir):
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
        self.params["channel"]["detector"]["qe_sigma"] = 0

        qe = QuantumEfficiencyMap(options_file=self.params, output=self.fname)
        q_map = qe.results["test"]

        constat = np.ones(
            (
                1,
                self.params["channel"]["detector"]["spatial_pix"],
                self.params["channel"]["detector"]["spectral_pix"],
            )
        )
        np.testing.assert_array_equal(q_map.data, constat)

        os.remove(self.fname)

    def test_no_output(self):
        self.params["channel"]["detector"]["qe_sigma"] = 0

        qe = QuantumEfficiencyMap(options_file=self.params)

    def test_size(self):
        qe = QuantumEfficiencyMap(options_file=self.params, output=self.fname)
        q_map = qe.results["test"]
        print(q_map.data.shape)

        assert q_map.data.shape == (1, 10, 20)
        os.remove(self.fname)

    def test_value(self):
        self.params["channel"]["detector"]["qe_sigma"] = 0.1

        qe = QuantumEfficiencyMap(options_file=self.params, output=self.fname)
        q_map = qe.results["test"]
        print(np.std(q_map.data[0]), np.mean(q_map.data[0]))
        np.testing.assert_almost_equal(np.std(q_map.data[0]), 0.1, decimal=1)
        np.testing.assert_almost_equal(np.mean(q_map.data[0]), 1, decimal=1)
        os.remove(self.fname)

    def test_time_variartion(self):
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

        print(
            np.std(q_map.data[0]),
            np.mean(q_map.data[0]),
            np.std(q_map.data[-1]),
            np.mean(q_map.data[-1]),
        )
        np.testing.assert_almost_equal(np.std(q_map.data[0]), 0.1, decimal=1)
        np.testing.assert_almost_equal(np.mean(q_map.data[0]), 1, decimal=1)
        np.testing.assert_almost_equal(
            np.mean(q_map.data[-1]), 1 - 0.05, decimal=1
        )
        os.remove(self.fname)


class TestDeadPixelMap:
    def test_constant_value(self):
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

        for x, y in dead_coords["spectral_coords", "spatial_coords"]:
            dead_pixels_map[y, x] = 0

    def test_output(self, test_data_dir):
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


class TestADCGain:
    def test_value(self):
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
        params = {
            "channel": {"value": "test", "detector": {"well_depth": 120000}}
        }
        adc = ADCGainEstimator(params)
        res = adc.results["test"]
        np.testing.assert_equal(res["integer dtype"], np.dtype("int32"))

    def test_errors(self):
        params = {
            "channel": {
                "value": "test",
                "detector": {"ADC_num_bit": "16", "ADC_max_value": 120000},
            }
        }
        with pytest.raises(TypeError):
            ADCGainEstimator(params)

        params = {
            "channel": {
                "value": "test",
                "detector": {"ADC_num_bit": 64, "ADC_max_value": 120000},
            }
        }
        with pytest.raises(ValueError):
            ADCGainEstimator(params)


class TestDarkCurrentMap:
    parameters = {
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

    times = np.arange(start=0.0, stop=1.0, step=0.25) * u.s

    def test_values(self):
        darkCurrentMap = DarkCurrentMap()
        dc_map = darkCurrentMap(parameters=self.parameters, time=self.times)

        # testing the map shape
        assert dc_map.data.shape == (
            self.times.size,
            self.parameters["detector"]["spatial_pix"]
            * self.parameters["detector"]["oversampling"],
            self.parameters["detector"]["spectral_pix"]
            * self.parameters["detector"]["oversampling"],
        )

        # testing the dictionary dc_mean key
        assert "dc_mean" in self.parameters["detector"]

        # testing the distribution
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



class TestPixelNonLinearity:
    params = {
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
        test_out = PixelsNonLinearity(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["coeff"]
        assert results is not None  # Verifica placeholder

    def test_map(self, skip_plot):
        test_out = PixelsNonLinearity(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["map"]
        coeff = test_out.results["test"]["coeff"]
        expected_std = self.params["channel"]["detector"]["pnl_coeff_std"]

        for i, map_ in enumerate(results):
            mean = np.mean(map_)
            std = np.std(map_) / np.abs(mean)
            np.testing.assert_allclose(mean, coeff[i], rtol=5 * 1e-03)
            np.testing.assert_allclose(std, expected_std, rtol=5 * 1e-02)

        # set std to 0 from input
        self.params["channel"]["detector"].pop("pnl_coeff_std")
        test_out = PixelsNonLinearity(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["map"]
        coeff = test_out.results["test"]["coeff"]

        for i, map_ in enumerate(results):
            np.testing.assert_allclose(map_, np.ones(map_.shape) * coeff[i])

class TestPixelNonLinearityFromCorrection:
    params = {
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
        test_out = PixelsNonLinearityFromCorrection(self.params, show_results=not skip_plot)
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
        test_out = PixelsNonLinearityFromCorrection(self.params, show_results=not skip_plot)
        results = test_out.results["test"]["coeff"]
        p = np.polynomial.Polynomial(results)

        Q = np.linspace(1, test_out.results["test"]["saturation"], 2**10)  # detector pixel counts in adu
        Q_det = Q * p(Q)

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

    def test_missingkeys(self, skip_plot):
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
