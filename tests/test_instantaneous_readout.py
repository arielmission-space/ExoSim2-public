import logging
import os

import astropy.units as u
import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from exosim.log import setLogLevel
from exosim.models.signal import CountsPerSecond
from exosim.output import SetOutput
from exosim.tasks.subexposures import (
    EstimateChJitter,
    InstantaneousReadOut,
    PrepareInstantaneousReadOut,
)

# setLogLevel(logging.DEBUG)

@pytest.fixture(autouse=True)
def inject_test_data_dir(request, test_data_dir):
    request.cls.test_data_dir = test_data_dir


@pytest.mark.usefixtures("inject_test_data_dir")
class TestInstantaneousReadOut:

    def setup_method(self):
        self.osf = 3
        self.npix = 64
        x, y = np.meshgrid(
            np.arange(-self.npix * self.osf // 2, self.npix * self.osf // 2, 1),
            np.arange(-self.npix * self.osf // 2, self.npix * self.osf // 2, 1),
        )
        FWHM = 3 * self.osf
        self.sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * self.sigma**2)))
        focal = g * 49 * u.ct / u.s
        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        focal_plane = CountsPerSecond(
            data=focal,
            spectral=np.arange(-self.npix * self.osf // 2, self.npix * self.osf // 2, 1),
            metadata={"oversampling": self.osf},
        )

        frg = np.ones_like(focal)
        frg_plane = CountsPerSecond(
            data=frg, spectral=np.arange(-self.npix * self.osf // 2, self.npix * self.osf // 2, 1)
        )

        self.main_parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            }
        }

        self.parameters = {
            "detector": {
                "well_depth": 1000 * u.ct,
                "f_well_depth": 1,
                "delta_pix": 18 * u.um,
                "oversampling": self.osf,
                "plate_scale": {
                    "spatial": 0.05 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
            },
            "readout": {
                "readout_frequency": 100 * u.Hz,
                "n_NRDs_per_group": 1,
                "n_groups": 2,
                "n_sim_clocks_Ground": 20,
                "n_sim_clocks_first_NDR": 20,
                "n_sim_clocks_NDR": 1,
                "n_sim_clocks_Reset": 10,
                "n_sim_clocks_groups": 1950,
                "n_exposures": 1,
            },
            "value": "test_channel",
        }

        self.int_time = (
            self.parameters["readout"]["n_sim_clocks_groups"]
            * 1
            / self.parameters["readout"]["readout_frequency"]
        ).to(u.s)

        self.jitter_time = np.arange(0, 25, 0.01) * u.s
        self.jitter_spa = np.random.normal(0, 0.05, self.jitter_time.size) * u.arcsec
        self.jitter_spe = np.random.normal(0, 0.05, self.jitter_time.size) * u.arcsec

        self.fname = os.path.join(self.test_data_dir, "test_data/output_test_jit.h5")
        output = SetOutput(self.fname)

        with output.use(append=True, cache=True) as out:
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            readout_parameters, self.integration_time = prepareInstantaneousReadOut(
                main_parameters=self.main_parameters,
                parameters=self.parameters,
                focal_plane=focal_plane,
                pointing_jitter=(self.jitter_spa, self.jitter_spe, self.jitter_time),
                output_file=out,
            )

            instantaneousReadOut = InstantaneousReadOut()
            self.dset = instantaneousReadOut(
                readout_parameters=readout_parameters,
                parameters=self.parameters,
                focal_plane=focal_plane,
                pointing_jitter=(self.jitter_spa, self.jitter_spe, self.jitter_time),
                output_file=out,
            )
        self.focal=focal



    def test_output_shape(self):
        assert (
            self.dset.shape[0]
            == self.parameters["readout"]["n_groups"]
            * self.parameters["readout"]["n_NRDs_per_group"]
            * self.parameters["readout"]["n_exposures"]
        )
        assert self.dset.shape[0] == self.integration_time.size
        assert self.dset.shape[1] == self.npix
        assert self.dset.shape[2] == self.npix

    def twoD_Gaussian(
        self, xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    ):
        (x, y) = xdata_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (
            np.sin(theta) ** 2
        ) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
            4 * sigma_y**2
        )
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (
            np.cos(theta) ** 2
        ) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(
            -(
                a * ((x - xo) ** 2)
                + 2 * b * (x - xo) * (y - yo)
                + c * ((y - yo) ** 2)
            )
        )
        return g.ravel()

    def test_value(self):
        try:
            f = h5py.File(self.fname, "r")
            se_out = f["SubExposures"]["data"][1]

        except FileNotFoundError:
            print("creating new dataset")
            with self.output.use(append=True, cache=True) as out:
                prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
                (
                    readout_parameters,
                    integration_time,
                ) = prepareInstantaneousReadOut(
                    main_parameters=self.main_parameters,
                    parameters=self.parameters,
                    focal_plane=self.focal_plane,
                    pointing_jitter=(
                        self.jitter_spa,
                        self.jitter_spe,
                        self.jitter_time,
                    ),
                    output_file=out,
                )

                instantaneousReadOut = InstantaneousReadOut()
                dset = instantaneousReadOut(
                    readout_parameters=readout_parameters,
                    parameters=self.parameters,
                    focal_plane=self.focal_plane,
                    pointing_jitter=(
                        self.jitter_spa,
                        self.jitter_spe,
                        self.jitter_time,
                    ),
                    output_file=out,
                )

                se_out = dset.dataset[1]

        # re-estimate the jitter
        estimateChJitter = EstimateChJitter()
        jitter_spe, jitter_spa, y_jit, x_jit, jit_time = estimateChJitter(
            pointing_jitter=(
                self.jitter_spa,
                self.jitter_spe,
                self.jitter_time,
            ),
            parameters=self.parameters,
        )
        y_jit = np.round(y_jit).astype(int)
        x_jit = np.round(x_jit).astype(int)

        # fit gaussian over jittered dataset
        initial_guess = (
            se_out.max(),
            se_out.shape[0] / 2,
            se_out.shape[1] / 2,
            (self.sigma + y_jit.mean()) / self.osf,
            (self.sigma + x_jit.mean()) / self.osf,
            0,
            0,
        )
        x = np.arange(0, se_out.shape[0])
        y = np.arange(0, se_out.shape[1])
        x, y = np.meshgrid(x, y)
        popt_dset, pcov = curve_fit(
            self.twoD_Gaussian, (x, y), se_out.flatten(), p0=initial_guess
        )

        # ideal convolution with gaussian kernel (which simulates the jitter)
        result = gaussian_filter(
            self.focal[0, self.osf // 2 :: self.osf, self.osf // 2 :: self.osf]
            * self.int_time.value,
            sigma=(np.std(y_jit / self.osf), np.std(x_jit / self.osf)),
        )

        # compare max vals
        tops = np.abs(se_out.max() - result.max()) / result.max()

        assert (tops < 0.1)

        # fit gaussian over ideal jittered dataset
        initial_guess = (
            result.max(),
            result.shape[0] / 2,
            result.shape[1] / 2,
            (self.sigma + y_jit.mean()) / self.osf,
            (self.sigma + x_jit.mean()) / self.osf,
            0,
            0,
        )

        popt_res, pcov = curve_fit(
            self.twoD_Gaussian, (x, y), result.flatten(), p0=initial_guess
        )

        # compare shapes
        popt_res[0] = 1
        popt_res[-1] = 1
        res_fitted = self.twoD_Gaussian((x, y), *popt_res)
        popt_dset[0] = 1
        popt_dset[-1] = 1
        data_fitted = self.twoD_Gaussian((x, y), *popt_dset)

        diff = (np.abs(res_fitted - data_fitted) / res_fitted).max()
        print(diff)
        assert(diff < 0.065)

@pytest.mark.usefixtures("fast_test")
class InstantaneousReadOutPowerConservationTest():
    @pytest.fixture(autouse=True)
    def _inject_fixture(self, fast_test):
        if fast_test:
            pytest.skip("Skipping this test class in fast mode")

    def test_nyquist_sampled(self, test_data_dir):
        osf = 4
        npix = 64
        x, y = np.meshgrid(
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
        )
        FWHM = 3 * osf
        sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        focal = g * u.ct / u.s
        focal /= focal.sum().value
        focal_sum = 100
        focal *= focal_sum

        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        focal_plane = CountsPerSecond(
            data=focal,
            spectral=np.arange(-npix * osf // 2, npix * osf // 2, 1),
            metadata={
                "oversampling": osf,
                "focal_plane_delta": 18 * u.um / osf,
            },
        )

        main_parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            }
        }

        parameters = {
            "detector": {
                "well_depth": 1000 * u.ct,
                "f_well_depth": 1,
                "delta_pix": 18 * u.um,
                "diffusion_length": 1.7 * u.um,
                "intra_pix_distance": 0.0 * u.um,
                "oversampling": osf,
                "plate_scale": {
                    "spatial": 0.05 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
            },
            "readout": {
                "readout_frequency": 100 * u.Hz,
                "n_NRDs_per_group": 1,
                "n_groups": 2,
                "n_sim_clocks_Ground": 1,
                "n_sim_clocks_first_NDR": 1,
                "n_sim_clocks_NDR": 1,
                "n_sim_clocks_Reset": 1,
                "n_sim_clocks_groups": 1000,
                "n_exposures": 3,
            },
            "value": "test_channel",
        }

        from exosim.tasks.instrument import (
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        )

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        parameters["psf_shape"] = [64, 64]
        kernel, delta_kernel = createIntrapixelResponseFunction(
            parameters=parameters
        )

        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane = applyIntraPixelResponseFunction(
            focal_plane=focal_plane,
            irf_kernel=kernel,
            irf_kernel_delta=delta_kernel,
        )

        readout = parameters["readout"]
        readout_frequency = readout["readout_frequency"]
        n_exposures = readout["n_exposures"]

        # Conta totale dei clock per esposizione
        n_clocks = (
            readout["n_sim_clocks_Ground"]
            + readout["n_sim_clocks_first_NDR"]
            + readout["n_sim_clocks_NDR"]
            + readout["n_sim_clocks_groups"]
            + readout["n_sim_clocks_Reset"]
        )

        # Tempo per esposizione
        t_per_exposure = n_clocks / readout_frequency  # u.s
        t_total = t_per_exposure * n_exposures         # u.s

        # Jitter time array
        jitter_dt = 0.01 * u.s
        jitter_time = np.arange(0, t_total.to_value(u.s) + jitter_dt.to_value(u.s), jitter_dt.to_value(u.s)) * u.s
        jitter_spa = np.random.normal(0, 0.5, jitter_time.size) * u.arcsec
        jitter_spe = np.random.normal(0, 0.5, jitter_time.size) * u.arcsec

        fname = os.path.join(test_data_dir, "test_data/output_test_jit_power_1.h5")
        output = SetOutput(fname)
        if os.path.isfile(fname):
            os.remove(fname)

        with output.use(append=True, cache=True) as out:
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            readout_parameters, integration_time = prepareInstantaneousReadOut(
                main_parameters=main_parameters,
                parameters=parameters,
                focal_plane=focal_plane,
                pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
                output_file=out,
            )

            instantaneousReadOut = InstantaneousReadOut()
            dset = instantaneousReadOut(
                readout_parameters=readout_parameters,
                parameters=parameters,
                focal_plane=focal_plane,
                pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
                output_file=out,
            )

            ndrs = dset.dataset[:]
            cds = dset.dataset[1::2]
        time_line = cds.sum(axis=-1).sum(axis=-1)
        time_line_ndrs = ndrs.sum(axis=-1).sum(axis=-1)

        for x in time_line:
            np.testing.assert_allclose(
            x, time_line[0], rtol=1e-14
        )  # constant value

        np.testing.assert_allclose(
            time_line_ndrs, (focal_sum * integration_time).value, rtol=1e-6
        )  # espected value

    def test_not_nyquist_sampled(self, test_data_dir):
        osf = 4
        npix = 64
        x, y = np.meshgrid(
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
        )
        FWHM = 1 * osf
        sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        focal = g * u.ct / u.s
        focal /= focal.sum().value
        focal_sum = 100
        focal *= focal_sum

        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        focal_plane = CountsPerSecond(
            data=focal,
            spectral=np.arange(-npix * osf // 2, npix * osf // 2, 1),
            metadata={
                "oversampling": osf,
                "focal_plane_delta": 18 * u.um / osf,
            },
        )

        main_parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            }
        }

        parameters = {
            "detector": {
                "well_depth": 1000 * u.ct,
                "f_well_depth": 1,
                "delta_pix": 18 * u.um,
                "diffusion_length": 1.7 * u.um,
                "intra_pix_distance": 0.0 * u.um,
                "oversampling": osf,
                "plate_scale": {
                    "spatial": 0.05 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
            },
            "readout": {
                "readout_frequency": 100 * u.Hz,
                "n_NRDs_per_group": 1,
                "n_groups": 2,
                "n_sim_clocks_Ground": 1,
                "n_sim_clocks_first_NDR": 1,
                "n_sim_clocks_NDR": 1,
                "n_sim_clocks_Reset": 1,
                "n_sim_clocks_groups": 1000,
                "n_exposures": 3,
            },
            "value": "test_channel",
        }

        from exosim.tasks.instrument import (
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        )

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        parameters["psf_shape"] = [64, 64]
        kernel, delta_kernel = createIntrapixelResponseFunction(
            parameters=parameters
        )

        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane = applyIntraPixelResponseFunction(
            focal_plane=focal_plane,
            irf_kernel=kernel,
            irf_kernel_delta=delta_kernel,
        )
        readout = parameters["readout"]
        readout_frequency = readout["readout_frequency"]
        n_exposures = readout["n_exposures"]

        # Conta totale dei clock per esposizione
        n_clocks = (
            readout["n_sim_clocks_Ground"]
            + readout["n_sim_clocks_first_NDR"]
            + readout["n_sim_clocks_NDR"]
            + readout["n_sim_clocks_groups"]
            + readout["n_sim_clocks_Reset"]
        )

        # Tempo per esposizione
        t_per_exposure = n_clocks / readout_frequency  # u.s
        t_total = t_per_exposure * n_exposures         # u.s

        # Jitter time array
        jitter_dt = 0.01 * u.s
        jitter_time = np.arange(0, t_total.to_value(u.s) + jitter_dt.to_value(u.s), jitter_dt.to_value(u.s)) * u.s
        jitter_spa = np.random.normal(0, 0.5, jitter_time.size) * u.arcsec
        jitter_spe = np.random.normal(0, 0.5, jitter_time.size) * u.arcsec

        fname = os.path.join(test_data_dir, "test_data/output_test_jit_power_2.h5")
        output = SetOutput(fname)
        if os.path.isfile(fname):
            os.remove(fname)

        with output.use(append=True, cache=True) as out:
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            readout_parameters, integration_time = prepareInstantaneousReadOut(
                main_parameters=main_parameters,
                parameters=parameters,
                focal_plane=focal_plane,
                pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
                output_file=out,
            )

            instantaneousReadOut = InstantaneousReadOut()
            dset = instantaneousReadOut(
                readout_parameters=readout_parameters,
                parameters=parameters,
                focal_plane=focal_plane,
                pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
                output_file=out,
            )

            cds = dset.dataset[1::2]
        time_line = cds.sum(axis=-1).sum(axis=-1)

        #        for x in time_line:
        #            np.testing.assert_allclose(x, time_line[0], rtol=1e-14) # constant

        with pytest.raises(AssertionError):
            for x in time_line:
                np.testing.assert_allclose(
                    x, time_line[0], rtol=1e-14
                )  # constant

    def test_not_nyquist_sampled_forced(self, test_data_dir):
        osf = 4
        npix = 64
        x, y = np.meshgrid(
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
        )
        FWHM = 1 * osf
        sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        focal = g * u.ct / u.s
        focal /= focal.sum().value
        focal_sum = 100
        focal *= focal_sum

        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        focal_plane = CountsPerSecond(
            data=focal,
            spectral=np.arange(-npix * osf // 2, npix * osf // 2, 1),
            metadata={
                "oversampling": osf,
                "focal_plane_delta": 18 * u.um / osf,
            },
        )

        main_parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            }
        }

        parameters = {
            "detector": {
                "well_depth": 1000 * u.ct,
                "f_well_depth": 1,
                "delta_pix": 18 * u.um,
                "diffusion_length": 1.7 * u.um,
                "intra_pix_distance": 0.0 * u.um,
                "oversampling": osf,
                "plate_scale": {
                    "spatial": 0.05 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
            },
            "readout": {
                "readout_frequency": 100 * u.Hz,
                "n_NRDs_per_group": 1,
                "n_groups": 2,
                "n_sim_clocks_Ground": 1,
                "n_sim_clocks_first_NDR": 1,
                "n_sim_clocks_NDR": 1,
                "n_sim_clocks_Reset": 1,
                "n_sim_clocks_groups": 1000,
                "n_exposures": 3,
            },
            "value": "test_channel",
        }

        from exosim.tasks.instrument import (
            ApplyIntraPixelResponseFunction,
            CreateIntrapixelResponseFunction,
        )

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        parameters["psf_shape"] = [64, 64]
        kernel, delta_kernel = createIntrapixelResponseFunction(
            parameters=parameters
        )

        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane = applyIntraPixelResponseFunction(
            focal_plane=focal_plane,
            irf_kernel=kernel,
            irf_kernel_delta=delta_kernel,
        )

        readout = parameters["readout"]
        readout_frequency = readout["readout_frequency"]
        n_exposures = readout["n_exposures"]

        # Conta totale dei clock per esposizione
        n_clocks = (
            readout["n_sim_clocks_Ground"]
            + readout["n_sim_clocks_first_NDR"]
            + readout["n_sim_clocks_NDR"]
            + readout["n_sim_clocks_groups"]
            + readout["n_sim_clocks_Reset"]
        )

        # Tempo per esposizione
        t_per_exposure = n_clocks / readout_frequency  # u.s
        t_total = t_per_exposure * n_exposures         # u.s

        # Jitter time array
        jitter_dt = 0.01 * u.s
        jitter_time = np.arange(0, t_total.to_value(u.s) + jitter_dt.to_value(u.s), jitter_dt.to_value(u.s)) * u.s
        jitter_spa = np.random.normal(0, 0.5, jitter_time.size) * u.arcsec
        jitter_spe = np.random.normal(0, 0.5, jitter_time.size) * u.arcsec

        fname = os.path.join(test_data_dir, "test_data/output_test_jit_power_3.h5")
        output = SetOutput(fname)
        if os.path.isfile(fname):
            os.remove(fname)

        parameters["force_power_conservation"] = True

        with output.use(append=True, cache=True) as out:
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            readout_parameters, integration_time = prepareInstantaneousReadOut(
                main_parameters=main_parameters,
                parameters=parameters,
                focal_plane=focal_plane,
                pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
                output_file=out,
            )

            instantaneousReadOut = InstantaneousReadOut()
            dset = instantaneousReadOut(
                readout_parameters=readout_parameters,
                parameters=parameters,
                focal_plane=focal_plane,
                pointing_jitter=(jitter_spa, jitter_spe, jitter_time),
                output_file=out,
            )

            ndrs = dset.dataset[:]
            cds = dset.dataset[1::2]
        time_line = cds.sum(axis=-1).sum(axis=-1)
        time_line_ndrs = ndrs.sum(axis=-1).sum(axis=-1)

        for x in time_line:
            np.testing.assert_allclose(x, time_line[0], rtol=1e-14)

        np.testing.assert_allclose(
            time_line_ndrs, (focal_sum * integration_time).value, rtol=1e-6
        )

@pytest.mark.usefixtures("inject_test_data_dir")
class InstantaneousReadOutResamplerTest():

    def setup_method(self):
        osf = 3
        self.npix = 64
        x, y = np.meshgrid(
            np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1),
            np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1),
        )
        FWHM = 4 * osf
        sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        focal = g * 49 * u.ct / u.s
        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        focal_plane = CountsPerSecond(
            data=focal,
            spectral=np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1),
            metadata={"oversampling": osf},
        )

        frg = np.ones_like(focal)
        frg_plane = CountsPerSecond(
            data=frg, spectral=np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1)
        )

        self.main_parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            }
        }

        self.parameters = {
            "detector": {
                "well_depth": 1000 * u.ct,
                "f_well_depth": 1,
                "delta_pix": 18 * u.um,
                "oversampling": osf,
                "plate_scale": {
                    "spatial": 0.05 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
            },
            "readout": {
                "readout_frequency": 100 * u.Hz,
                "n_NRDs_per_group": 1,
                "n_groups": 2,
                "n_sim_clocks_Ground": 20,
                "n_sim_clocks_first_NDR": 20,
                "n_sim_clocks_NDR": 1,
                "n_sim_clocks_Reset": 10,
                "n_sim_clocks_groups": 1950,
                "n_exposures": 1,
            },
            "value": "test_channel",
        }

        int_time = (
            self.parameters["readout"]["n_sim_clocks_groups"]
            * 1
            / self.parameters["readout"]["readout_frequency"]
        ).to(u.s)

        self.jitter_time = np.arange(0, 25, 0.01) * u.s
        self.jitter_spa = np.random.normal(0, 0.02, self.jitter_time.size) * u.arcsec
        self.jitter_spe = np.random.normal(0, 0.02, self.jitter_time.size) * u.arcsec

        fname = os.path.join(self.test_data_dir, "test_data/output_test_jit_resampled.h5")
        output = SetOutput(fname)

        with output.use(append=True, cache=True) as out:
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            readout_parameters, self.integration_time = prepareInstantaneousReadOut(
                main_parameters=self.main_parameters,
                parameters=self.parameters,
                focal_plane=focal_plane,
                pointing_jitter=(self.jitter_spa, self.jitter_spe, self.jitter_time),
                output_file=out,
            )

            instantaneousReadOut = InstantaneousReadOut()
            self.dset = instantaneousReadOut(
                readout_parameters=readout_parameters,
                parameters=self.parameters,
                focal_plane=focal_plane,
                pointing_jitter=(self.jitter_spa, self.jitter_spe, self.jitter_time),
                output_file=out,
            )

    def twoD_Gaussian(
        self, xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    ):
        (x, y) = xdata_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (
            np.sin(theta) ** 2
        ) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
            4 * sigma_y**2
        )
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (
            np.cos(theta) ** 2
        ) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(
            -(
                a * ((x - xo) ** 2)
                + 2 * b * (x - xo) * (y - yo)
                + c * ((y - yo) ** 2)
            )
        )
        return g.ravel()

    def test_value(self):
        try:
            f = h5py.File(self.fname, "r")
            se_out = f["SubExposures"]["data"][1]
            new_osf = f["instantaneous_readout_params"]["effective_osf"][()]

        except FileNotFoundError:
            print("creating new dataset")
            with self.output.use(append=True, cache=True) as out:
                prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
                (
                    readout_parameters,
                    integration_time,
                ) = prepareInstantaneousReadOut(
                    main_parameters=self.main_parameters,
                    parameters=self.parameters,
                    focal_plane=self.focal_plane,
                    pointing_jitter=(
                        self.jitter_spa,
                        self.jitter_spe,
                        self.jitter_time,
                    ),
                    output_file=out,
                )

                instantaneousReadOut = InstantaneousReadOut()
                dset = instantaneousReadOut(
                    readout_parameters=readout_parameters,
                    parameters=self.parameters,
                    focal_plane=self.focal_plane,
                    pointing_jitter=(
                        self.jitter_spa,
                        self.jitter_spe,
                        self.jitter_time,
                    ),
                    output_file=out,
                )

            f = h5py.File(self.fname, "r")
            se_out = f["SubExposures"]["data"][1]
            new_osf = f["instantaneous_readout_params"]["effective osf"][()]

        # re-estimate the jitter
        estimateChJitter = EstimateChJitter()
        jitter_spe, jitter_spa, y_jit, x_jit, jit_time = estimateChJitter(
            pointing_jitter=(
                self.jitter_spa,
                self.jitter_spe,
                self.jitter_time,
            ),
            parameters=self.parameters,
        )

        # fit gaussian over jittered dataset
        initial_guess = (
            se_out.max(),
            se_out.shape[0] / 2,
            se_out.shape[1] / 2,
            (self.sigma + y_jit.mean()) / self.osf,
            (self.sigma + x_jit.mean()) / self.osf,
            0,
            0,
        )
        x = np.arange(0, se_out.shape[0])
        y = np.arange(0, se_out.shape[1])
        x, y = np.meshgrid(x, y)
        popt_dset, pcov = curve_fit(
            self.twoD_Gaussian, (x, y), se_out.flatten(), p0=initial_guess
        )

        # estimate new osf
        jitter_rms = min(
            np.sqrt(np.mean(y_jit**2)), np.sqrt(np.mean(x_jit**2))
        )
        new_osf_ = self.osf * np.ceil(3.0 / jitter_rms.value).astype(int)
        assert new_osf_ == new_osf

        # estimate new jitter
        y_jit *= new_osf / self.osf
        x_jit *= new_osf / self.osf

        y_jit = np.round(y_jit).astype(int)
        x_jit = np.round(x_jit).astype(int)

        # estimate new focal plane
        osf = new_osf
        npix = 64
        x_, y_ = np.meshgrid(
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
            np.arange(-npix * osf // 2, npix * osf // 2, 1),
        )
        FWHM = 4 * osf
        sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x_ * x_ + y_ * y_)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        focal = g * 49 * u.ct / u.s
        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        # ideal convolution with gaussian kernel (which simulates the jitter)
        result = gaussian_filter(
            focal[0, new_osf // 2 :: new_osf, new_osf // 2 :: new_osf]
            * self.int_time.value,
            sigma=(np.std(y_jit / new_osf), np.std(x_jit / new_osf)),
        )

        # compare max vals
        tops = np.abs(se_out.max() - result.max()) / result.max()
        print(tops)
        assert(tops < 0.1)

        # fit gaussian over ideal jittered dataset
        initial_guess = (
            result.max(),
            result.shape[0] / 2,
            result.shape[1] / 2,
            (sigma + y_jit.mean()) / new_osf,
            (sigma + x_jit.mean()) / new_osf,
            0,
            0,
        )

        print(x.shape, y.shape, result.shape)
        popt_res, pcov = curve_fit(
            self.twoD_Gaussian, (x, y), result.flatten(), p0=initial_guess
        )

        # compare shapes
        popt_res[0] = 1
        popt_res[-1] = 1
        res_fitted = self.twoD_Gaussian((x, y), *popt_res)
        popt_dset[0] = 1
        popt_dset[-1] = 1
        data_fitted = self.twoD_Gaussian((x, y), *popt_dset)

        diff = (np.abs(res_fitted - data_fitted) / res_fitted).max()
        print(diff)

        assert(diff < 0.065)


@pytest.mark.usefixtures("inject_test_data_dir")
class InstantaneousReadOutNoJitterTest():

    def setup_method(self):

        osf = 3
        self.npix = 64
        x, y = np.meshgrid(
            np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1),
            np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1),
        )
        FWHM = 3 * osf
        sigma, mu = FWHM / 2.355, 0.0
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        focal = g * 49 * u.ct / u.s
        focal = focal[np.newaxis, ...]
        focal = np.repeat(focal, 1, axis=0)

        focal_plane = CountsPerSecond(
            data=focal,
            spectral=np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1),
            metadata={"oversampling": osf},
        )

        frg = np.ones_like(focal)
        frg_plane = CountsPerSecond(
            data=frg, spectral=np.arange(-self.npix * osf // 2, self.npix * osf // 2, 1)
        )

        self.main_parameters = {
            "time_grid": {
                "start_time": 0.0 * u.hr,
                "end_time": 2.0 * u.hr,
                "low_frequencies_resolution": 2.0 * u.hr,
            }
        }

        self.parameters = {
            "detector": {
                "well_depth": 1000 * u.ct,
                "f_well_depth": 1,
                "delta_pix": 18 * u.um,
                "oversampling": osf,
                "plate_scale": {
                    "spatial": 0.05 * u.arcsec / u.pixel,
                    "spectral": 0.05 * u.arcsec / u.pixel,
                },
            },
            "readout": {
                "readout_frequency": 100 * u.Hz,
                "n_NRDs_per_group": 1,
                "n_groups": 2,
                "n_sim_clocks_Ground": 20,
                "n_sim_clocks_first_NDR": 20,
                "n_sim_clocks_NDR": 1,
                "n_sim_clocks_Reset": 10,
                "n_sim_clocks_groups": 1950,
                "n_exposures": 1,
            },
            "value": "test_channel",
        }

        int_time = (
            self.parameters["readout"]["n_sim_clocks_groups"]
            * 1
            / self.parameters["readout"]["readout_frequency"]
        ).to(u.s)

        fname = os.path.join(self.test_data_dir, "test_data/output_test_no_jit.h5")
        output = SetOutput(fname)

        with output.use(append=True, cache=True) as out:
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            readout_parameters, self.integration_time = prepareInstantaneousReadOut(
                main_parameters=self.main_parameters,
                parameters=self.parameters,
                focal_plane=focal_plane,
                pointing_jitter=(None, None, None),
                output_file=out,
            )

            instantaneousReadOut = InstantaneousReadOut()
            self.dset = instantaneousReadOut(
                readout_parameters=readout_parameters,
                parameters=self.parameters,
                focal_plane=focal_plane,
                pointing_jitter=(None, None, None),
                output_file=out,
            )

    def test_out(self):
        assert(
            self.dset.shape[0]
            == self.parameters["readout"]["n_groups"]
            * self.parameters["readout"]["n_NRDs_per_group"]
            * self.parameters["readout"]["n_exposures"]
        )
        assert(self.dset.shape[0] == self.integration_time.size)
        assert(self.dset.shape[1] == self.npix)
        assert(self.dset.shape[2] == self.npix)

    def twoD_Gaussian(
        self, xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    ):
        (x, y) = xdata_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (
            np.sin(theta) ** 2
        ) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
            4 * sigma_y**2
        )
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (
            np.cos(theta) ** 2
        ) / (2 * sigma_y**2)
        g = offset + amplitude * np.exp(
            -(
                a * ((x - xo) ** 2)
                + 2 * b * (x - xo) * (y - yo)
                + c * ((y - yo) ** 2)
            )
        )
        return g.ravel()

    def test_value(self):
        try:
            f = h5py.File(self.fname, "r")
            se_out = f["SubExposures"]["data"][1]

        except FileNotFoundError:
            print("creating new dataset")
            with self.output.use(append=True, cache=True) as out:
                prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
                (
                    readout_parameters,
                    integration_time,
                ) = prepareInstantaneousReadOut(
                    main_parameters=self.main_parameters,
                    parameters=self.parameters,
                    focal_plane=self.focal_plane,
                    pointing_jitter=(None, None, None),
                    output_file=out,
                )

                instantaneousReadOut = InstantaneousReadOut()
                dset = instantaneousReadOut(
                    readout_parameters=readout_parameters,
                    parameters=self.parameters,
                    focal_plane=self.focal_plane,
                    pointing_jitter=(None, None, None),
                    output_file=out,
                )
                se_out = dset.dataset[1]

        # fit gaussian over jittered dataset
        initial_guess = (
            se_out.max(),
            se_out.shape[0] / 2,
            se_out.shape[1] / 2,
            (self.sigma) / self.osf,
            (self.sigma) / self.osf,
            0,
            0,
        )
        x = np.arange(0, se_out.shape[0])
        y = np.arange(0, se_out.shape[1])
        x, y = np.meshgrid(x, y)
        popt_dset, pcov = curve_fit(
            self.twoD_Gaussian, (x, y), se_out.flatten(), p0=initial_guess
        )

        # ideal convolution with gaussian kernel (which simulates the jitter)
        result = (
            self.focal[0, 0 :: self.osf, 0 :: self.osf].value
            * self.int_time.value
        )

        # compare max vals
        tops = np.abs(se_out.max() - result.max()) / result.max()
        print(tops)
        assert(tops < 0.1)

        # fit gaussian over ideal jittered dataset
        initial_guess = (
            result.max(),
            result.shape[0] / 2,
            result.shape[1] / 2,
            (self.sigma) / self.osf,
            (self.sigma) / self.osf,
            0,
            0,
        )

        popt_res, pcov = curve_fit(
            self.twoD_Gaussian, (x, y), result.flatten(), p0=initial_guess
        )

        # compare shapes
        popt_res[0] = 1
        popt_res[-1] = 1
        res_fitted = self.twoD_Gaussian((x, y), *popt_res)
        popt_dset[0] = 1
        popt_dset[-1] = 1
        data_fitted = self.twoD_Gaussian((x, y), *popt_dset)

        diff = (np.abs(res_fitted - data_fitted) / res_fitted).max()
        print(diff)

        assert(diff < 0.05)
