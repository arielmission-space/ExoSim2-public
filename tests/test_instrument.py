import itertools
import logging
import os
import unittest

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.table import QTable
from inputs import payload_file
from scipy.signal import fftconvolve
from test_sources import exolib_bb_model

import exosim.utils as utils
from exosim.log import setLogLevel
from exosim.models.signal import CountsPerSecond
from exosim.models.signal import Dimensionless
from exosim.models.signal import Signal
from exosim.tasks.instrument.applyIntraPixelResponseFunction import (
    ApplyIntraPixelResponseFunction,
)
from exosim.tasks.instrument.computeSaturation import ComputeSaturation
from exosim.tasks.instrument.computeSolidAngle import ComputeSolidAngle
from exosim.tasks.instrument.computeSourcesPointingOffset import (
    ComputeSourcesPointingOffset,
)
from exosim.tasks.instrument.createFocalPlane import CreateFocalPlane
from exosim.tasks.instrument.createFocalPlaneArray import CreateFocalPlaneArray
from exosim.tasks.instrument.createIntrapixelResponseFunction import (
    CreateIntrapixelResponseFunction,
)
from exosim.tasks.instrument.createOversampledIntrapixelResponseFunction import (
    CreateOversampledIntrapixelResponseFunction,
)
from exosim.tasks.instrument.foregroundsToFocalPlane import (
    ForegroundsToFocalPlane,
)
from exosim.tasks.instrument.loadResponsivity import LoadResponsivity
from exosim.tasks.instrument.loadWavelengthSolution import (
    LoadWavelengthSolution,
)
from exosim.tasks.instrument.populateFocalPlane import PopulateFocalPlane
from exosim.tasks.instrument.propagateForegrounds import PropagateForegrounds
from exosim.tasks.instrument.propagateSources import PropagateSources
from exosim.tasks.load.loadOptions import LoadOptions
from exosim.tasks.parse import ParsePath
from exosim.tasks.sed import CreatePlanckStar
from exosim.utils.psf import create_psf

setLogLevel(logging.DEBUG)


class FalseLoadResponsivity(LoadResponsivity):
    def model(self, parameters, wavelength, time):
        return 0.0


class FalseLoadResponsivity2(LoadResponsivity):
    def model(self, parameters, wavelength, time):
        return Signal(spectral=wavelength, data=np.ones_like(wavelength))


class ResponsivityTest(unittest.TestCase):
    loadOption = LoadOptions()
    mainConfig = loadOption(filename=payload_file)

    wl = utils.grids.wl_grid(
        mainConfig["wl_grid"]["wl_min"],
        9 * u.um,
        mainConfig["wl_grid"]["logbin_resolution"],
    )

    tt = utils.grids.time_grid(
        mainConfig["time_grid"]["start_time"],
        mainConfig["time_grid"]["end_time"],
        mainConfig["time_grid"]["low_frequencies_resolution"],
    )
    paylaod = mainConfig["payload"]

    def test_responsivity(self):
        loadResponsivity = LoadResponsivity()
        resp = loadResponsivity(
            parameters=self.paylaod["channel"]["Photometer"],
            wavelength=self.wl,
            time=self.tt,
        )

        rest_test = np.ones(len(self.wl)) * u.Unit("")
        rest_test *= 0.7 * self.wl.to(u.m) / const.c / const.h * u.count
        rest_test[self.wl >= 4.45 * u.um] = 0

        np.testing.assert_array_equal(resp.data[0, 0], rest_test.value)

    def test_responsivity_error(self):
        with self.assertRaises(TypeError) as contex:
            falseLoadResponsivity = FalseLoadResponsivity()
            resp = falseLoadResponsivity(
                parameters=self.paylaod["channel"]["Photometer"],
                wavelength=self.wl,
                time=self.tt,
            )

        with self.assertRaises(u.UnitConversionError) as contex:
            falseLoadResponsivity = FalseLoadResponsivity2()
            resp = falseLoadResponsivity(
                parameters=self.paylaod["channel"]["Photometer"],
                wavelength=self.wl,
                time=self.tt,
            )


class SolidAngleTest(unittest.TestCase):
    def test_omegapix(self):
        channel = {"Fnum_x": 15, "detector": {"delta_pix": 18 * u.um}}

        computeSolidAngle = ComputeSolidAngle()
        solid_angle = computeSolidAngle(parameters=channel)

        solid_angle_test = computeSolidAngle._omega_pix(15) * (18 * u.um) ** 2
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )

    def test_omegapix_diff_fnum(self):
        channel = {
            "Fnum_x": 15,
            "Fnum_y": 15,
            "detector": {"delta_pix": 18 * u.um},
        }

        computeSolidAngle = ComputeSolidAngle()
        solid_angle = computeSolidAngle(parameters=channel)

        solid_angle_test = computeSolidAngle._omega_pix(15) * (18 * u.um) ** 2
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )

        solid_angle_test = (
            computeSolidAngle._omega_pix(15, 15) * (18 * u.um) ** 2
        )
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )

        channel = {
            "Fnum_x": 15,
            "Fnum_y": 10,
            "detector": {"delta_pix": 18 * u.um},
        }

        computeSolidAngle = ComputeSolidAngle()
        solid_angle = computeSolidAngle(parameters=channel)

        solid_angle_test = (
            computeSolidAngle._omega_pix(15, 10) * (18 * u.um) ** 2
        )
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )

    def test_pi(self):
        channel = {"Fnum_x": 15, "detector": {"delta_pix": 18 * u.um}}
        other_par = {"solid_angle": "pi"}

        computeSolidAngle = ComputeSolidAngle()
        solid_angle = computeSolidAngle(
            parameters=channel, other_parameters=other_par
        )

        solid_angle_test = np.pi * u.sr * (18 * u.um) ** 2
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )

    def test_pi_omega(self):
        channel = {"Fnum_x": 15, "detector": {"delta_pix": 18 * u.um}}
        other_par = {"solid_angle": "pi-omega_pix"}

        computeSolidAngle = ComputeSolidAngle()
        solid_angle = computeSolidAngle(
            parameters=channel, other_parameters=other_par
        )

        solid_angle_test = (
            np.pi * u.sr - computeSolidAngle._omega_pix(15)
        ) * (18 * u.um) ** 2
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )

    def test_custom(self):
        channel = {"Fnum_x": 15, "detector": {"delta_pix": 18 * u.um}}
        other_par = {"solid_angle": 10 * u.sr}

        computeSolidAngle = ComputeSolidAngle()
        solid_angle = computeSolidAngle(
            parameters=channel, other_parameters=other_par
        )

        solid_angle_test = 10 * u.sr * (18 * u.um) ** 2
        self.assertAlmostEqual(
            solid_angle.value, solid_angle_test.to(u.sr * u.m**2).value
        )


class PropagateSourceTest(unittest.TestCase):
    loadOption = LoadOptions()
    mainConfig = loadOption(filename=payload_file)

    wl = utils.grids.wl_grid(0.5 * u.um, 9 * u.um, 1000)

    def test_propagation(self):
        createPlanckStar = CreatePlanckStar()
        T = 5778 * u.K
        R = 1 * u.R_sun
        D = 1 * u.au
        sed = createPlanckStar(wavelength=self.wl, T=T, R=R, D=D)

        omega_star = np.pi * (R.si / D.si) ** 2 * u.sr
        sed_exolib = omega_star * exolib_bb_model(self.wl, T)

        eff = Dimensionless(spectral=self.wl, data=np.ones(len(self.wl)) * 0.8)

        res = Signal(
            spectral=self.wl,
            data=np.ones(len(self.wl))
            * 0.8
            * self.wl.to(u.m)
            / const.c
            / const.h
            * u.count,
        )

        Atel = 1 * u.m**2
        sig = sed_exolib * eff * res * Atel

        propagateSources = PropagateSources()
        sources = propagateSources(
            sources={"test star": sed},
            Atel=Atel,
            efficiency=eff,
            responsivity=res,
        )

        self.assertEqual(sources["test star"].data_units, u.ct / u.s / u.um)

        np.testing.assert_array_almost_equal(
            sources["test star"].data / sig.data,
            np.ones((1, 1, len(self.wl))),
            decimal=5,
        )


class LoadWavelengthSolutionWorking(LoadWavelengthSolution):
    def model(self, parameters):
        tab = QTable()
        tab["wavelength"] = np.linspace(1, 10, 10) * u.um
        tab["spectral"] = np.linspace(1, 10, 10) * u.um
        tab["spatial"] = np.linspace(1, 5, 10) * u.um
        if "wl_solution" in parameters.keys():
            if "spatial" in parameters["wl_solution"].keys():
                if not parameters["wl_solution"]["spatial"]:
                    tab["spatial"] = np.zeros_like(tab["spectral"])
        return tab


class LoadWLTest(unittest.TestCase):
    class LoadWavelengthSolutionFloat(LoadWavelengthSolution):
        def model(self, parameters):
            return 0.0

    def test_wrong_format(self):
        with self.assertRaises(TypeError):
            loadWavelengthSolutionFloat = self.LoadWavelengthSolutionFloat()
            testval = loadWavelengthSolutionFloat(parameters={})

    class LoadWavelengthSolutionNoKey(LoadWavelengthSolution):
        def model(self, parameters):
            tab = QTable()
            tab["wavelength"] = [0.0] * u.um
            tab["spectral"] = [0.0] * u.um
            return tab

    def test_missing_k(self):
        with self.assertRaises(KeyError):
            loadWavelengthSolutionNoKey = self.LoadWavelengthSolutionNoKey()
            testval = loadWavelengthSolutionNoKey(parameters={})

    class LoadWavelengthSolutionUnit(LoadWavelengthSolution):
        def model(self, parameters):
            tab = QTable()
            tab["wavelength"] = [0.0] * u.um
            tab["spectral"] = [0.0] * u.um
            tab["spatial"] = [0.0] * u.kg
            return tab

    def test_wrong_unit(self):
        with self.assertRaises(u.UnitsError):
            loadWavelengthSolutionUnit = self.LoadWavelengthSolutionUnit()
            testval = loadWavelengthSolutionUnit(parameters={})

    def test_valid(self):
        loadWavelengthSolutionWorking = LoadWavelengthSolutionWorking()
        testval = loadWavelengthSolutionWorking(parameters={})

    def test_default(self):
        loadWavelengthSolution = LoadWavelengthSolution()
        wl_data = QTable()
        wl_data["Wavelength"] = np.arange(1, 10, 1) * u.um
        wl_data["x"] = np.arange(-5, 4, 1) * u.um
        wl_data["y"] = np.arange(1, 10, 1) * u.um
        parameters = {"wl_solution": {"data": wl_data}}
        wl_sol = loadWavelengthSolution(parameters=parameters)


class CreateFocalPlaneTest(unittest.TestCase):
    parameters_spec = {
        "detector": {
            "spatial_pix": 10,
            "spectral_pix": 10,
            "oversampling": 1,
            "delta_pix": 1 * u.um,
        },
        "type": "spectrometer",
        "wl_solution": {"wl_solution_task": LoadWavelengthSolutionWorking},
    }
    parameters_phot = {
        "detector": {
            "spatial_pix": 10,
            "spectral_pix": 10,
            "oversampling": 1,
            "delta_pix": 1 * u.um,
        },
        "type": "photometer",
    }
    phot_efficiency = Dimensionless(
        spectral=np.linspace(1, 10, 5) * u.um, data=np.ones(5) * 0.8
    )

    def test_spectrometer_array(self):
        # easy run
        createFocalPlaneArray = CreateFocalPlaneArray()
        focal = createFocalPlaneArray(
            parameters=self.parameters_spec, efficiency=0
        )
        print(focal.spectral)
        print(focal.spatial)
        np.testing.assert_array_equal(focal.data, np.zeros((1, 10, 10)))

        # no oversampling no spatial
        self.parameters_spec["detector"].pop("oversampling")
        self.parameters_spec["wl_solution"]["spatial"] = False
        createFocalPlaneArray = CreateFocalPlaneArray()
        focal = createFocalPlaneArray(
            parameters=self.parameters_spec, efficiency=0
        )
        print(focal.spectral)
        print(focal.spatial)
        np.testing.assert_array_equal(focal.data, np.zeros((1, 10, 10)))

        # manual offset
        self.parameters_spec["wl_solution"]["spectral_center"] = 2 * u.um
        self.parameters_spec["wl_solution"]["spatial_center"] = 2 * u.um
        focal = createFocalPlaneArray(
            parameters=self.parameters_spec, efficiency=0
        )
        print(focal.spectral)
        print(focal.spatial)
        np.testing.assert_array_equal(focal.data, np.zeros((1, 10, 10)))

        # auto offset
        self.parameters_spec["wl_solution"].pop("spectral_center")
        self.parameters_spec["wl_solution"].pop("spatial_center")

        self.parameters_spec["wl_solution"]["center"] = "auto"
        self.parameters_spec["wl_min"] = 0.1 * u.um
        self.parameters_spec["wl_max"] = 10 * u.um
        focal = createFocalPlaneArray(
            parameters=self.parameters_spec, efficiency=0
        )
        print(focal.spectral)
        print(focal.spatial)
        np.testing.assert_array_equal(focal.data, np.zeros((1, 10, 10)))
        #

    def test_spectrometer_plane(self):
        # easy run
        createFocalPlane = CreateFocalPlane()
        focal = createFocalPlane(
            parameters=self.parameters_spec,
            efficiency=0,
            time=np.linspace(1, 10, 10) * u.hr,
            output=None,
        )

        np.testing.assert_array_equal(focal.data, np.zeros((10, 10, 10)))

    def test_photometer_array(self):
        # easy run
        createFocalPlaneArray = CreateFocalPlaneArray()
        focal = createFocalPlaneArray(
            parameters=self.parameters_phot, efficiency=self.phot_efficiency
        )
        print(focal.spectral)
        print(focal.spatial)
        np.testing.assert_array_equal(focal.data, np.zeros((1, 10, 10)))

        #

    def test_photometer_plane(self):
        # easy run
        createFocalPlane = CreateFocalPlane()
        focal = createFocalPlane(
            parameters=self.parameters_phot,
            efficiency=self.phot_efficiency,
            time=np.linspace(1, 10, 10) * u.hr,
            output=None,
        )
        print(focal.spectral)
        print(focal.spatial)
        np.testing.assert_array_equal(focal.data, np.zeros((10, 10, 10)))


class ForegroundToFocalPlaneTest(unittest.TestCase):
    parameters_spec = {
        "detector": {
            "spatial_pix": 10,
            "spectral_pix": 100,
            "oversampling": 1,
            "delta_pix": 15 * u.um,
        },
        "type": "spectrometer",
        "Fnum_x": 5,
        "wl_solution": {"wl_solution_task": LoadWavelengthSolutionWorking},
    }

    wl = np.linspace(1, 100, 100) * u.um
    tt = np.linspace(1, 10, 10) * u.hr

    createFocalPlane = CreateFocalPlane()
    focal = createFocalPlane(
        parameters=parameters_spec, efficiency=0, time=tt, output=None
    )

    loadOption = LoadOptions()
    mainConfig = loadOption(filename=payload_file)

    parsePath = ParsePath()
    path_ch1 = parsePath(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ],
        wavelength=wl,
        time=tt,
    )

    path_ch2 = parsePath(
        parameters=mainConfig["payload"]["channel"]["Spectrometer"][
            "optical_path"
        ],
        wavelength=wl,
        time=tt,
    )

    def test_no_slit(self):
        foregroundsToFocalPlane = ForegroundsToFocalPlane()
        fore_1, _ = foregroundsToFocalPlane(
            parameters=self.parameters_spec,
            focal_plane=self.focal,
            path=self.path_ch1,
        )
        self.assertEqual(fore_1.data.shape, (10, 10, 100))
        np.testing.assert_array_equal(
            fore_1.data, np.ones(fore_1.data.shape) * fore_1.data[0, 0, 0]
        )

    def test_slit(self):
        foregroundsToFocalPlane = ForegroundsToFocalPlane()
        fore_2, _ = foregroundsToFocalPlane(
            parameters=self.parameters_spec,
            focal_plane=self.focal,
            path=self.path_ch2,
        )
        self.assertEqual(fore_2.data.shape, (10, 10, 100))
        #
        # plt.plot(fore_2.data[0].sum(axis=0))
        # plt.show()

    def test_propagate(self):
        propagateForegrounds = PropagateForegrounds()
        propagateForegrounds(
            light_path=self.path_ch1,
            responsivity=Signal(
                spectral=self.wl,
                data=np.ones(100) * u.m / const.c / const.h * u.count,
            ),
            parameters=self.parameters_spec,
        )


class IntrapixelResponseFunctionTest(unittest.TestCase):
    parameters = {
        "detector": {
            "delta_pix": 18 * u.um,
            "oversampling": 4,
            "diffusion_length": 1.7 * u.m,
            "intra_pix_distance": 0 * u.um,
        },
        "wmax": 1.95 * u.um,
        "wmin": 0.9 * u.um,
    }

    def intrapixel_response_function_slim(self, osf, delta, lx, ipd):
        lx += 1e-20 * u.um  # to avoid problems if user pass lx=0
        lx = lx.to(delta.unit)

        kernel = np.zeros((osf, osf))
        print("kernel size: {}".format(kernel.shape))
        # prepare the kernel stamp grid
        kernel_delta = delta / osf
        scale = np.arange(0, 2 * osf / 2) - osf / 2 + 0.5
        yy = scale * kernel_delta
        xx = scale * kernel_delta

        xx, yy = np.meshgrid(xx, yy)

        # compute kernel stamp
        kernel_stamp = np.arctan(
            np.tanh((0.5 * (0.5 * delta - xx) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - xx) / lx).value))

        kernel_stamp *= np.arctan(
            np.tanh((0.5 * (0.5 * delta - yy) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - yy) / lx).value))

        # deal with border
        i_mask_xx = np.where(np.abs(xx) > 0.5 * (delta - ipd))
        i_mask_yy = np.where(np.abs(yy) > 0.5 * (delta - ipd))
        # set the unused area of kernel stamp to zero
        kernel_stamp[i_mask_yy] = 0.0
        kernel_stamp[i_mask_xx] = 0.0

        # Normalise the kernel such that the pixel has QE=1
        kernel_stamp /= kernel_stamp.sum()
        kernel_stamp *= osf * osf

        return kernel_stamp, kernel_delta

    def create_focal_planes(self, shape, wl, fnum, delta_im):
        # large roi
        nzero = 8
        psf = create_psf(wl, fnum, delta_im, nzero, shape)
        im_large = psf[0]
        im_large /= im_large.sum()

        # small roi
        nzero = 4
        psf = create_psf(wl, fnum, delta_im, nzero, shape)
        im_small = psf[0]
        im_small /= im_small.sum()

        # ## zero pad the image to get the right size: power of 2
        size = 2 ** int(np.ceil(np.log2(im_large.shape[0])))
        im_large = self.zero_padding(im_large, size)
        im_small = self.zero_padding(im_small, size)

        print(im_small.shape, im_small.shape)
        return im_large, im_small

    def zero_padding(self, im, n):
        missing_border = int((n - im.shape[0]) // 2)
        odd = int((n - im.shape[0]) % 2)
        return np.pad(
            im,
            [
                (missing_border, missing_border + odd),
                (missing_border, missing_border + odd),
            ],
            mode="constant",
        )

    def test_create_iprf(self):
        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        kernel_e, kernel_delta_e = self.intrapixel_response_function_slim(
            self.parameters["detector"]["oversampling"],
            self.parameters["detector"]["delta_pix"],
            self.parameters["detector"]["diffusion_length"],
            self.parameters["detector"]["intra_pix_distance"],
        )
        np.testing.assert_array_almost_equal(kernel, kernel_e)
        np.testing.assert_array_almost_equal(
            kernel_delta.value, kernel_delta_e.value
        )

    def test_missing_val(self):
        self.parameters["detector"].pop("oversampling")
        self.parameters["detector"].pop("diffusion_length")
        self.parameters["detector"].pop("intra_pix_distance")

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        kernel_e, kernel_delta_e = self.intrapixel_response_function_slim(
            1,
            self.parameters["detector"]["delta_pix"],
            0 * u.um,
            0 * u.um,
        )

        np.testing.assert_array_almost_equal(kernel, kernel_e)
        np.testing.assert_array_almost_equal(
            kernel_delta.value, kernel_delta_e.value
        )

        self.parameters["detector"]["oversampling"] = 4
        self.parameters["detector"]["diffusion_length"] = (1.7 * u.m,)
        self.parameters["detector"]["intra_pix_distance"] = (0 * u.um,)

    def test_convolve_with_flat(self):
        """This test assures that for a flat IRF for both Gaussian PSF the power is conserved
        for large and small ROI up to 1e-4 ppm and for and Airy PSF up to 1 ppm level
        """

        detector = {
            "wmax": 1.95 * u.um,
            "wmin": 0.9 * u.um,
            "delta_pix": 18.0 * u.um,
            "diffusion_length": 0.0 * u.um,
            "intra_pix_distance": 0.0 * u.um,
            "oversampling": 16,
        }
        parameters = {"detector": detector}

        fnum = 45
        delta_im = detector["delta_pix"] / detector["oversampling"]

        wl = np.linspace(detector["wmin"], detector["wmax"], 10)

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=parameters
        )

        #### GAUSSIAN PSF
        im_large_gauss, im_small_gauss = self.create_focal_planes(
            "gauss", wl, fnum, delta_im
        )

        conv_image = fftconvolve(im_large_gauss, in2=kernel, mode="same")
        conv_image_small = fftconvolve(im_small_gauss, in2=kernel, mode="same")
        # testing total power
        np.testing.assert_almost_equal(
            np.sum(conv_image), detector["oversampling"] ** 2
        )
        np.testing.assert_almost_equal(
            np.sum(conv_image_small), detector["oversampling"] ** 2
        )

        # big image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        self.assertTrue(np.std(power) < 1e-10)

        # small image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image_small[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        self.assertTrue(np.std(power) < 1e-10)

        #### AIRY PSF
        im_large_airy, im_small_airy = self.create_focal_planes(
            "airy", wl, fnum, delta_im
        )

        conv_image = fftconvolve(im_large_airy, in2=kernel, mode="same")
        conv_image_small = fftconvolve(im_small_airy, in2=kernel, mode="same")

        # testing total power
        np.testing.assert_almost_equal(
            np.sum(conv_image), detector["oversampling"] ** 2
        )
        np.testing.assert_almost_equal(
            np.sum(conv_image_small), detector["oversampling"] ** 2
        )

        # big image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        self.assertTrue(np.std(power) < 1e-6)

        # small image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image_small[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        self.assertTrue(np.std(power) < 1e-5)

    def test_gauss_power(self):
        """This test assures that in case of an Gauss function the power is conserved
        for large and small ROI up to 1e-3 ppm."""

        detector = {
            "wmax": 1.95 * u.um,
            "wmin": 0.9 * u.um,
            "delta_pix": 18.0 * u.um,
            "diffusion_length": 1.7 * u.um,
            "intra_pix_distance": 0.0 * u.um,
            "oversampling": 16,
        }
        parameters = {"detector": detector}
        fnum = 45
        delta_im = detector["delta_pix"] / detector["oversampling"]

        wl = np.linspace(detector["wmin"], detector["wmax"], 10)
        im_large_gauss, im_small_gauss = self.create_focal_planes(
            "gauss", wl, fnum, delta_im
        )

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=parameters
        )

        conv_image = fftconvolve(im_large_gauss, in2=kernel, mode="same")
        conv_image_small = fftconvolve(im_small_gauss, in2=kernel, mode="same")

        # testing total power
        np.testing.assert_almost_equal(
            np.sum(conv_image), detector["oversampling"] ** 2
        )
        np.testing.assert_almost_equal(
            np.sum(conv_image_small), detector["oversampling"] ** 2
        )

        # big image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        sts = np.std(power)
        self.assertTrue(np.std(power) < 1e-8)

        # small image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image_small[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        sts = np.std(power)
        self.assertTrue(np.std(power) < 1e-8)

    def test_airy_power(self):
        """This test assures that in case of an Airy function the power is conserved
        up to 10ppm for a large ROI and up to 100 ppm for small ROI."""

        detector = {
            "wmax": 1.95 * u.um,
            "wmin": 0.9 * u.um,
            "delta_pix": 18.0 * u.um,
            "diffusion_length": 1.7 * u.um,
            "intra_pix_distance": 0.0 * u.um,
            "oversampling": 16,
        }
        parameters = {"detector": detector}

        fnum = 45
        delta_im = detector["delta_pix"] / detector["oversampling"]

        wl = np.linspace(detector["wmin"], detector["wmax"], 10)
        im_large_airy, im_small_airy = self.create_focal_planes(
            "airy", wl, fnum, delta_im
        )

        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=parameters
        )

        conv_image = fftconvolve(im_large_airy, in2=kernel, mode="same")
        conv_image_small = fftconvolve(im_small_airy, in2=kernel, mode="same")

        # testing total power
        np.testing.assert_almost_equal(
            np.sum(conv_image), detector["oversampling"] ** 2
        )
        np.testing.assert_almost_equal(
            np.sum(conv_image_small), detector["oversampling"] ** 2
        )

        # big image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)

        self.assertTrue(np.std(power) < 1e-4)

        # small image
        power = []
        for i in range(detector["oversampling"]):
            p = np.sum(
                conv_image_small[
                    i :: detector["oversampling"],
                    i :: detector["oversampling"],
                ]
            )
            power.append(p)
        power = np.array(power)
        self.assertTrue(np.std(power) < 1e-3)


class IntrapixelResponseFunctionOversampledTest(unittest.TestCase):
    def pixelResponseFunction(
        self, psf_shape, osf, delta, lx=0.0 * u.um, ipd=0.0 * u.um
    ):
        # from Exosim1: https://github.com/ExoSim/ExoSim/blob/master/exosim/lib/exolib.py

        lx += 1e-8 * u.um  # to avoid problems if user pass lx=0
        lx = lx.to(delta.unit)

        kernel = np.zeros((psf_shape[0] * osf, psf_shape[1] * osf))
        kernel_delta = delta / osf
        yc, xc = np.array(kernel.shape) // 2
        yy = (np.arange(kernel.shape[0]) - yc) * kernel_delta
        xx = (np.arange(kernel.shape[1]) - xc) * kernel_delta
        mask_xx = np.where(np.abs(xx) > 0.5 * (delta - ipd))
        mask_yy = np.where(np.abs(yy) > 0.5 * (delta - ipd))
        xx, yy = np.meshgrid(xx, yy)

        kernel = np.arctan(
            np.tanh((0.5 * (0.5 * delta - xx) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - xx) / lx).value))

        kernel *= np.arctan(
            np.tanh((0.5 * (0.5 * delta - yy) / lx).value)
        ) - np.arctan(np.tanh((0.5 * (-0.5 * delta - yy) / lx).value))

        kernel[mask_yy, ...] = 0.0
        kernel[..., mask_xx] = 0.0

        # Normalise the kernel such that the pixel has QE=1
        kernel *= osf**2 / kernel.sum()
        kernel = np.roll(kernel, -xc, axis=1)
        kernel = np.roll(kernel, -yc, axis=0)

        return kernel, kernel_delta

    parameters = {
        "detector": {
            "delta_pix": 18 * u.um,
            "oversampling": 3,
            "diffusion_length": 1.7 * u.m,
            "intra_pix_distance": 0 * u.um,
        },
        "psf_shape": (10, 10),
    }

    def test_full(self):
        createIntrapixelResponseFunction = (
            CreateOversampledIntrapixelResponseFunction()
        )
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        kernel_e, kernel_delta_e = self.pixelResponseFunction(
            self.parameters["psf_shape"],
            8 * self.parameters["detector"]["oversampling"],
            self.parameters["detector"]["delta_pix"],
            self.parameters["detector"]["diffusion_length"],
            self.parameters["detector"]["intra_pix_distance"],
        )
        np.testing.assert_array_almost_equal(kernel, kernel_e)
        np.testing.assert_array_almost_equal(
            kernel_delta.value, kernel_delta_e.value
        )

    def test_missing_val(self):
        self.parameters["detector"].pop("oversampling")
        self.parameters["detector"].pop("diffusion_length")
        self.parameters["detector"].pop("intra_pix_distance")

        createIntrapixelResponseFunction = (
            CreateOversampledIntrapixelResponseFunction()
        )
        kernel, kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        kernel_e, kernel_delta_e = self.pixelResponseFunction(
            self.parameters["psf_shape"],
            7,
            self.parameters["detector"]["delta_pix"],
        )
        np.testing.assert_array_almost_equal(kernel, kernel_e)
        np.testing.assert_array_almost_equal(
            kernel_delta.value, kernel_delta_e.value
        )


class ApplyIntrapixelResponseFunctionTest(unittest.TestCase):
    parameters = {
        "detector": {
            "delta_pix": 18 * u.um,
            "oversampling": 4,
            "diffusion_length": 1.7 * u.m,
            "intra_pix_distance": 0 * u.um,
        },
        "psf_shape": (10, 10),
    }
    focal_plane = CountsPerSecond(
        spectral=np.linspace(1, 10, 10) * u.um,
        data=np.ones((10, 10, 10)),
        metadata={"focal_plane_delta": 18 * u.um},
    )

    def test_apply_default(self):
        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        irf_kernel, irf_kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane_irf = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=irf_kernel,
            irf_kernel_delta=irf_kernel_delta,
            convolution_method="fftconvolve",
        )

        default_focal_plane_irf = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=irf_kernel,
            irf_kernel_delta=irf_kernel_delta,
        )
        self.assertEqual(
            default_focal_plane_irf.data.shape, self.focal_plane.data.shape
        )
        np.testing.assert_array_equal(
            default_focal_plane_irf.data, focal_plane_irf.data
        )

    def test_apply_fftconvolve(self):
        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        irf_kernel, irf_kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane_irf = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=irf_kernel,
            irf_kernel_delta=irf_kernel_delta,
            convolution_method="fftconvolve",
        )
        self.assertEqual(
            focal_plane_irf.data.shape, self.focal_plane.data.shape
        )

    def test_apply_convolve(self):
        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        irf_kernel, irf_kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane_irf = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=irf_kernel,
            irf_kernel_delta=irf_kernel_delta,
            convolution_method="convolve",
        )
        self.assertEqual(
            focal_plane_irf.data.shape, self.focal_plane.data.shape
        )

    def test_apply_ndimage_convolve(self):
        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()

        irf_kernel, irf_kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane_irf = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=irf_kernel,
            irf_kernel_delta=irf_kernel_delta,
            convolution_method="ndimage.convolve",
        )
        self.assertEqual(
            focal_plane_irf.data.shape, self.focal_plane.data.shape
        )

    def test_apply_oversampled_iprf(self):
        createIntrapixelResponseFunction = (
            CreateOversampledIntrapixelResponseFunction()
        )

        irf_kernel, irf_kernel_delta = createIntrapixelResponseFunction(
            parameters=self.parameters
        )
        applyIntraPixelResponseFunction = ApplyIntraPixelResponseFunction()
        focal_plane_irf = applyIntraPixelResponseFunction(
            focal_plane=self.focal_plane,
            irf_kernel=irf_kernel,
            irf_kernel_delta=irf_kernel_delta,
            convolution_method="fast_convolution",
        )
        self.assertEqual(
            focal_plane_irf.data.shape, self.focal_plane.data.shape
        )


class PopulateFocalPlaneTest(unittest.TestCase):
    parameters = {
        "psf": {"shape": "Airy"},
        "Fnum_x": 5,
        "Fnum_y": 5,
        "type": "spectrometer",
        "detector": {},
    }

    sources = {
        "test_source": CountsPerSecond(
            spectral=np.linspace(1, 10, 100) * u.um,
            data=np.ones((10, 1, 100)),
            metadata={"parsed_parameters": {}},
        )
    }

    output = None
    pointing = None

    def test_airy(self):
        populateFocalPlane = PopulateFocalPlane()
        self.parameters["psf"]["shape"] = "Airy"
        focal_plane = CountsPerSecond(
            spectral=np.linspace(1, 15, 100) * u.um,
            spatial=np.linspace(1, 15, 100) * u.um,
            time=np.linspace(1, 10, 10) * u.hr,
            data=np.zeros((10, 100, 100)),
            metadata={"focal_plane_delta": 10 * u.um},
        )
        focal_p, psf = populateFocalPlane(
            parameters=self.parameters,
            focal_plane=focal_plane,
            sources=self.sources,
        )

        # test no spatial
        focal_plane = CountsPerSecond(
            spectral=np.linspace(1, 15, 100) * u.um,
            spatial=np.zeros(100) * u.um,
            time=np.linspace(1, 10, 10) * u.hr,
            data=np.zeros((10, 100, 100)),
            metadata={"focal_plane_delta": 10 * u.um},
        )
        populateFocalPlane = PopulateFocalPlane()
        focal_p, psf = populateFocalPlane(
            parameters=self.parameters,
            focal_plane=focal_plane,
            sources=self.sources,
        )

    def test_gauss(self):
        self.parameters["psf"]["shape"] = "Gauss"
        populateFocalPlane = PopulateFocalPlane()
        focal_plane = CountsPerSecond(
            spectral=np.linspace(1, 15, 100) * u.um,
            spatial=np.linspace(1, 15, 100) * u.um,
            time=np.linspace(1, 10, 10) * u.hr,
            data=np.zeros((10, 100, 100)),
            metadata={"focal_plane_delta": 10 * u.um},
        )
        focal_p, psf = populateFocalPlane(
            parameters=self.parameters,
            focal_plane=focal_plane,
            sources=self.sources,
        )

    def test_photometer(self):
        self.parameters["type"] = "photometer"
        populateFocalPlane = PopulateFocalPlane()
        focal_plane = CountsPerSecond(
            spectral=np.linspace(1, 15, 100) * u.um,
            spatial=np.linspace(1, 15, 100) * u.um,
            time=np.linspace(1, 10, 10) * u.hr,
            data=np.zeros((10, 100, 100)),
            metadata={"focal_plane_delta": 10 * u.um},
        )
        focal_p, psf = populateFocalPlane(
            parameters=self.parameters,
            focal_plane=focal_plane,
            sources=self.sources,
        )

    # TODO test loaded PSF


class SaturationTimeTest(unittest.TestCase):
    spectral = np.arange(0, 9) * u.pix
    spatial = np.arange(0, 9) * u.pix
    time = np.arange(0, 4) * u.hr
    data = np.zeros((5, 10, 10)) * u.ct / u.s

    focal_plane = CountsPerSecond(
        spectral=spectral, spatial=spatial, data=data, time=time
    )

    def test_saturationTime(self):
        computeSaturationTime = ComputeSaturation()

        self.focal_plane.data[:, 5, 5] = 1
        well_depth = 10 * u.ct

        sat, _, _, _ = computeSaturationTime(
            well_depth=well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
        )
        expected_sat = np.array([10.0, 10.0, 10.0, 10.0, 10.0]) * u.s
        np.testing.assert_array_equal(sat, expected_sat)

    def test_integrationTime(self):
        computeSaturationTime = ComputeSaturation()

        self.focal_plane.data[:, 5, 5] = 1
        well_depth = 10 * u.ct
        f_well_depth = 0.9

        _, int_, _, _ = computeSaturationTime(
            well_depth=well_depth,
            f_well_depth=f_well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
        )
        expected_sat = np.array([9.0, 9.0, 9.0, 9.0, 9.0]) * u.s
        np.testing.assert_array_equal(int_, expected_sat)

    def test_MaxSig(self):
        computeSaturationTime = ComputeSaturation()

        self.focal_plane.data[:, 5, 5] = 1
        well_depth = 10 * u.ct

        _, _, max_, _ = computeSaturationTime(
            well_depth=well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
        )
        expected_max = 1 * u.ct / u.s
        np.testing.assert_array_equal(max_, expected_max)

        f_well_depth = 0.9

        _, _, max_, _ = computeSaturationTime(
            well_depth=well_depth,
            f_well_depth=f_well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
        )
        np.testing.assert_array_equal(max_, expected_max)

    def test_MinSig(self):
        computeSaturationTime = ComputeSaturation()

        self.focal_plane.data[:, 5, 5] = 1
        well_depth = 10 * u.ct

        _, _, _, min_ = computeSaturationTime(
            well_depth=well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
        )
        expected_min = 0 * u.ct / u.s
        np.testing.assert_array_equal(min_, expected_min)

        f_well_depth = 0.9

        _, _, _, min_ = computeSaturationTime(
            well_depth=well_depth,
            f_well_depth=f_well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
        )
        np.testing.assert_array_equal(min_, expected_min)

    def test_foreground(self):
        computeSaturationTime = ComputeSaturation()

        self.focal_plane.data[:, 5, 5] = 1
        well_depth = 10 * u.ct

        sat, _, _, _ = computeSaturationTime(
            well_depth=well_depth,
            focal_plane=self.focal_plane.data * self.focal_plane.data_units,
            frg_focal_plane=self.focal_plane.data
            * self.focal_plane.data_units,
        )
        expected_sat = np.array([5.0, 5.0, 5.0, 5.0, 5.0]) * u.s
        np.testing.assert_array_equal(sat, expected_sat)


class PointingTest(unittest.TestCase):
    computeSourcesPointingOffset = ComputeSourcesPointingOffset()

    def test_pointing_units(self):
        source = {"parsed_parameters": {"ra": 0 * u.deg, "dec": 0 * u.deg}}
        parameters = {
            "detector": {
                "plate_scale": 1 * u.deg / u.micron,
                "delta_pix": 1 * u.micron,
                "oversampling": 1,
            }
        }

        ras = [1 * u.deg, "0h4m0.00000001s", 0.0667 * u.hourangle]
        decs = [1 * u.deg, "1d0m0s", 0.0667 * u.hourangle]

        for ra, dec in itertools.product(ras, decs):
            print(ra, dec)
            a = self.computeSourcesPointingOffset(
                source=source, parameters=parameters, pointing=(ra, dec)
            )
            self.assertListEqual(a, [1, 1])

    def test_angles(self):
        source = {"parsed_parameters": {"ra": 0 * u.deg, "dec": 0 * u.deg}}
        pointing = (1 * u.deg, 1 * u.deg)

        parameters = {
            "detector": {
                "plate_scale": 1 * u.deg / u.micron,
                "delta_pix": 1 * u.micron,
                "oversampling": 1,
            }
        }

        a = self.computeSourcesPointingOffset(
            source=source, parameters=parameters, pointing=pointing
        )
        self.assertListEqual(a, [1, 1])

        parameters = {
            "detector": {
                "plate_scale": 1 * u.deg / u.micron,
                "delta_pix": 1 * u.micron,
                "oversampling": 2,
            }
        }
        a = self.computeSourcesPointingOffset(
            source=source, parameters=parameters, pointing=pointing
        )
        self.assertListEqual(a, [2, 2])

        parameters = {
            "detector": {
                "plate_scale": 1 * u.deg / u.micron,
                "delta_pix": 0.5 * u.micron,
                "oversampling": 1,
            }
        }
        a = self.computeSourcesPointingOffset(
            source=source, parameters=parameters, pointing=pointing
        )
        self.assertListEqual(a, [2, 2])

    def test_plate_scale_units(self):
        source = {"parsed_parameters": {"ra": 0 * u.deg, "dec": 0 * u.deg}}
        pointing = (1 * u.deg, 1 * u.deg)

        parameters = {
            "detector": {
                "plate_scale": 1 * u.deg / u.micron,
                "delta_pix": 1 * u.micron,
                "oversampling": 1,
            }
        }
        val_list = [
            1 * u.deg / u.micron,
            1 * u.deg / u.pixel,
            3600 * u.arcsec / u.micron,
            1000000 * u.deg / u.m,
        ]
        for val in val_list:
            parameters["detector"]["plate_scale"] = val

            a = self.computeSourcesPointingOffset(
                source=source, parameters=parameters, pointing=pointing
            )
            self.assertListEqual(a, [1, 1])

        val_list = [1 * u.deg / u.s, 1 * u.m / u.pixel]
        for val in val_list:
            parameters["detector"]["plate_scale"] = val
            with self.assertRaises(u.UnitConversionError):
                self.computeSourcesPointingOffset(
                    source=source, parameters=parameters, pointing=pointing
                )
        parameters["detector"]["plate_scale"] = 1
        with self.assertRaises(IOError):
            self.computeSourcesPointingOffset(
                source=source, parameters=parameters, pointing=pointing
            )
