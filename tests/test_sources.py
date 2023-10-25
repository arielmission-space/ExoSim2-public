import logging
import os
import unittest
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as cc
from astropy.io import ascii
from inputs import example_dir
from inputs import phoenix_file
from inputs import phoenix_stellar_model
from inputs import skip_plot
from inputs import test_dir

from exosim.log import setLogLevel
from exosim.tasks.parse import ParseSource
from exosim.tasks.parse import ParseSources
from exosim.tasks.sed import CreateCustomSource
from exosim.tasks.sed import CreatePlanckStar
from exosim.tasks.sed import LoadCustom
from exosim.tasks.sed import LoadPhoenix
from exosim.tasks.sed import PrepareSed

setLogLevel(logging.DEBUG)


def exolib_bb_model(wl, T):
    a = np.float64(1.191042768e8) * u.um**5 * u.W / u.m**2 / u.sr / u.um
    b = np.float64(14387.7516) * 1 * u.um * u.K
    try:
        x = b / (wl * T)
        bb = a / wl**5 / (np.exp(x) - 1.0)
    except ArithmeticError:
        bb = np.zeros_like(wl)
    return bb


class PlanckStarTest(unittest.TestCase):
    createPlanckStar = CreatePlanckStar()
    wl = np.linspace(0.5, 7.8, 10000) * u.um
    T = 5778 * u.K
    R = 1 * u.R_sun
    D = 1 * u.au
    sed = createPlanckStar(wavelength=wl, T=T, R=R, D=D)

    omega_star = np.pi * (R.si / D.si) ** 2 * u.sr
    sed_exolib = omega_star * exolib_bb_model(wl, T)

    def test_units(self):
        self.assertEqual(self.sed.data_units, self.sed_exolib.unit)

    def test_values(self):
        np.testing.assert_array_almost_equal(
            self.sed_exolib.value / self.sed.data,
            np.ones_like(self.sed.data),
            decimal=5,
        )


class LoadPhoenixTest(unittest.TestCase):
    loadPhoenix = LoadPhoenix()

    @unittest.skipIf(
        not os.path.isdir(phoenix_stellar_model), "phoenix dir not found"
    )
    def test_load_star_from_dir(self):
        D = 12.975 * u.pc
        T = 3016 * u.K
        M = 0.15 * u.Msun
        R = 0.218 * u.Rsun
        z = 0.0
        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)

        sed = self.loadPhoenix(
            path=phoenix_stellar_model, T=T, D=D, R=R, z=z, logg=logg
        )

    @unittest.skipIf(
        not os.path.isfile(phoenix_file), "phoenix file not found"
    )
    def test_load_star_from_file(self):
        D = 12.975 * u.pc
        R = 0.218 * u.Rsun

        self.loadPhoenix(filename=phoenix_file, D=D, R=R)

    @unittest.skipIf(
        not os.path.isfile(phoenix_file)
        or not os.path.isdir(phoenix_stellar_model)
        or "Windows" in os.environ.get("OS", ""),
        "phoenix data not found or Windows OS in use",
    )
    def test_compare(self):
        D = 12.975 * u.pc
        T = 3016 * u.K
        M = 0.15 * u.Msun
        R = 0.218 * u.Rsun
        z = 0.0
        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)

        sed_dir = self.loadPhoenix(
            path=phoenix_stellar_model, T=T, D=D, R=R, z=z, logg=logg
        )
        sed_file = self.loadPhoenix(filename=phoenix_file, D=D, R=R)

        self.assertEqual(sed_dir.metadata["phoenix_file"], phoenix_file)
        self.assertEqual(sed_file.metadata["phoenix_file"], phoenix_file)

        np.testing.assert_array_equal(sed_dir.data, sed_file.data)

    @unittest.skipIf(
        not os.path.isfile(phoenix_file)
        or not os.path.isdir(phoenix_stellar_model),
        "phoenix data not found",
    )
    def test_errors(self):
        D = 12.975 * u.pc
        T = 3016
        M = 0.15 * u.Msun
        R = 0.218 * u.Rsun
        z = 0.0
        g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
        logg = np.log10(g.value)

        with self.assertRaises(IOError):
            self.loadPhoenix(path="not a path", T=T, D=D, R=R, z=z, logg=logg)
        with self.assertRaises(IOError):
            self.loadPhoenix(T=T, D=D, R=R, z=z, logg=logg)
        with self.assertRaises(KeyError):
            self.loadPhoenix(
                path=phoenix_stellar_model, D=D, R=R, z=z, logg=logg
            )
        with self.assertRaises(KeyError):
            self.loadPhoenix(
                path=phoenix_stellar_model,
                T=T,
                D=D,
                R=R,
                z=z,
            )
        with self.assertRaises(KeyError):
            self.loadPhoenix(
                path=phoenix_stellar_model, T=T, D=D, z=z, logg=logg
            )
        with self.assertRaises(KeyError):
            self.loadPhoenix(
                path=phoenix_stellar_model, T=T, R=R, z=z, logg=logg
            )
        with self.assertRaises(OSError):
            self.loadPhoenix(path=test_dir, T=T, D=D, R=R, z=z, logg=logg)


class LoadCustomTest(unittest.TestCase):
    custom_file = os.path.join(example_dir, "customsed.csv")
    loadCustom = LoadCustom()

    @unittest.skipIf(not os.path.isfile(custom_file), "custom file not found")
    def test_load(self):
        D = 1 * u.au
        R = 1 * u.Rsun
        sed = self.loadCustom(filename=self.custom_file, D=D, R=R)

        ph = ascii.read(self.custom_file, format="ecsv")
        ph_sed = ph["Sed"].data * (R.to(u.m) / D.to(u.m)) ** 2

        np.testing.assert_array_equal(sed.data[0, 0], ph_sed)


class LoadSedTest(unittest.TestCase):
    wl = np.linspace(0.5, 7.8, 10000) * u.um
    T = 5778 * u.K
    R = 1 * u.R_sun
    D = 1 * u.au
    M = 1 * u.Msun
    z = 0.0
    g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
    logg = np.log10(g.value)

    loadSed = PrepareSed()
    createPlanckStar = CreatePlanckStar()
    loadPhoenix = LoadPhoenix()

    def test_planck(self):
        sed_l = self.loadSed(
            source_type="planck",
            wavelength=self.wl,
            T=self.T,
            R=self.R,
            D=self.D,
        )
        sed_p = self.createPlanckStar(
            wavelength=self.wl, T=self.T, R=self.R, D=self.D
        )

        np.testing.assert_array_equal(sed_l.data, sed_p.data)
        np.testing.assert_array_equal(sed_l.spectral, sed_p.spectral)

    @unittest.skipIf(
        not os.path.isfile(phoenix_file), "phoenix file not found"
    )
    def test_phoenix_dir(self):
        sed_l = self.loadSed(
            source_type="phoenix",
            path=phoenix_stellar_model,
            T=self.T,
            R=self.R,
            D=self.D,
            logg=self.logg,
            z=self.z,
        )
        sed_p = self.loadPhoenix(
            path=phoenix_stellar_model,
            T=self.T,
            R=self.R,
            D=self.D,
            logg=self.logg,
            z=self.z,
        )

        np.testing.assert_array_equal(sed_l.data, sed_p.data)
        np.testing.assert_array_equal(sed_l.spectral, sed_p.spectral)

    @unittest.skipIf(
        not os.path.isfile(phoenix_file), "phoenix file not found"
    )
    def test_phoenix_file(self):
        sed_l = self.loadSed(
            source_type="phoenix", filename=phoenix_file, R=self.R, D=self.D
        )
        sed_p = self.loadPhoenix(filename=phoenix_file, R=self.R, D=self.D)

        np.testing.assert_array_equal(sed_l.data, sed_p.data)
        np.testing.assert_array_equal(sed_l.spectral, sed_p.spectral)


@unittest.skipIf(skip_plot, "This test only produces plots")
class PlotAndBinningTest(unittest.TestCase):
    T = 5778 * u.K
    R = 1 * u.R_sun
    D = 1 * u.au
    M = 1 * u.Msun
    z = 0.0
    g = (cc.G * M.si / R.si**2).to(u.cm / u.s**2)
    logg = np.log10(g.value)

    wl = np.linspace(0, 8, 1000) * u.um

    @unittest.skipIf(
        not os.path.isdir(phoenix_stellar_model), "phoenix dir not found"
    )
    def test_plot_phoenix(self):
        loadPhoenix = LoadPhoenix()
        sed_ph = loadPhoenix(
            path=phoenix_stellar_model,
            T=self.T,
            D=self.D,
            R=self.R,
            z=self.z,
            logg=self.logg,
        )

        plt.plot(
            sed_ph.spectral, sed_ph.data[0, 0], label="phoenix", alpha=0.5
        )
        sed_ph.spectral_rebin(self.wl)
        plt.plot(
            sed_ph.spectral,
            sed_ph.data[0, 0],
            label="phoenix binned",
            ls=":",
            c="r",
        )
        plt.legend()
        plt.xlim(0, 8)
        plt.show()

    def test_plot_planck(self):
        wl_ = np.linspace(0, 8, 10000) * u.um
        createPlanckStar = CreatePlanckStar()
        sed_planck = createPlanckStar(
            wavelength=wl_, T=self.T, R=self.R, D=self.D
        )
        plt.plot(sed_planck.spectral, sed_planck.data[0, 0], label="planck")
        sed_planck.spectral_rebin(self.wl)
        plt.plot(
            sed_planck.spectral,
            sed_planck.data[0, 0],
            label="planck binned",
            ls=":",
            c="r",
        )
        plt.legend()
        plt.xlim(0, 8)
        plt.show()


class ParseSourceTest(unittest.TestCase):
    wl = np.linspace(0.5, 8, 1000) * u.um
    tt = np.linspace(0.5, 1, 10) * u.hr
    custom_file = os.path.join(example_dir, "customsed.csv")

    def test_raw_data_parse(self):
        parameters = {
            "value": "test_star",
            "source_type": "planck",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
        }
        parseSource = ParseSource()
        out = parseSource(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )

        omega_star = (
            np.pi * (parameters["R"].si / parameters["D"].si) ** 2 * u.sr
        )
        sed_exolib = omega_star * exolib_bb_model(self.wl, parameters["T"])

        np.testing.assert_array_almost_equal(
            out["test_star"].data[0, 0] / sed_exolib.value,
            np.ones(len(self.wl)),
            decimal=5,
        )

    @unittest.skipIf(
        not os.path.isdir(phoenix_stellar_model), "phoenix dir not found"
    )
    def test_phoenix_parse(self):
        parameters = {
            "value": "test_star",
            "source_type": "phoenix",
            "path": phoenix_stellar_model,
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
        }

        parseSource = ParseSource()
        out = parseSource(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )

        g = (cc.G * parameters["M"].si / parameters["R"].si ** 2).to(
            u.cm / u.s**2
        )
        logg = np.log10(g.value)
        loadPhoenix = LoadPhoenix()
        sed = loadPhoenix(
            path=phoenix_stellar_model,
            T=parameters["T"],
            D=parameters["D"],
            R=parameters["R"],
            z=parameters["z"],
            logg=logg,
        )

        sed.spectral_rebin(out["test_star"].spectral)
        sed.temporal_rebin(out["test_star"].time)
        np.testing.assert_array_almost_equal(out["test_star"].data, sed.data)

    @unittest.skipIf(not os.path.isfile(custom_file), "custom file not found")
    def test_custom_parse(self):
        parameters = {
            "value": "test_star",
            "source_type": "custom",
            "filename": os.path.join(example_dir, "customsed.csv"),
            "R": 1 * u.R_sun,
            "D": 10 * u.pc,
        }

        parseSource = ParseSource()
        out = parseSource(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )
        loadCustom = LoadCustom()
        sed = loadCustom(
            filename=parameters["filename"],
            D=parameters["D"],
            R=parameters["R"],
        )

        sed.spectral_rebin(out["test_star"].spectral)
        sed.temporal_rebin(out["test_star"].time)
        np.testing.assert_array_almost_equal(out["test_star"].data, sed.data)

    def test_source_task(self):
        parameters = {
            "value": "test_star",
            "source_task": "CreateCustomSource",
            "R": 1 * u.R_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "wl_min": 0.5 * u.um,
            "wl_max": 7.8 * u.um,
            "n_points": 10000,
        }

        parseSource = ParseSource()
        out = parseSource(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )
        createCustomSource = CreateCustomSource()
        sed = createCustomSource(parameters=parameters)
        sed.spectral_rebin(out["test_star"].spectral)
        sed.temporal_rebin(out["test_star"].time)
        np.testing.assert_array_almost_equal(out["test_star"].data, sed.data)

    def test_wrong_parse(self):
        parameters = {
            "value": "test_star",
            "source_type": "unsupported",
            "R": 1 * u.R_sun,
            "D": 10 * u.pc,
        }

        parseSource = ParseSource()
        with self.assertRaises(KeyError):
            parseSource(
                parameters=parameters, wavelength=self.wl, time=self.tt
            )


class ParseSourcesTest(unittest.TestCase):
    wl = np.linspace(0.5, 8, 1000) * u.um
    tt = np.linspace(0.5, 1, 10) * u.hr

    def test_raw_data_parse(self):
        star1 = {
            "value": "test_star",
            "source_type": "planck",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
        }

        star2 = {
            "value": "test_star2",
            "source_type": "planck",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 20 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
        }
        parameters = OrderedDict(
            {
                "star1": star1,
                "star2": star2,
            }
        )
        parseSources = ParseSources()
        out = parseSources(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )

        omega_star = np.pi * (star1["R"].si / star1["D"].si) ** 2 * u.sr

        sed_exolib = omega_star * exolib_bb_model(self.wl, star1["T"])
        np.testing.assert_array_almost_equal(
            out["test_star"].data[0, 0] / sed_exolib.value,
            np.ones(len(self.wl)),
            decimal=5,
        )

        np.testing.assert_array_almost_equal(
            out["test_star2"].data[0, 0] / (sed_exolib.value / 4),
            np.ones(len(self.wl)),
            decimal=5,
        )


class CreateCustomSourceTest(unittest.TestCase):
    wl = np.linspace(0.5, 7.8, 10000) * u.um
    T = 5778 * u.K
    R = 1 * u.R_sun
    D = 1 * u.au

    omega_star = np.pi * (R.si / D.si) ** 2 * u.sr
    sed_exolib = omega_star * exolib_bb_model(wl, T)

    parameters = {
        "T": T,
        "R": R,
        "D": D,
        "wl_min": 0.5 * u.um,
        "wl_max": 7.8 * u.um,
        "n_points": 10000,
    }

    createCustomSource = CreateCustomSource()
    sed = createCustomSource(parameters=parameters)

    def test_units(self):
        self.assertEqual(self.sed.data_units, self.sed_exolib.unit)

    def test_values(self):
        np.testing.assert_array_almost_equal(
            self.sed_exolib.value / self.sed.data,
            np.ones_like(self.sed.data),
            decimal=5,
        )

    def test_error(self):
        class ExampleCreateCustomSource(CreateCustomSource):
            def model(self, parameters):
                return None

        exampleCreateCustomSource = ExampleCreateCustomSource()
        with self.assertRaises(TypeError):
            exampleCreateCustomSource(parameters=self.parameters)
