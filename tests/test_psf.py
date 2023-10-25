import logging
import os
import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import photutils
from inputs import skip_plot
from inputs import test_dir

import exosim.tasks.instrument as instrument
import exosim.utils.aperture as psf_util
from exosim.log import setLogLevel
from exosim.utils.klass_factory import find_task
from exosim.utils.psf import create_psf

setLogLevel(logging.DEBUG)
"""
These are not real tests, are only checkers.
"""


class CreatePsfTest(unittest.TestCase):
    def test_airy(self):
        wl = np.linspace(1.95, 3.78, 10) * u.um
        psf = create_psf(wl, 15.5, 18 * u.um, shape="airy")
        print(psf.shape)

    def test_gauss(self):
        wl = np.linspace(1.95, 3.78, 10) * u.um
        psf = create_psf(wl, 15.5, 18 * u.um, shape="gauss")
        print(psf.shape)

    def test_nzero(self):
        wl = np.linspace(1.95, 3.78, 10) * u.um
        psf_4 = create_psf(wl, 15.5, 18 * u.um, shape="airy", nzero=4)
        psf_8 = create_psf(wl, 15.5, 18 * u.um, shape="airy", nzero=8)
        self.assertGreater(psf_8.shape, psf_4.shape)

    def test_max_size(self):
        wl = np.linspace(1.95, 3.78, 10) * u.um
        psf = create_psf(wl, 15.5, 18 * u.um, shape="airy")
        self.assertNotEqual(psf.shape, (10, 5, 5))
        psf = create_psf(
            wl, 15.5, 18 * u.um, shape="airy", max_array_size=(5, 5)
        )
        self.assertEqual(psf.shape, (10, 5, 5))
        psf = create_psf(
            wl, 15.5, 18 * u.um, shape="airy", max_array_size=(4, 4)
        )
        self.assertEqual(psf.shape, (10, 5, 5))

    def test_array_size(self):
        wl = np.linspace(1.95, 3.78, 10) * u.um
        psf = create_psf(wl, 15.5, 18 * u.um, shape="airy")
        self.assertNotEqual(psf.shape, (10, 31, 31))
        psf = create_psf(
            wl, 15.5, 18 * u.um, shape="airy", array_size=(31, 21)
        )
        self.assertEqual(psf.shape, (10, 31, 21))
        psf = create_psf(
            wl, 15.5, 18 * u.um, shape="airy", array_size=(30, 20)
        )
        self.assertEqual(psf.shape, (10, 31, 21))

    def test_array_size_full(self):
        wl = np.linspace(1.95, 3.78, 10) * u.um
        psf = create_psf(wl, 15.5, 18 * u.um, shape="airy")
        self.assertNotEqual(psf.shape, (10, 31, 31))
        psf = create_psf(
            wl,
            15.5,
            18 * u.um,
            shape="airy",
            max_array_size=(21, 21),
            array_size=("full", "full"),
        )
        self.assertEqual(psf.shape, (10, 21, 21))
        with self.assertRaises(ValueError):
            psf = create_psf(
                wl, 15.5, 18 * u.um, shape="airy", array_size=("full", "full")
            )

        psf = create_psf(
            wl,
            15.5,
            18 * u.um,
            shape="airy",
            max_array_size=(21, 21),
            array_size=(31, "full"),
        )
        self.assertEqual(psf.shape, (10, 31, 21))
        with self.assertRaises(ValueError):
            psf = create_psf(
                wl, 15.5, 18 * u.um, shape="airy", array_size=(31, "full")
            )

        psf = create_psf(
            wl,
            15.5,
            18 * u.um,
            shape="airy",
            max_array_size=(21, 21),
            array_size=("full", 31),
        )
        self.assertEqual(psf.shape, (10, 21, 31))
        with self.assertRaises(ValueError):
            psf = create_psf(
                wl, 15.5, 18 * u.um, shape="airy", array_size=("full", 31)
            )


class LoadPAOSTest(unittest.TestCase):
    paos_data = os.path.join(test_dir, "PAOS_ab0.h5")

    parameters = {
        "detector": {
            "oversampling": 2,
            "delta_pix": 10 * u.um,
            "spatial_pix": 32,
            "spectral_pix": 32,
        }
    }

    def test_finder(self):
        psf_task = find_task("LoadPsfPaos", instrument.LoadPsf)

    def test_loader(self):
        wl = np.linspace(1, 2.8, 5) * u.um
        tt = np.linspace(0, 10, 2) * u.hr

        loadPsfPaos = instrument.LoadPsfPaos()
        cube, norms = loadPsfPaos(
            filename=self.paos_data,
            parameters=self.parameters,
            wavelength=wl,
            time=tt,
        )
        print("cube", cube.shape)
        print("norms", norms.shape)
        # plt.plot(wl, norms[0])
        # plt.show()
        #
        # plt.imshow(cube[0,0, :,:])
        # plt.show()


class LoadPAOSTimeInterpTest(unittest.TestCase):
    paos_data = os.path.join(test_dir, "PAOS_ab0.h5")
    paos_data_1 = os.path.join(test_dir, "PAOS_ab1.h5")

    parameters = {
        "detector": {
            "oversampling": 2,
            "delta_pix": 10 * u.um,
            "spatial_pix": 32,
            "spectral_pix": 32,
        },
        "psf": {"t0": 5 * u.hr},
    }

    def test_finder(self):
        psf_task = find_task("LoadPsfPaosTimeInterp", instrument.LoadPsf)

    def test_loader(self):
        wl = np.linspace(1, 2.8, 5) * u.um
        tt = np.linspace(0, 5, 2) * u.hr

        paos_data = self.paos_data + ", " + self.paos_data_1
        loadPsfPaosTimeInterp = instrument.LoadPsfPaosTimeInterp()
        cube, norms = loadPsfPaosTimeInterp(
            filename=paos_data,
            parameters=self.parameters,
            wavelength=wl,
            time=tt,
        )
        print("cube", cube.shape)
        print("norms", norms.shape)
        # plt.imshow(cube[0, 0, :, :])
        # plt.show()
        #
        # plt.imshow(cube[-1, 0, :, :])
        # plt.show()


class PsfApertureTest(unittest.TestCase):
    psf = create_psf(4 * u.um, 15.5, 6 * u.um, shape="airy")
    print(psf.shape)

    def test_energy_rectangular(self):
        sizes, surf, ene = psf_util.find_rectangular_aperture(self.psf, 0.84)
        print(sizes, ene)
        self.assertGreaterEqual(np.round(ene, decimals=2), 0.84)
        print(sizes, surf, ene)
        aper = photutils.aperture.RectangularAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2),
            h=sizes[1],
            w=sizes[0],
        )
        phot_ = photutils.aperture.aperture_photometry(self.psf, aper)
        phot = phot_["aperture_sum"].data[0]
        new_ene = phot / self.psf.sum()
        self.assertGreaterEqual(np.round(new_ene, decimals=2), 0.84)
        print(new_ene)

    @unittest.skipIf(skip_plot, "This test only produces plots")
    def test_aperture_plot_rectangular(self):
        sizes, surf, ene = psf_util.find_rectangular_aperture(self.psf, 0.84)

        aper = photutils.aperture.RectangularAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2),
            h=sizes[1],
            w=sizes[0],
        )
        phot_ = photutils.aperture.aperture_photometry(self.psf, aper)
        phot = phot_["aperture_sum"].data[0]
        new_ene = phot / self.psf.sum()
        self.assertGreater(new_ene, 0.84)

        plt.imshow(self.psf)
        aper.plot(color="g", lw=2, label="Photometry aperture")
        plt.show()

    def test_energy_elliptical(self):
        sizes, surf, ene = psf_util.find_elliptical_aperture(
            self.psf,
            0.84,
        )
        self.assertGreaterEqual(np.round(ene, decimals=2), 0.84)
        print(sizes, surf, ene)
        aper = photutils.aperture.EllipticalAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2),
            a=sizes[1],
            b=sizes[0],
        )
        phot_ = photutils.aperture.aperture_photometry(self.psf, aper)
        phot = phot_["aperture_sum"].data[0]
        new_ene = phot / self.psf.sum()
        self.assertGreaterEqual(np.round(new_ene, decimals=2), 0.84)
        print(new_ene)

    @unittest.skipIf(skip_plot, "This test only produces plots")
    def test_aperture_plot_elliptical(self):
        sizes, surf, ene = psf_util.find_elliptical_aperture(
            self.psf, 0.84, resolution=1
        )

        aper = photutils.aperture.EllipticalAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2),
            a=sizes[1],
            b=sizes[0],
        )
        phot_ = photutils.aperture.aperture_photometry(self.psf, aper)
        phot = phot_["aperture_sum"].data[0]
        new_ene = phot / self.psf.sum()
        self.assertGreater(new_ene, 0.84)

        plt.imshow(self.psf)
        aper.plot(color="g", lw=2, label="Photometry aperture")
        plt.show()

    def test_energy_bin(self):
        sizes, surf, ene = psf_util.find_bin_aperture(self.psf, 0.84, 10)
        self.assertGreaterEqual(np.round(ene, decimals=2), 0.84)
        print(sizes, surf, ene)
        aper = photutils.aperture.RectangularAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2), h=sizes, w=10
        )
        aper_full = photutils.aperture.RectangularAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2),
            h=self.psf.shape[1],
            w=10,
        )
        phot_ = photutils.aperture.aperture_photometry(self.psf, aper)
        phot = phot_["aperture_sum"].data[0]

        phot_ = photutils.aperture.aperture_photometry(self.psf, aper_full)
        phot_full = phot_["aperture_sum"].data[0]

        new_ene = phot / phot_full
        self.assertGreaterEqual(np.round(new_ene, decimals=2), 0.84)
        print(new_ene)

    @unittest.skipIf(skip_plot, "This test only produces plots")
    def test_aperture_plot_bin(self):
        sizes, surf, ene = psf_util.find_bin_aperture(
            self.psf, 0.84, 10, resolution=1
        )

        aper = photutils.aperture.RectangularAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2), h=sizes, w=10
        )
        aper_full = photutils.aperture.RectangularAperture(
            (self.psf.shape[1] // 2, self.psf.shape[0] // 2),
            h=self.psf.shape[1],
            w=10,
        )
        plt.imshow(self.psf)
        aper.plot(color="b", lw=2, label="Photometry aperture")
        aper_full.plot(color="g", lw=2, label="Photometry aperture")

        plt.show()
