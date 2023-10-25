import logging
import os
import unittest

import astropy.units as u
import numpy as np
from inputs import main_path
from test_sources import exolib_bb_model

from exosim.log import setLogLevel
from exosim.tasks.foregrounds import EstimateZodi
from exosim.tasks.parse import ParseZodi

setLogLevel(logging.DEBUG)


class ZodiacalTest(unittest.TestCase):
    estimateZodi = EstimateZodi()

    def test_zodi_model(self):
        wl = np.logspace(np.log10(0.45), np.log10(2.2), 6000) * u.um

        zodi = self.estimateZodi(wavelength=wl, zodiacal_factor=1)

        units = u.W / (u.m**2 * u.um * u.sr)
        zodi_emission = (
            3.5e-14 * exolib_bb_model(wl, 5500.0 * u.K)
            + 3.58e-8 * exolib_bb_model(wl, 270.0 * u.K)
        ).to(units)

        np.testing.assert_array_almost_equal(
            zodi_emission.value / zodi.data, np.ones_like(zodi.data), decimal=5
        )

    def test_fit_coordinate(self):
        wl = np.logspace(np.log10(0.45), np.log10(2.2), 6000) * u.um

        zodi = self.estimateZodi(
            wavelength=wl,
            coordinates=(
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
        )

        zodi_knwon = self.estimateZodi(
            wavelength=wl, zodiacal_factor=1.4536394185097168
        )

        np.testing.assert_array_almost_equal(
            zodi_knwon.data / zodi.data, np.ones_like(zodi.data), decimal=5
        )


class ZodiacalParseTest(unittest.TestCase):
    parseZodi = ParseZodi()
    wl = np.logspace(np.log10(0.45), np.log10(2.2), 100) * u.um
    tt = np.linspace(1, 10, 5) * u.hr

    def test_zodi_model(self):
        parameters = {"zodiacal_factor": 25, "value": "zodi"}
        self.parseZodi(parameters=parameters, wavelength=self.wl, time=self.tt)

    def test_zodi_coordinates(self):
        parameters = {
            "coordinates": (
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
            "value": "zodi",
        }
        self.parseZodi(parameters=parameters, wavelength=self.wl, time=self.tt)

    def test_zodi_map(self):
        file_map = os.path.join(main_path, "data/Zodi_map.hdf5")
        parameters = {
            "coordinates": (
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
            "zodi_map": file_map,
            "value": "zodi",
        }
        self.parseZodi(parameters=parameters, wavelength=self.wl, time=self.tt)

        with self.assertRaises(OSError):
            parameters = {
                "coordinates": (
                    90.03841366076144 * u.deg,
                    -66.55432012293919 * u.deg,
                ),
                "zodi_map": "wrong_dir",
                "value": "zodi",
            }
            self.parseZodi(
                parameters=parameters, wavelength=self.wl, time=self.tt
            )
