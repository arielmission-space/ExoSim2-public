import logging
import unittest

import astropy.units as u
import numpy as np
from astropy.modeling.physical_models import BlackBody
from inputs import payload_file

import exosim.utils as utils
from exosim.log import setLogLevel
from exosim.models.signal import Dimensionless
from exosim.models.signal import Radiance
from exosim.tasks.load.loadOptions import LoadOptions
from exosim.tasks.parse import ParseOpticalElement
from exosim.tasks.parse import ParsePath

setLogLevel(logging.INFO)


class OpticalPathTest(unittest.TestCase):
    loadOption = LoadOptions()
    mainConfig = loadOption(filename=payload_file)

    wl = utils.grids.wl_grid(
        mainConfig["wl_grid"]["wl_min"],
        mainConfig["wl_grid"]["wl_max"],
        mainConfig["wl_grid"]["logbin_resolution"],
    )

    tt = utils.grids.time_grid(
        mainConfig["time_grid"]["start_time"],
        mainConfig["time_grid"]["end_time"],
        mainConfig["time_grid"]["low_frequencies_resolution"],
    )

    def test_parser_single(self):
        parseOpticalElement = ParseOpticalElement()
        path = parseOpticalElement(
            parameters=self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ]["opticalElement"]["Phot-M3"],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(path, dict)
        self.assertIsInstance(path["radiance"], Radiance)
        self.assertIsInstance(path["efficiency"], Dimensionless)
        eff = np.ones(len(self.wl)) * 0.9
        np.testing.assert_array_equal(path["efficiency"].data[0, 0], eff)
        bb = BlackBody(80 * u.K)
        bb_ = 0.03 * bb(self.wl).to(
            u.W / u.m**2 / u.sr / u.um, u.spectral_density(self.wl)
        )
        np.testing.assert_array_almost_equal(
            path["radiance"].data[0, 0], bb_.value
        )

    def test_parser_list(self):
        parsePath = ParsePath()
        path = parsePath(
            parameters=self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(path, dict)
        self.assertIsInstance(path["radiance_0"], Radiance)
        self.assertIsInstance(path["efficiency"], Dimensionless)
        eff = np.ones(len(self.wl)) * 0.9**5
        idx = np.where(self.wl < 1.0 * u.um)[0]

        np.testing.assert_array_almost_equal(
            path["efficiency"].data[0, 0, idx], eff[idx]
        )

        bb = BlackBody(80 * u.K)
        bb_ = 0.03 * bb(self.wl).to(
            u.W / u.m**2 / u.sr / u.um, u.spectral_density(self.wl)
        )
        rad = (
            bb_ * 0.9**4 + bb_ * 0.9**3 + bb_ * 0.9**2 + bb_ * 0.9 + bb_
        )

        np.testing.assert_array_almost_equal(
            path["radiance_0"].data[0, 0, idx], rad[idx].value
        )

    def test_parser_slit(self):
        parsePath = ParsePath()
        path = parsePath(
            parameters=self.mainConfig["payload"]["channel"]["Spectrometer"][
                "optical_path"
            ],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(path["radiance_0"], Radiance)
        self.assertIsInstance(path["radiance_1"], Radiance)
        self.assertIsInstance(path["efficiency"], Dimensionless)
        self.assertTrue("slit_width" in path["radiance_0"].metadata.keys())
        self.assertEqual(path["radiance_0"].metadata["slit_width"], 0.5 * u.mm)
        self.assertFalse("slit_width" in path["radiance_1"].metadata.keys())

    def test_iterative_building(self):
        parseOpticalElement = ParseOpticalElement()
        path_prev = parseOpticalElement(
            parameters=self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ]["opticalElement"]["D1"],
            wavelength=self.wl,
            time=self.tt,
        )

        parsePath = ParsePath()
        path_new = parsePath(
            parameters=self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ],
            wavelength=self.wl,
            time=self.tt,
            light_path=path_prev,
        )
        self.assertIsInstance(path_new, dict)
        self.assertIsInstance(path_new["radiance_0"], Radiance)
        self.assertIsInstance(path_new["efficiency"], Dimensionless)
        eff = np.ones(len(self.wl)) * 0.9**6
        eff[self.wl > 1.0 * u.um] = 0
        eff[self.wl > 1.0 * u.um] = 0
        idx = np.where(self.wl < 1.0 * u.um)[0]
        np.testing.assert_array_almost_equal(
            path_new["efficiency"].data[0, 0, idx], eff[idx]
        )


class OpticalPathIsolatedTest(unittest.TestCase):
    loadOption = LoadOptions()
    mainConfig = loadOption(filename=payload_file)

    wl = utils.grids.wl_grid(
        mainConfig["wl_grid"]["wl_min"],
        mainConfig["wl_grid"]["wl_max"],
        mainConfig["wl_grid"]["logbin_resolution"],
    )

    tt = utils.grids.time_grid(
        mainConfig["time_grid"]["start_time"],
        mainConfig["time_grid"]["end_time"],
        mainConfig["time_grid"]["low_frequencies_resolution"],
    )

    def test_isolate(self):
        for opt in self.mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ]["opticalElement"]:
            self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ]["opticalElement"][opt]["isolate"] = True

        parsePath = ParsePath()
        path_new = parsePath(
            parameters=self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(path_new, dict)
        self.assertIsInstance(path_new["radiance_0_D1"], Radiance)
        self.assertIsInstance(path_new["radiance_1_Phot-M1"], Radiance)
        self.assertIsInstance(path_new["efficiency"], Dimensionless)


class ForegroundPathTest(unittest.TestCase):
    loadOption = LoadOptions()
    mainConfig = loadOption(filename=payload_file)

    wl = utils.grids.wl_grid(
        mainConfig["wl_grid"]["wl_min"],
        mainConfig["wl_grid"]["wl_max"],
        mainConfig["wl_grid"]["logbin_resolution"],
    )

    tt = utils.grids.time_grid(
        mainConfig["time_grid"]["start_time"],
        mainConfig["time_grid"]["end_time"],
        mainConfig["time_grid"]["low_frequencies_resolution"],
    )

    def test_parser_single(self):
        parseOpticalElement = ParseOpticalElement()
        path = parseOpticalElement(
            parameters=self.mainConfig["sky"]["foregrounds"]["opticalElement"][
                "earthsky"
            ],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(path, dict)
        self.assertIsInstance(path["radiance"], Radiance)
        self.assertIsInstance(path["efficiency"], Dimensionless)

    def test_parser_list(self):
        parsePath = ParsePath()
        path = parsePath(
            parameters=self.mainConfig["sky"]["foregrounds"],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(path, dict)
        self.assertIsInstance(path["radiance_0"], Radiance)
        self.assertIsInstance(path["efficiency"], Dimensionless)

    def test_iterative_building(self):
        parseOpticalElement = ParseOpticalElement()
        path_prev = parseOpticalElement(
            parameters=self.mainConfig["sky"]["foregrounds"]["opticalElement"][
                "earthsky"
            ],
            wavelength=self.wl,
            time=self.tt,
        )

        parsePath = ParsePath()
        path_new = parsePath(
            parameters=self.mainConfig["sky"]["foregrounds"],
            wavelength=self.wl,
            time=self.tt,
            light_path=path_prev,
        )
        self.assertIsInstance(path_new, dict)
        self.assertIsInstance(path_new["radiance_0"], Radiance)
        self.assertIsInstance(path_new["efficiency"], Dimensionless)
