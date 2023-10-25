import logging
import unittest

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.physical_models import BlackBody
from inputs import payload_file
from inputs import skip_plot

import exosim.utils as utils
from exosim.log import setLogLevel
from exosim.models.signal import Dimensionless
from exosim.models.signal import Radiance
from exosim.models.signal import Signal
from exosim.tasks.load.loadOpticalElement import LoadOpticalElement
from exosim.tasks.load.loadOptions import LoadOptions

setLogLevel(logging.DEBUG)


class WrongLoadOpticalElement1(LoadOpticalElement):
    def model(self, parameters, wavelength, time):
        return Signal(spectral=[0], data=np.array([0])), Dimensionless(
            spectral=[0], data=np.array([0])
        )


class WrongLoadOpticalElement2(LoadOpticalElement):
    def model(self, parameters, wavelength, time):
        return Radiance(spectral=[0], data=np.array([0])), Signal(
            spectral=[0], data=np.array([0])
        )


class OpticalElementTest(unittest.TestCase):
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

    loadOpticsDefault = LoadOpticalElement()

    def test_loader(self):
        radiance, efficiency = self.loadOpticsDefault(
            parameters=self.mainConfig["payload"]["channel"]["Photometer"][
                "optical_path"
            ]["opticalElement"]["Phot-M3"],
            wavelength=self.wl,
            time=self.tt,
        )
        self.assertIsInstance(radiance, Radiance)
        self.assertIsInstance(efficiency, Dimensionless)
        eff = np.ones(len(self.wl)) * 0.9
        np.testing.assert_array_equal(efficiency.data[0, 0], eff)

        bb = BlackBody(80 * u.K)
        bb_ = 0.03 * bb(self.wl).to(
            u.W / u.m**2 / u.sr / u.um, u.spectral_density(self.wl)
        )
        np.testing.assert_array_almost_equal(radiance.data[0, 0], bb_.value)

    @unittest.skipIf(skip_plot, "This test only produces plots")
    def test_loader_binner(self):
        radiance, efficiency = self.loadOpticsDefault(
            parameters=self.mainConfig["payload"]["channel"]["ch1"][
                "optical_path"
            ]["opticalElement"]["M3"],
            wavelength=self.wl,
            time=self.tt,
        )
        data = self.mainConfig["payload"]["channel"]["ch1"]["optical_path"][
            "opticalElement"
        ]["M3"]["data"]
        wl_data = data["Wavelength"]
        eff_data = data["Reflectivity"]
        plt.plot(wl_data, eff_data, label="data")
        plt.plot(
            efficiency.spectral,
            efficiency.data[0, 0],
            label="parsed",
            ls=":",
            c="r",
        )
        plt.legend()
        plt.show()

    def test_wrong_output(self):
        with self.assertRaises(TypeError) as context:
            wrongLoadOpticalElement1 = WrongLoadOpticalElement1()
            rad, eff = wrongLoadOpticalElement1(
                parameters={}, wavelength=self.wl, time=self.tt
            )

        with self.assertRaises(TypeError) as context:
            wrongLoadOpticalElement2 = WrongLoadOpticalElement2()
            rad, eff = wrongLoadOpticalElement2(
                parameters={}, wavelength=self.wl, time=self.tt
            )
