import logging
import os
import tempfile

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as const
from astropy.modeling.physical_models import BlackBody

import exosim.utils as utils
from exosim.log import setLogLevel
from exosim.models.signal import Dimensionless, Radiance, Signal
from exosim.tasks.instrument.loadResponsivity import LoadResponsivity
from exosim.tasks.load.loadOpticalElement import LoadOpticalElement
from exosim.tasks.load.loadOpticalElementHDF5 import LoadOpticalElementHDF5
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


@pytest.fixture
def load_main_config(payload_file):
    """Fixture per caricare la configurazione principale e creare griglie di lunghezza d'onda e tempo."""
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
    return mainConfig, wl, tt


def test_loader(load_main_config):
    mainConfig, wl, tt = load_main_config
    loadOpticsDefault = LoadOpticalElement()

    radiance, efficiency = loadOpticsDefault(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ]["opticalElement"]["Phot-M3"],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(radiance, Radiance)
    assert isinstance(efficiency, Dimensionless)

    eff = np.ones(len(wl)) * 0.9
    np.testing.assert_array_equal(efficiency.data[0, 0], eff)

    bb = BlackBody(80 * u.K)
    bb_ = 0.03 * bb(wl).to(u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl))
    np.testing.assert_array_almost_equal(radiance.data[0, 0], bb_.value)


def test_loader_binner(load_main_config, skip_plot):
    if skip_plot:
        pytest.skip("This test only produces plots")

    mainConfig, wl, tt = load_main_config
    loadOpticsDefault = LoadOpticalElement()

    radiance, efficiency = loadOpticsDefault(
        parameters=mainConfig["payload"]["channel"]["ch1"]["optical_path"][
            "opticalElement"
        ]["M3"],
        wavelength=wl,
        time=tt,
    )

    data = mainConfig["payload"]["channel"]["ch1"]["optical_path"][
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

@pytest.mark.usefixtures("payload_file")
class TestResponsivity:
    @pytest.fixture(autouse=True)
    def _setup(self, payload_file):
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

        self.wl = wl
        self.tt = tt
        self.paylaod = paylaod

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
        with pytest.raises(TypeError):
            wrongLoadOpticalElement1 = WrongLoadOpticalElement1()
            rad, eff = wrongLoadOpticalElement1(
                parameters={}, wavelength=self.wl, time=self.tt
            )

        with pytest.raises(u.UnitConversionError):
            wrongLoadOpticalElement2 = WrongLoadOpticalElement2()
            rad, eff = wrongLoadOpticalElement2(
                parameters={}, wavelength=self.wl, time=self.tt
            )


@pytest.fixture
def hdf5_file_setup_teardown(tmp_path):
    # Use a valid temporary directory
    hdf5_file_path = tmp_path / "test.h5"

    # Create some test data
    wavelength = np.linspace(1, 10, 100)
    rng = np.random.default_rng(42)
    radiance = rng.random((100,))
    efficiency = rng.random((100,))

    # Write the test data to the HDF5 file
    with h5py.File(str(hdf5_file_path), "w") as f:
        group = f.create_group("test_group")
        group.create_dataset("wavelength", data=wavelength)
        group.create_dataset("radiance", data=radiance)
        group.create_dataset("efficiency", data=efficiency)

    # Set up the test parameters
    parameters = {
        "hdf5_file": str(hdf5_file_path),
        "group_key": "test_group",
        "wavelength_key": "wavelength",
        "radiance_key": "radiance",
        "efficiency_key": "efficiency",
    }
    wavelength_unit = np.linspace(1, 10, 100) * u.um
    time_unit = np.linspace(0, 10, 100) * u.s

    yield parameters, wavelength_unit, time_unit


class TestLoadOpticalElementHDF5:
    def test_model(self, hdf5_file_setup_teardown):
        parameters, wavelength, time = hdf5_file_setup_teardown
        # Test the model method
        load_optical_element = LoadOpticalElementHDF5()
        radiance, efficiency = load_optical_element.model(parameters, wavelength, time)
        assert isinstance(radiance, Radiance)
        assert isinstance(efficiency, Dimensionless)

    def test_get_data(self, hdf5_file_setup_teardown):
        parameters, wavelength, time = hdf5_file_setup_teardown
        # Test the _get_data method
        load_optical_element = LoadOpticalElementHDF5()
        data = load_optical_element._get_data(parameters, wavelength, time, 'radiance_key', Radiance)
        assert isinstance(data, Radiance)
