import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.physical_models import BlackBody

import exosim.utils as utils
from exosim.log import setLogLevel
from exosim.models.signal import Dimensionless, Radiance
from exosim.tasks.load.loadOptions import LoadOptions
from exosim.tasks.parse import ParseOpticalElement, ParsePath

setLogLevel(logging.INFO)


@pytest.fixture
def load_main_config(payload_file):
    """Load the main configuration for tests."""
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


def test_parser_single(load_main_config):
    mainConfig, wl, tt = load_main_config
    parseOpticalElement = ParseOpticalElement()

    path = parseOpticalElement(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ]["opticalElement"]["Phot-M3"],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(path, dict)
    assert isinstance(path["radiance"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)

    eff = np.ones(len(wl)) * 0.9
    np.testing.assert_array_equal(path["efficiency"].data[0, 0], eff)

    bb = BlackBody(80 * u.K)
    bb_ = 0.03 * bb(wl).to(u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl))
    np.testing.assert_array_almost_equal(path["radiance"].data[0, 0], bb_.value)


def test_parser_list(load_main_config):
    mainConfig, wl, tt = load_main_config
    parsePath = ParsePath()

    path = parsePath(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(path, dict)
    assert isinstance(path["radiance_0"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)

    eff = np.ones(len(wl)) * 0.9**5
    idx = np.where(wl < 1.0 * u.um)[0]
    np.testing.assert_array_almost_equal(
        path["efficiency"].data[0, 0, idx], eff[idx]
    )


def test_parser_slit(load_main_config):
    mainConfig, wl, tt = load_main_config
    parsePath = ParsePath()

    path = parsePath(
        parameters=mainConfig["payload"]["channel"]["Spectrometer"][
            "optical_path"
        ],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(path["radiance_0"], Radiance)
    assert isinstance(path["radiance_1"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)
    assert "slit_width" in path["radiance_0"].metadata.keys()
    assert path["radiance_0"].metadata["slit_width"] == 0.5 * u.mm
    assert "slit_width" not in path["radiance_1"].metadata.keys()


def test_iterative_building(load_main_config):
    mainConfig, wl, tt = load_main_config
    parseOpticalElement = ParseOpticalElement()
    path_prev = parseOpticalElement(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ]["opticalElement"]["D1"],
        wavelength=wl,
        time=tt,
    )

    parsePath = ParsePath()
    path_new = parsePath(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ],
        wavelength=wl,
        time=tt,
        light_path=path_prev,
    )

    assert isinstance(path_new, dict)
    assert isinstance(path_new["radiance_0"], Radiance)
    assert isinstance(path_new["efficiency"], Dimensionless)

    eff = np.ones(len(wl)) * 0.9**6
    eff[wl > 1.0 * u.um] = 0
    idx = np.where(wl < 1.0 * u.um)[0]
    np.testing.assert_array_almost_equal(
        path_new["efficiency"].data[0, 0, idx], eff[idx]
    )


def test_isolated_optical_path(load_main_config):
    mainConfig, wl, tt = load_main_config

    # Set all optical elements to isolated mode
    for opt in mainConfig["payload"]["channel"]["Photometer"][
        "optical_path"
    ]["opticalElement"]:
        mainConfig["payload"]["channel"]["Photometer"]["optical_path"][
            "opticalElement"
        ][opt]["isolate"] = True

    parsePath = ParsePath()
    path_new = parsePath(
        parameters=mainConfig["payload"]["channel"]["Photometer"][
            "optical_path"
        ],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(path_new, dict)
    assert isinstance(path_new["radiance_0_D1"], Radiance)
    assert isinstance(path_new["radiance_1_Phot-M1"], Radiance)
    assert isinstance(path_new["efficiency"], Dimensionless)


def test_foreground_parser_single(load_main_config):
    mainConfig, wl, tt = load_main_config
    parseOpticalElement = ParseOpticalElement()

    path = parseOpticalElement(
        parameters=mainConfig["sky"]["foregrounds"]["opticalElement"][
            "earthsky"
        ],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(path, dict)
    assert isinstance(path["radiance"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)


def test_foreground_parser_list(load_main_config):
    mainConfig, wl, tt = load_main_config
    parsePath = ParsePath()

    path = parsePath(
        parameters=mainConfig["sky"]["foregrounds"],
        wavelength=wl,
        time=tt,
    )

    assert isinstance(path, dict)
    assert isinstance(path["radiance_0"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)


def test_foreground_iterative_building(load_main_config):
    mainConfig, wl, tt = load_main_config
    parseOpticalElement = ParseOpticalElement()
    path_prev = parseOpticalElement(
        parameters=mainConfig["sky"]["foregrounds"]["opticalElement"][
            "earthsky"
        ],
        wavelength=wl,
        time=tt,
    )

    parsePath = ParsePath()
    path_new = parsePath(
        parameters=mainConfig["sky"]["foregrounds"],
        wavelength=wl,
        time=tt,
        light_path=path_prev,
    )

    assert isinstance(path_new, dict)
    assert isinstance(path_new["radiance_0"], Radiance)
    assert isinstance(path_new["efficiency"], Dimensionless)
