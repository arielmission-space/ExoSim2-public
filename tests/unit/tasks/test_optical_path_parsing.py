"""
Unit tests for optical path parsing functionality.

This module contains tests for parsing optical elements and optical paths,
including single elements, lists of elements, and complex optical systems
with foreground sources and slit configurations.
"""

import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.physical_models import BlackBody

import exosim.utils as utils
from exosim.log import set_log_level
from exosim.models.signal import Dimensionless, Radiance
from exosim.tasks.load.load_options import LoadOptions
from exosim.tasks.parse import ParseOpticalElement, ParsePath

set_log_level(logging.INFO)


@pytest.fixture
def load_main_config(payload_file):
    """
    Load the main configuration for optical path tests.

    Parameters
    ----------
    payload_file : str
        Path to the payload configuration file

    Returns
    -------
    tuple
        Tuple containing (main_config, wavelength_grid, time_grid)
    """
    load_option = LoadOptions()
    main_config = load_option(filename=payload_file)

    # Create wavelength grid from configuration
    wl = utils.grids.wl_grid(
        main_config["wl_grid"]["wl_min"],
        main_config["wl_grid"]["wl_max"],
        main_config["wl_grid"]["logbin_resolution"],
    )

    # Create time grid from configuration
    tt = utils.grids.time_grid(
        main_config["time_grid"]["start_time"],
        main_config["time_grid"]["end_time"],
        main_config["time_grid"]["low_frequencies_resolution"],
    )

    return main_config, wl, tt


def test_parser_single(load_main_config):
    """
    Test parsing of a single optical element.

    This test verifies that a single optical element can be parsed correctly,
    producing valid radiance and efficiency data with expected values.
    """
    main_config, wl, tt = load_main_config
    parse_optical_element = ParseOpticalElement()

    # Parse a single mirror element (Phot-M3)
    path = parse_optical_element(
        parameters=main_config["payload"]["channel"]["Photometer"]["optical_path"][
            "opticalElement"
        ]["Phot-M3"],
        wavelength=wl,
        time=tt,
    )

    # Verify the parsed path structure
    assert isinstance(path, dict)
    assert isinstance(path["radiance"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)

    # Check efficiency values (expected 90% efficiency)
    expected_efficiency = np.ones(len(wl)) * 0.9
    np.testing.assert_array_equal(path["efficiency"].data[0, 0], expected_efficiency)

    # Check radiance values (blackbody at 80K with 3% emissivity)
    bb = BlackBody(80 * u.K)
    expected_radiance = 0.03 * bb(wl).to(
        u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl)
    )
    np.testing.assert_array_almost_equal(
        path["radiance"].data[0, 0], expected_radiance.value
    )


def test_parser_list(load_main_config):
    """
    Test parsing of a list of optical elements (full optical path).

    This test verifies that multiple optical elements can be parsed together
    to form a complete optical path with combined efficiencies.
    """
    main_config, wl, tt = load_main_config
    parse_path = ParsePath()

    # Parse the complete photometer optical path
    path = parse_path(
        parameters=main_config["payload"]["channel"]["Photometer"]["optical_path"],
        wavelength=wl,
        time=tt,
    )

    # Verify the parsed path structure
    assert isinstance(path, dict)
    assert isinstance(path["radiance_0"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)

    # Check combined efficiency (5 elements at 90% each = 0.9^5)
    expected_combined_efficiency = np.ones(len(wl)) * 0.9**5
    idx = np.where(wl < 1.0 * u.um)[0]
    np.testing.assert_array_almost_equal(
        path["efficiency"].data[0, 0, idx], expected_combined_efficiency[idx]
    )


def test_parser_slit(load_main_config):
    """
    Test parsing of optical path with slit configuration.

    This test verifies that slit metadata is correctly parsed and attached
    to the appropriate radiance components.
    """
    main_config, wl, tt = load_main_config
    parse_path = ParsePath()

    # Parse the spectrometer optical path (includes slit)
    path = parse_path(
        parameters=main_config["payload"]["channel"]["Spectrometer"]["optical_path"],
        wavelength=wl,
        time=tt,
    )

    # Verify the parsed path structure
    assert isinstance(path["radiance_0"], Radiance)
    assert isinstance(path["radiance_1"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)

    # Check slit metadata
    assert "slit_width" in path["radiance_0"].metadata
    assert path["radiance_0"].metadata["slit_width"] == 0.5 * u.mm
    assert "slit_width" not in path["radiance_1"].metadata


def test_iterative_building(load_main_config):
    """
    Test iterative building of optical paths.

    This test verifies that optical paths can be built incrementally,
    starting from a pre-existing path component.
    """
    main_config, wl, tt = load_main_config

    # First parse a single element (D1)
    parse_optical_element = ParseOpticalElement()
    path_prev = parse_optical_element(
        parameters=main_config["payload"]["channel"]["Photometer"]["optical_path"][
            "opticalElement"
        ]["D1"],
        wavelength=wl,
        time=tt,
    )

    # Then build the complete path using the previous path as starting point
    parse_path = ParsePath()
    path_new = parse_path(
        parameters=main_config["payload"]["channel"]["Photometer"]["optical_path"],
        wavelength=wl,
        time=tt,
        light_path=path_prev,
    )

    # Verify the new path structure
    assert isinstance(path_new, dict)
    assert isinstance(path_new["radiance_0"], Radiance)
    assert isinstance(path_new["efficiency"], Dimensionless)

    # Check combined efficiency (6 elements total: 1 from path_prev + 5 from full path)
    expected_efficiency = np.ones(len(wl)) * 0.9**6
    expected_efficiency[wl > 1.0 * u.um] = 0  # Wavelength cutoff
    idx = np.where(wl < 1.0 * u.um)[0]
    np.testing.assert_array_almost_equal(
        path_new["efficiency"].data[0, 0, idx], expected_efficiency[idx]
    )


def test_isolated_optical_path(load_main_config):
    """
    Test parsing of optical path with isolated elements.

    This test verifies that when optical elements are set to isolated mode,
    each element produces separate radiance components.
    """
    main_config, wl, tt = load_main_config

    # Set all optical elements to isolated mode
    for opt_element in main_config["payload"]["channel"]["Photometer"]["optical_path"][
        "opticalElement"
    ]:
        main_config["payload"]["channel"]["Photometer"]["optical_path"][
            "opticalElement"
        ][opt_element]["isolate"] = True

    # Parse the isolated optical path
    parse_path = ParsePath()
    path_new = parse_path(
        parameters=main_config["payload"]["channel"]["Photometer"]["optical_path"],
        wavelength=wl,
        time=tt,
    )

    # Verify isolated element structure
    assert isinstance(path_new, dict)
    assert isinstance(path_new["radiance_0_D1"], Radiance)
    assert isinstance(path_new["radiance_1_Phot-M1"], Radiance)
    assert isinstance(path_new["efficiency"], Dimensionless)


def test_foreground_parser_single(load_main_config):
    """
    Test parsing of a single foreground optical element.

    This test verifies that foreground sources (like Earth sky) can be
    parsed correctly as optical elements.
    """
    main_config, wl, tt = load_main_config
    parse_optical_element = ParseOpticalElement()

    # Parse a single foreground element (Earth sky)
    path = parse_optical_element(
        parameters=main_config["sky"]["foregrounds"]["opticalElement"]["earthsky"],
        wavelength=wl,
        time=tt,
    )

    # Verify the parsed foreground structure
    assert isinstance(path, dict)
    assert isinstance(path["radiance"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)


def test_foreground_parser_list(load_main_config):
    """
    Test parsing of multiple foreground elements.

    This test verifies that multiple foreground sources can be parsed
    together to form a combined foreground contribution.
    """
    main_config, wl, tt = load_main_config
    parse_path = ParsePath()

    # Parse all foreground elements
    path = parse_path(
        parameters=main_config["sky"]["foregrounds"],
        wavelength=wl,
        time=tt,
    )

    # Verify the parsed foreground structure
    assert isinstance(path, dict)
    assert isinstance(path["radiance_0"], Radiance)
    assert isinstance(path["efficiency"], Dimensionless)


def test_foreground_iterative_building(load_main_config):
    """
    Test iterative building of foreground optical paths.

    This test verifies that foreground paths can be built incrementally,
    combining multiple foreground sources step by step.
    """
    main_config, wl, tt = load_main_config

    # First parse a single foreground element
    parse_optical_element = ParseOpticalElement()
    path_prev = parse_optical_element(
        parameters=main_config["sky"]["foregrounds"]["opticalElement"]["earthsky"],
        wavelength=wl,
        time=tt,
    )

    # Then build the complete foreground path
    parse_path = ParsePath()
    path_new = parse_path(
        parameters=main_config["sky"]["foregrounds"],
        wavelength=wl,
        time=tt,
        light_path=path_prev,
    )

    # Verify the combined foreground structure
    assert isinstance(path_new, dict)
    assert isinstance(path_new["radiance_0"], Radiance)
    assert isinstance(path_new["efficiency"], Dimensionless)


def test_parse_optical_element_requires_a_value_key():
    import pytest

    task = ParsePath()
    task.set_log_name()
    with pytest.raises(KeyError, match="'value' key is required"):
        task._parse_optical_element(
            {"type": "filter", "temperature": 70}, output_file=None
        )
