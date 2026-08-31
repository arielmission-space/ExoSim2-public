"""
Unit tests for zodiacal light foreground modeling.

This module contains tests for zodiacal light emission modeling,
including direct modeling with zodiacal factors, coordinate-based
estimation, and integration with parsing functionality.
"""

import logging
import os

import astropy.units as u
import numpy as np
import pytest

from exosim.log import set_log_level
from exosim.tasks.foregrounds import EstimateZodi
from exosim.tasks.parse import ParseZodi

set_log_level(logging.DEBUG)


def exolib_bb_model(wl, T):
    """
    Blackbody model using pre-defined constants.

    This function implements the same blackbody model used in the original
    exosim tests for consistent comparison with zodiacal light calculations.

    Parameters
    ----------
    wl : astropy.units.Quantity
        Wavelengths to evaluate the blackbody function
    T : astropy.units.Quantity
        Blackbody temperature

    Returns
    -------
    astropy.units.Quantity
        Blackbody spectral radiance in wavelength space
    """
    a = np.float64(1.191042768e8) * u.um**5 * u.W / u.m**2 / u.sr / u.um
    b = np.float64(14387.7516) * 1 * u.um * u.K
    x = b / (wl * T)
    return a / wl**5 / (np.exp(x) - 1.0)


class TestZodiacal:
    """Test suite for zodiacal light estimation functionality."""

    estimate_zodi = EstimateZodi()

    def test_zodi_model(self):
        """
        Test basic zodiacal light model calculation.

        This test verifies that the zodiacal light model produces
        the expected spectral emission based on the standard
        dual blackbody model (solar-like + warm dust components).
        """
        wl = np.logspace(np.log10(0.45), np.log10(2.2), 6000) * u.um

        zodi = self.estimate_zodi(wavelength=wl, zodiacal_factor=1)

        # Expected zodiacal emission: solar component + thermal dust component
        units = u.W / (u.m**2 * u.um * u.sr)
        zodi_emission = (
            3.5e-14 * exolib_bb_model(wl, 5500.0 * u.K)
            + 3.58e-8 * exolib_bb_model(wl, 270.0 * u.K)
        ).to(units)

        # Test that computed and expected emissions match within precision
        np.testing.assert_array_almost_equal(
            zodi_emission.value / zodi.data, np.ones_like(zodi.data), decimal=5
        )

    def test_zodi_factor_defaults_to_zero_when_missing(self):
        wl = np.logspace(np.log10(0.5), np.log10(2.0), 100) * u.um
        # no zodiacal_factor and no coordinates -> the model treats ``a`` as 0
        zodi = self.estimate_zodi(wavelength=wl)
        assert np.all(zodi.data == 0)

    def test_default_map_search_gives_up_instead_of_looping_forever(
        self, tmp_path, monkeypatch
    ):
        # point the module at a deep path with no 'data' directory anywhere
        # above it: the walk-up must terminate and raise, not spin forever.
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        monkeypatch.setattr(
            "os.path.realpath", lambda _p: str(deep / "estimate_zodi.py")
        )
        task = EstimateZodi()
        task.set_log_name()
        with pytest.raises(OSError, match="default zodi map file not found"):
            task.zodiacal_fit_direction((10.0 * u.deg, -20.0 * u.deg))

    def test_fit_coordinate(self):
        """
        Test zodiacal light estimation from sky coordinates.

        This test verifies that zodiacal light can be estimated
        from sky coordinates and that the result matches a known
        zodiacal factor for the given coordinates.
        """
        wl = np.logspace(np.log10(0.45), np.log10(2.2), 6000) * u.um

        # Estimate zodiacal light from specific sky coordinates
        zodi = self.estimate_zodi(
            wavelength=wl,
            coordinates=(
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
        )

        # Compare with known zodiacal factor for these coordinates
        zodi_known = self.estimate_zodi(
            wavelength=wl, zodiacal_factor=1.4536394185097168
        )

        np.testing.assert_array_almost_equal(
            zodi_known.data / zodi.data, np.ones_like(zodi.data), decimal=5
        )


class TestZodiacalParse:
    """Test suite for zodiacal light parsing and parameter handling."""

    parse_zodi = ParseZodi()
    wl = np.logspace(np.log10(0.45), np.log10(2.2), 100) * u.um
    tt = np.linspace(1, 10, 5) * u.hr

    def test_zodi_model(self):
        """
        Test parsing of zodiacal light parameters with zodiacal factor.

        This test verifies that the zodiacal light parser can process
        parameters with a specific zodiacal factor value.
        """
        parameters = {"zodiacal_factor": 25, "value": "zodi"}
        result = self.parse_zodi(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )

        # Verify that parsing completes without error and returns valid result
        assert result is not None

    def test_zodi_coordinates(self):
        """
        Test parsing of zodiacal light parameters with sky coordinates.

        This test verifies that the zodiacal light parser can process
        parameters with sky coordinates to determine zodiacal factors.
        """
        parameters = {
            "coordinates": (
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
            "value": "zodi",
        }
        result = self.parse_zodi(
            parameters=parameters, wavelength=self.wl, time=self.tt
        )

        # Verify that parsing completes without error and returns valid result
        assert result is not None

    def test_zodi_map(self, project_root):
        """
        Test parsing with zodiacal light map file.

        This test verifies that the parser can use zodiacal light map
        files for more detailed spatial modeling, and properly handles
        error cases for missing files.
        """
        file_map = os.path.join(project_root, "data/Zodi_map.hdf5")
        parameters = {
            "coordinates": (
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
            "zodi_map": file_map,
            "value": "zodi",
        }

        # Test successful parsing with valid map file
        if os.path.exists(file_map):
            result = self.parse_zodi(
                parameters=parameters, wavelength=self.wl, time=self.tt
            )
            assert result is not None
        else:
            # Skip test if map file doesn't exist
            pytest.skip(f"Zodiacal map file not found: {file_map}")

        # Test error handling for invalid map file path
        parameters_invalid = {
            "coordinates": (
                90.03841366076144 * u.deg,
                -66.55432012293919 * u.deg,
            ),
            "zodi_map": "wrong_dir/nonexistent_file.hdf5",
            "value": "zodi",
        }
        with pytest.raises(OSError, match="Zodi map file not found"):
            self.parse_zodi(
                parameters=parameters_invalid, wavelength=self.wl, time=self.tt
            )
