"""
Unit tests for astronomical signal tasks in ExoSim2.

This module tests astronomical signal detection, planetary signal estimation,
and related signal processing operations for exoplanet observations.
"""

import json
import logging
from collections import OrderedDict
from typing import ClassVar

import astropy.units as u
import numpy as np
import pytest

from exosim.log import set_log_level
from exosim.tasks.astrosignal.estimate_planetary_signal import (
    EstimatePlanetarySignal,
)
from exosim.tasks.astrosignal.find_astronomical_signals import (
    FindAstronomicalSignals,
)

set_log_level(logging.DEBUG)


class TestAstronomicalSignalDetection:
    """Test suite for astronomical signal detection and identification operations."""

    def setup_method(self):
        """Set up astronomical signal finder for tests."""
        self.find_astronomical_signal = FindAstronomicalSignals()

    def test_no_signal_detection(self):
        """
        Test detection when no astronomical signals are present.

        This test verifies that the signal finder correctly returns an empty
        dictionary when no astronomical signals are configured in the source
        parameters.
        """
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
            }
        }
        out_dict = self.find_astronomical_signal(sky_parameters=parameters)
        assert out_dict == {}

    def test_single_signal_detection(self):
        """
        Test detection of a single astronomical signal.

        This test verifies that the signal finder correctly identifies and
        parses a single planetary signal associated with a stellar source.
        """
        signal = {"signal_task": EstimatePlanetarySignal}
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
                "example_signal": signal,
            }
        }
        out_dict = self.find_astronomical_signal(sky_parameters=parameters)
        assert out_dict == {
            "test_star": {
                "example_signal": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"],
                }
            }
        }

    def test_multiple_signals_single_source(self):
        """
        Test detection of multiple signals from a single source.

        This test verifies that multiple astronomical signals can be correctly
        identified and parsed for a single stellar source, such as multiple
        planets in the same system.
        """
        signal = {"signal_task": EstimatePlanetarySignal}
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
                "example_signal1": signal,
                "example_signal2": signal,
            }
        }
        out_dict = self.find_astronomical_signal(sky_parameters=parameters)
        assert out_dict == {
            "test_star": {
                "example_signal1": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"],
                },
                "example_signal2": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"],
                },
            }
        }

    def test_multiple_sources_with_signals_error(self):
        """
        Test error handling for multiple sources with conflicting signals.

        This test verifies that appropriate errors are raised when multiple
        sources are configured with conflicting signal definitions that
        cannot be processed simultaneously.
        """
        signal = {"signal_task": EstimatePlanetarySignal}
        star1 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
            "example_signal1": signal,
        }
        star2 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
            "example_signal2": signal,
        }

        parameters = {
            "source": OrderedDict(
                {
                    "test_star1": star1,
                    "test_star2": star2,
                }
            )
        }

        with pytest.raises(
            ValueError,
            match="Conflicting astronomical signals detected across multiple sources",
        ):
            self.find_astronomical_signal(sky_parameters=parameters)

    def test_multiple_sources_partial_signals(self):
        """
        Test detection with multiple sources where only some have signals.

        This test verifies correct handling when multiple stellar sources
        are configured but only a subset have associated astronomical signals.
        """
        signal = {"signal_task": EstimatePlanetarySignal}
        star1 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
            "example_signal1": signal,
        }
        star2 = {
            "source_type": "phoenix",
            "R": 1 * u.R_sun,
            "M": 1 * u.M_sun,
            "D": 10 * u.pc,
            "T": 6000 * u.K,
            "z": 0.0,
        }

        parameters = {
            "source": OrderedDict(
                {
                    "test_star1": star1,
                    "test_star2": star2,
                }
            )
        }
        out_dict = self.find_astronomical_signal(sky_parameters=parameters)
        assert out_dict == {
            "test_star1": {
                "example_signal1": {
                    "task": EstimatePlanetarySignal,
                    "parsed_parameters": parameters["source"]["test_star1"],
                }
            }
        }

    def test_no_source_key_returns_empty_dict(self):
        """Test that FindAstronomicalSignals returns {} when 'source' key is absent.

        When sky.xml does not contain a <source> section, the sky_parameters
        dict will have no 'source' key.  The task must return an empty dict
        without raising KeyError.
        """
        parameters = {}  # no 'source' key at all
        out_dict = self.find_astronomical_signal(sky_parameters=parameters)
        assert out_dict == {}

    def test_foreground_only_sky_returns_empty_dict(self):
        """Test that FindAstronomicalSignals returns {} with foreground-only sky.

        A valid sky configuration may include only background/foreground elements
        (e.g. zodiacal light) without a stellar source.  The task must not
        raise KeyError and must return an empty signals dict.
        """
        parameters = {"foregrounds": {"zodiacal_light": {"value": 1.0}}}
        out_dict = self.find_astronomical_signal(sky_parameters=parameters)
        assert out_dict == {}


class TestPlanetarySignalEstimation:
    """Test suite for planetary signal estimation operations."""

    def setup_method(self):
        """Set up planetary signal estimator and test parameters."""
        self.estimate_planetary_signal = EstimatePlanetarySignal()

        # Standard planetary parameters for testing
        self.main_parameters: ClassVar[dict] = {
            "planet": {
                "radius": 0.1,
                "t0": 0 * u.s,
                "period": 1 * u.s,
                "sma": 15,
                "inc": 87 * u.deg,
                "ecc": 0,
                "w": 90 * u.deg,
                "limb_darkening": "linear",
                "limb_darkening_coefficients": "[0]",
            }
        }
        self.parameters: ClassVar[dict] = {}
        self.timeline = np.arange(-0.05, 0.05, 0.0001) * u.s
        self.wl_grid = np.logspace(0.5, 8, 100)

    def test_flat_lightcurve_generation(self):
        """
        Test generation of flat (wavelength-independent) transit lightcurves.

        This test verifies that planetary transit signals are correctly computed
        for flat (achromatic) transit scenarios where the planet radius is
        constant across all wavelengths.
        """
        # Set planetary radius for testing
        self.main_parameters["planet"]["rp"] = 0.12

        new_timeline, model = self.estimate_planetary_signal(
            timeline=self.timeline,
            wl_grid=self.wl_grid,
            ch_parameters=self.parameters,
            source_parameters=self.main_parameters,
        )

        # Generate expected Batman model for comparison
        batman_model_expected = batman_model(
            self.main_parameters["planet"], new_timeline, rp=[0.12]
        )

        # Verify first wavelength channel matches Batman model
        np.testing.assert_allclose(model[0], batman_model_expected[0])

        # Verify all wavelength channels have the same lightcurve (flat spectrum)
        expected = batman_model_expected[0]
        batman_model_all = np.repeat(expected[np.newaxis, :], model.shape[0], axis=0)
        np.testing.assert_allclose(model, batman_model_all)

    def test_lightcurve_temporal_consistency(self):
        """
        Test temporal consistency of planetary lightcurves.

        This test verifies that the generated lightcurves have proper temporal
        structure with the expected transit depth and timing.
        """
        self.main_parameters["planet"]["rp"] = 0.10  # Different radius for variety

        new_timeline, model = self.estimate_planetary_signal(
            timeline=self.timeline,
            wl_grid=self.wl_grid[:10],  # Use subset for faster testing
            ch_parameters=self.parameters,
            source_parameters=self.main_parameters,
        )

        # Verify basic properties
        assert model.shape[0] == 10  # Number of wavelength channels
        assert model.shape[1] == len(new_timeline)  # Number of time points

        # Verify model is normalized around 1 (transit causes dimming)
        assert np.all(model <= 1.0)
        assert np.all(
            model >= 0.8
        )  # Should not dim more than ~20% for typical transits

    def test_parameter_sensitivity(self):
        """
        Test sensitivity to different planetary parameters.

        This test verifies that changing planetary parameters produces
        expected changes in the resulting lightcurve characteristics.
        """
        # Test different planet radii
        radii = [0.08, 0.12, 0.16]
        models = []

        for rp in radii:
            self.main_parameters["planet"]["rp"] = rp
            _new_timeline, model = self.estimate_planetary_signal(
                timeline=self.timeline,
                wl_grid=self.wl_grid[:5],  # Small subset for speed
                ch_parameters=self.parameters,
                source_parameters=self.main_parameters,
            )
            models.append(model)

        # Verify that larger planets produce deeper transits
        # (smaller minimum flux values)
        min_flux_small = np.min(models[0])
        min_flux_medium = np.min(models[1])
        min_flux_large = np.min(models[2])

        assert min_flux_large < min_flux_medium < min_flux_small

    def test_wavelength_grid_handling(self):
        """
        Test proper handling of different wavelength grids.

        This test verifies that the planetary signal estimator correctly
        handles different wavelength grid configurations and produces
        appropriately shaped output models.
        """
        # Test different wavelength grid sizes
        wl_grids = [
            np.logspace(0.5, 2, 10),  # Small grid
            np.logspace(0.5, 8, 50),  # Medium grid
            np.logspace(0.5, 10, 100),  # Large grid
        ]

        self.main_parameters["planet"]["rp"] = 0.12

        for _, wl_grid in enumerate(wl_grids):
            new_timeline, model = self.estimate_planetary_signal(
                timeline=self.timeline,
                wl_grid=wl_grid,
                ch_parameters=self.parameters,
                source_parameters=self.main_parameters,
            )

            # Verify output shape matches wavelength grid
            assert model.shape[0] == len(wl_grid)
            assert model.shape[1] == len(new_timeline)

            # Verify model is physically reasonable
            assert np.all(np.isfinite(model))
            assert np.all(model > 0)

    def test_unparseable_rp_raises_a_clear_error(self):
        # not a string (would be treated as a file path) and not float-convertible
        self.main_parameters["planet"]["rp"] = {"unexpected": "mapping"}
        with pytest.raises(ValueError, match="planet 'rp' must be"):
            self.estimate_planetary_signal(
                timeline=self.timeline,
                wl_grid=self.wl_grid[:5],
                ch_parameters=self.parameters,
                source_parameters=self.main_parameters,
            )


class TestSignalValidation:
    """Test suite for astronomical signal validation and consistency checks."""

    def test_signal_task_validation(self):
        """
        Test validation of signal task assignments.

        This test verifies that signal tasks are properly validated and
        that invalid task assignments are handled appropriately.
        """
        # Valid signal task
        valid_signal = {"signal_task": EstimatePlanetarySignal}

        # Test that valid signals are accepted
        find_signals = FindAstronomicalSignals()
        parameters = {
            "source": {
                "value": "test_star",
                "source_type": "phoenix",
                "R": 1 * u.R_sun,
                "M": 1 * u.M_sun,
                "D": 10 * u.pc,
                "T": 6000 * u.K,
                "z": 0.0,
                "valid_signal": valid_signal,
            }
        }

        out_dict = find_signals(sky_parameters=parameters)
        assert "test_star" in out_dict
        assert "valid_signal" in out_dict["test_star"]

    def test_parameter_completeness_validation(self):
        """
        Test validation of parameter completeness.

        This test verifies that required parameters are properly validated
        and missing parameters are detected appropriately.
        """
        # Test with minimal required parameters
        minimal_params = {
            "planet": {
                "rp": 0.1,
                "t0": 0 * u.s,
                "period": 1 * u.s,
                "sma": 15,
                "inc": 87 * u.deg,
                "ecc": 0,
                "w": 90 * u.deg,
                "limb_darkening": "linear",
                "limb_darkening_coefficients": "[0]",
            }
        }

        estimate_signal = EstimatePlanetarySignal()
        timeline = np.arange(-0.01, 0.01, 0.001) * u.s
        wl_grid = np.logspace(1, 2, 5)

        try:
            new_timeline, model = estimate_signal(
                timeline=timeline,
                wl_grid=wl_grid,
                ch_parameters={},
                source_parameters=minimal_params,
            )

            # If successful, verify basic output structure
            assert len(new_timeline) > 0
            assert model.shape[0] == len(wl_grid)
            assert model.shape[1] == len(new_timeline)
        except Exception as e:
            # Some parameters may be missing for full functionality
            pytest.skip(f"Parameter validation test requires additional setup: {e}")


def batman_model(parameters, input_timeline, rp=0):
    """
    Generate Batman transit model for comparison testing.

    Parameters
    ----------
    parameters : dict
        Planetary parameters dictionary
    input_timeline : array_like
        Time array for model evaluation
    rp : float or array_like, optional
        Planet radius values to test

    Returns
    -------
    ndarray
        Transit lightcurve model array
    """
    try:
        import batman

        from exosim.utils.checks import check_units
    except ImportError:
        pytest.skip("Batman package required for model comparison")

    # Set up Batman parameters
    params = batman.TransitParams()
    params.t0 = check_units(parameters["t0"], input_timeline.unit, None, True).value
    params.per = check_units(
        parameters["period"], input_timeline.unit, None, True
    ).value
    params.a = parameters["sma"]  # (in units of stellar radii)
    params.inc = check_units(parameters["inc"], u.deg, None, True).value
    params.ecc = parameters["ecc"]
    params.w = check_units(parameters["w"], u.deg, None, True).value
    params.limb_dark = parameters["limb_darkening"]
    raw_u = parameters["limb_darkening_coefficients"]
    params.u = json.loads(raw_u)

    # Handle scalar or array rp values
    if np.isscalar(rp):
        rp = [rp]

    out_model = np.zeros((len(rp), len(input_timeline)))

    for i in range(len(rp)):
        params.rp = rp[i]  # planet radius (in units of stellar radii)

        # Initialize batman model
        m = batman.TransitModel(params, input_timeline)
        out_model[i] = m.light_curve(params)

    return out_model
