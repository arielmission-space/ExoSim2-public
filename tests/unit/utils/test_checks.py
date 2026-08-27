"""
Unit tests for ExoSim utility checks module.

This module tests the checks utility functions including:
- check_units: Unit validation and conversion
- find_key and look_for_key: Dictionary key search utilities
"""

import astropy.units as u
import numpy as np
import pytest

from exosim.utils.checks import check_units, find_key, look_for_key


class TestCheckUnits:
    """Test check_units function for unit validation and conversion."""

    def test_check_units_with_compatible_units(self):
        """Test check_units with compatible units."""
        # Test basic unit conversion
        input_data = 1000 * u.mm
        desired_units = u.m

        result = check_units(input_data, desired_units)

        assert result.unit == u.m
        assert np.isclose(result.value, 1.0)

    def test_check_units_with_same_units(self):
        """Test check_units when units are already correct."""
        input_data = 5.0 * u.m
        desired_units = u.m

        result = check_units(input_data, desired_units)

        assert result.unit == u.m
        assert result.value == 5.0

    def test_check_units_with_no_unit_force_true(self):
        """Test check_units with no unit and force=True."""
        input_data = 100.0  # No unit
        desired_units = u.m

        result = check_units(input_data, desired_units, force=True)

        assert result.unit == u.m
        assert result.value == 100.0

    def test_check_units_with_array_input(self):
        """Test check_units with array input."""
        input_data = np.array([1, 2, 3]) * u.km
        desired_units = u.m

        result = check_units(input_data, desired_units)

        assert result.unit == u.m
        assert np.array_equal(result.value, [1000, 2000, 3000])

    def test_check_units_with_string_input(self):
        """Test check_units with string input."""
        input_data = "123.45"
        desired_units = u.m

        result = check_units(input_data, desired_units, force=True)

        assert result.unit == u.m
        assert np.isclose(result.value, 123.45)

    def test_check_units_with_incompatible_units(self):
        """Test check_units with incompatible units raises error."""
        input_data = 5.0 * u.kg  # Mass unit
        desired_units = u.m  # Length unit

        with pytest.raises(u.UnitConversionError):
            check_units(input_data, desired_units)

    def test_check_units_with_zero_value(self):
        """Test check_units with zero value."""
        input_data = 0.0 * u.m
        desired_units = u.km

        result = check_units(input_data, desired_units)

        assert result.unit == u.km
        assert result.value == 0.0

    def test_check_units_with_negative_value(self):
        """Test check_units with negative value."""
        input_data = -5.0 * u.m
        desired_units = u.cm

        result = check_units(input_data, desired_units)

        assert result.unit == u.cm
        assert result.value == -500.0

    def test_check_units_with_large_numbers(self):
        """Test check_units with large numbers."""
        input_data = 1e6 * u.m
        desired_units = u.km

        result = check_units(input_data, desired_units)

        assert result.unit == u.km
        assert np.isclose(result.value, 1000.0)

    def test_check_units_with_small_numbers(self):
        """Test check_units with small numbers."""
        input_data = 1e-9 * u.m
        desired_units = u.nm

        result = check_units(input_data, desired_units)

        assert result.unit == u.nm
        assert np.isclose(result.value, 1.0)


class TestFindKey:
    """Test find_key function for dictionary key searching."""

    def test_find_key_basic_functionality(self):
        """Test basic find_key functionality."""
        # find_key takes input_class_keys (list) and key_list (list)
        input_class_keys = ["key1", "key2", "key3"]
        key_list = ["key1", "missing_key"]

        # Should find the first matching key from key_list in input_class_keys
        result = find_key(input_class_keys, key_list)
        assert result == "key1"

    def test_find_key_case_insensitive(self):
        """Test find_key case insensitive matching."""
        input_class_keys = ["CamelCase", "lowercase", "UPPERCASE"]
        key_list = ["camelcase", "LOWERCASE"]

        # Should find match ignoring case
        result = find_key(input_class_keys, key_list)
        assert result == "CamelCase"

    def test_find_key_missing_key(self):
        """Test find_key when key doesn't exist."""
        input_class_keys = ["key1", "key2", "key3"]
        key_list = ["missing_key", "also_missing"]

        # Should raise KeyError when no matching keys found
        with pytest.raises(KeyError, match="no matching key found"):
            find_key(input_class_keys, key_list)

    def test_find_key_first_match_priority(self):
        """Test find_key returns first match from key_list."""
        input_class_keys = ["alpha", "beta", "gamma"]
        key_list = ["gamma", "beta", "alpha"]  # gamma comes first in key_list

        result = find_key(input_class_keys, key_list)
        assert result == "gamma"

    def test_find_key_empty_lists(self):
        """Test find_key with empty inputs."""
        with pytest.raises(KeyError):
            find_key([], ["some_key"])

        with pytest.raises(KeyError):
            find_key(["some_key"], [])


class TestLookForKey:
    """Test look_for_key function for key-value searching in dictionaries."""

    def test_look_for_key_exact_match(self):
        """Test look_for_key with exact key-value match."""
        test_dict = {"temperature": 300, "pressure": 101325}

        # Should return True when key and value match
        result = look_for_key(test_dict, "temperature", 300)
        assert result is True

        # Should return False when key exists but value doesn't match
        result = look_for_key(test_dict, "temperature", 250)
        assert result is False

    def test_look_for_key_missing_key(self):
        """Test look_for_key with missing key."""
        test_dict = {"temperature": 300, "pressure": 101325}

        # Should return False when key doesn't exist
        result = look_for_key(test_dict, "humidity", 50)
        assert result is False

    def test_look_for_key_nested_dict(self):
        """Test look_for_key with nested dictionaries."""
        test_dict = {
            "level1": {"level2": {"target_key": "target_value"}},
            "other_key": "other_value",
        }

        # Should find key-value pair in nested structure
        result = look_for_key(test_dict, "target_key", "target_value")
        assert result is True

        # Should return False for wrong value in nested structure
        result = look_for_key(test_dict, "target_key", "wrong_value")
        assert result is False

    def test_look_for_key_various_types(self):
        """Test look_for_key with various value types."""
        test_dict = {
            "string_key": "string_value",
            "number_key": 42,
            "list_key": [1, 2, 3],
            "bool_key": True,
            "nested": {"float_key": 3.14, "none_key": None},
        }

        # Test various data types
        assert look_for_key(test_dict, "number_key", 42) is True
        assert look_for_key(test_dict, "bool_key", True) is True
        assert look_for_key(test_dict, "float_key", 3.14) is True
        assert look_for_key(test_dict, "none_key", None) is True

        # Test list comparison
        assert look_for_key(test_dict, "list_key", [1, 2, 3]) is True
        assert look_for_key(test_dict, "list_key", [1, 2, 4]) is False


class TestUtilsChecksIntegration:
    """Integration tests for checks module utilities."""

    def test_unit_conversions_realistic_scenarios(self):
        """Test realistic unit conversion scenarios."""
        # Wavelength conversions common in astronomy
        wavelength_angstrom = 5000 * u.AA  # Typical optical wavelength
        result_nm = check_units(wavelength_angstrom, u.nm)
        assert np.isclose(result_nm.value, 500.0)

        # Distance conversions
        distance_au = 1.0 * u.au
        result_km = check_units(distance_au, u.km)
        assert result_km.value > 1e8  # AU is large in km

    def test_temperature_conversions(self):
        """Test temperature unit conversions."""
        # Test that temperature conversions properly raise errors
        # astropy doesn't support direct K to deg_C conversion
        temp_kelvin = 293.15 * u.K

        # This should raise UnitConversionError for incompatible temperature units
        with pytest.raises(
            u.UnitConversionError, match="impossible to convert K into deg_C"
        ):
            check_units(temp_kelvin, u.deg_C)

    def test_spectral_units_conversion(self):
        """Test spectral quantity conversions."""
        # Frequency to wavelength type conversions are complex
        # Test basic frequency unit conversion
        freq_hz = 1e14 * u.Hz
        freq_ghz = check_units(freq_hz, u.GHz)
        assert np.isclose(freq_ghz.value, 1e5)
