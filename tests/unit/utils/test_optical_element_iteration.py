"""
Unit tests for optical element iteration utilities.

Tests functions that iterate over optical elements in configuration dictionaries,
used for applying parameters across multiple optical elements in the system.
"""

from collections import OrderedDict

from exosim.utils.iterators import _nested, iterate_over_optical_elements


class TestOpticalElementIteration:
    """Test optical element iteration functionality."""

    def test_missing_key_returns_unchanged_dict(self):
        """Test when key is not in input dictionary."""
        input_dict = {"other_key": {"some": "data"}}
        key = "missing_key"
        last_key = "test_key"
        val = "test_value"

        result = iterate_over_optical_elements(input_dict, key, last_key, val)

        # Should return input unchanged
        assert result == input_dict
        assert result is input_dict  # Same object

    def test_key_with_optical_path_structure(self):
        """Test when key contains optical_path structure."""
        input_dict = {
            "test_key": {
                "optical_path": {"opticalElement": {"element1": {"existing": "value"}}}
            }
        }
        key = "test_key"
        last_key = "new_param"
        val = "new_value"

        result = iterate_over_optical_elements(input_dict, key, last_key, val)

        # Check that the value was added to the optical element
        assert (
            result["test_key"]["optical_path"]["opticalElement"]["element1"][
                "new_param"
            ]
            == "new_value"
        )
        assert (
            result["test_key"]["optical_path"]["opticalElement"]["element1"]["existing"]
            == "value"
        )

    def test_key_with_multiple_channels_via_ordered_dict(self):
        """Test when key contains OrderedDict with multiple channels."""
        input_dict = {
            "test_key": OrderedDict(
                [
                    ("channel1", {"optical_path": {"opticalElement": {"elem1": {}}}}),
                    ("channel2", {"optical_path": {"opticalElement": {"elem2": {}}}}),
                ]
            )
        }
        key = "test_key"
        last_key = "param"
        val = "value"

        result = iterate_over_optical_elements(input_dict, key, last_key, val)

        # Check that both channels were updated
        assert (
            result["test_key"]["channel1"]["optical_path"]["opticalElement"]["elem1"][
                "param"
            ]
            == "value"
        )
        assert (
            result["test_key"]["channel2"]["optical_path"]["opticalElement"]["elem2"][
                "param"
            ]
            == "value"
        )

    def test_channels_without_optical_path_wrapper(self):
        """Test when OrderedDict channels have opticalElement directly."""
        input_dict = {
            "test_key": OrderedDict(
                [
                    ("channel1", {"opticalElement": {"elem1": {}}}),
                    ("channel2", {"opticalElement": {"elem2": {}}}),
                ]
            )
        }
        key = "test_key"
        last_key = "param"
        val = "value"

        result = iterate_over_optical_elements(input_dict, key, last_key, val)

        # Check that both channels were updated
        assert (
            result["test_key"]["channel1"]["opticalElement"]["elem1"]["param"]
            == "value"
        )
        assert (
            result["test_key"]["channel2"]["opticalElement"]["elem2"]["param"]
            == "value"
        )

    def test_direct_optical_element_structure(self):
        """Test when key directly contains opticalElement."""
        input_dict = {"test_key": {"opticalElement": {"elem1": {"existing": "data"}}}}
        key = "test_key"
        last_key = "new_param"
        val = "new_value"

        result = iterate_over_optical_elements(input_dict, key, last_key, val)

        # Check that the value was added
        assert result["test_key"]["opticalElement"]["elem1"]["new_param"] == "new_value"
        assert result["test_key"]["opticalElement"]["elem1"]["existing"] == "data"


class TestNestedHelperFunction:
    """Test the _nested helper function for optical element processing."""

    def test_nested_with_ordered_dict_optical_elements(self):
        """Test _nested when opticalElement contains OrderedDict."""
        input_dict = {
            "opticalElement": OrderedDict(
                [("elem1", {"existing": "value1"}), ("elem2", {"existing": "value2"})]
            )
        }
        key = "new_param"
        val = "new_value"

        _nested(input_dict, key, val)

        # Check that both elements were updated
        assert input_dict["opticalElement"]["elem1"]["new_param"] == "new_value"
        assert input_dict["opticalElement"]["elem2"]["new_param"] == "new_value"
        assert input_dict["opticalElement"]["elem1"]["existing"] == "value1"
        assert input_dict["opticalElement"]["elem2"]["existing"] == "value2"

    def test_nested_with_single_optical_element(self):
        """Test _nested when opticalElement is a simple dictionary."""
        input_dict = {"opticalElement": {"existing": "value"}}
        key = "new_param"
        val = "new_value"

        _nested(input_dict, key, val)

        # Check that the element was updated
        assert input_dict["opticalElement"]["new_param"] == "new_value"
        assert input_dict["opticalElement"]["existing"] == "value"
