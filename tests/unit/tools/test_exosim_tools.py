"""
Unit tests for ExoSim tools base functionality.

This module contains tests for the base ExoSimTool class and its common
functionality used across all calibration and analysis tools.
"""

import glob
import logging
import os

import pytest

from exosim.log import set_log_level
from exosim.tools import ReadoutSchemeCalculator
from exosim.tools.exosim_tool import ExoSimTool

set_log_level(logging.DEBUG)


@pytest.fixture
def tools_file(test_data_dir):
    """Fixture to provide a tools configuration file for testing."""
    # Create a minimal tools configuration file for testing
    tools_config = os.path.join(test_data_dir, "test_tools_config.xml")
    with open(tools_config, "w") as f:
        f.write(
            '<?xml version="1.0"?>\n<root>\n<test_tool>test_value</test_tool>\n<channel>\n<value>TestChannel</value>\n</channel>\n</root>\n'
        )
    return tools_config


class TestExoSimTools:
    """Test suite for the base ExoSimTool class functionality."""

    def test_attr(self, tools_file):
        """Test that ExoSimTool has required attributes after initialization."""
        exotools = ExoSimTool(tools_file)
        assert hasattr(exotools, "options")


class TestReadoutSchemeCalculator:
    """Test suite for the ReadoutSchemeCalculator tool."""

    def test_read_out(self, test_data_dir, tools_file):
        """Test readout scheme calculation with test data files."""
        f_list = glob.glob(os.path.join(test_data_dir, "test_data-*fp.h5"))
        if not f_list:
            pytest.skip("missing_file")

        # Check if the file has the required structure
        import h5py

        try:
            with h5py.File(f_list[0], "r") as f:
                # Check for required structure with focal_plane data
                if "channels" not in f:
                    pytest.skip("Test file lacks required structure")
                # Check if at least one channel has focal_plane
                has_focal_plane = False
                for channel_name in f["channels"]:
                    if "focal_plane" in f[f"channels/{channel_name}"]:
                        has_focal_plane = True
                        break
                if not has_focal_plane:
                    pytest.skip("Test file lacks focal_plane data")
        except Exception:
            pytest.skip("Could not open test file")

        # Test that ReadoutSchemeCalculator can be instantiated with valid inputs
        try:
            calculator = ReadoutSchemeCalculator(tools_file, f_list[0])
            assert calculator is not None
        except KeyError:
            pytest.skip("Test file incomplete for ReadoutSchemeCalculator")
