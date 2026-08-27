"""
Unit tests for configuration loading functionality.

This module contains tests for loading and parsing ExoSim configuration files,
including payload XML files and configuration path handling.
"""

import logging
import os

import pytest

from exosim.log import set_log_level
from exosim.tasks.load.load_options import LoadOptions

set_log_level(logging.DEBUG)


@pytest.fixture
def create_payload_file(example_dir, test_data_dir):
    """
    Fixture to create a temporary payload file for testing.

    Parameters
    ----------
    example_dir : str
        Directory containing example configuration files
    test_data_dir : str
        Directory for temporary test data files

    Yields
    ------
    callable
        Function that creates a payload file with specified source path
    """
    payload_file_path = os.path.join(example_dir, "main_example.xml")
    tmp_file_path = os.path.join(test_data_dir, "payload_test.xml")

    def _create_payload_file(source):
        """
        Create a payload file with modified ConfigPath.

        Parameters
        ----------
        source : str
            Path to use as the new ConfigPath value

        Returns
        -------
        str
            Path to the created temporary payload file
        """
        new_config_path = f"    <ConfigPath> {source}\n"
        import contextlib

        # Clean up any existing temporary file
        with contextlib.suppress(OSError):
            os.remove(tmp_file_path)

        # Create new payload file with modified ConfigPath
        with open(tmp_file_path, "w+") as new_file, open(payload_file_path) as old_file:
            for line in old_file:
                if "<ConfigPath>" in line:
                    new_file.write(new_config_path)
                else:
                    new_file.write(line)
        return tmp_file_path

    yield _create_payload_file

    # Cleanup after test
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)


def test_load_options(create_payload_file, example_dir):
    """
    Test loading configuration options from a payload file.

    This test verifies that the LoadOptions task can successfully load
    and parse a configuration file, returning a valid configuration object.

    Parameters
    ----------
    create_payload_file : callable
        Fixture function to create test payload files
    example_dir : str
        Directory containing example configuration files
    """
    # Initialize the LoadOptions task
    load_option = LoadOptions()

    # Create a test payload file pointing to the example directory
    payload_file_path = create_payload_file(source=example_dir)

    # Load the configuration from the payload file
    config = load_option(filename=payload_file_path)

    # Verify that a configuration was successfully loaded
    assert config is not None

    # Additional verification that the config object has expected structure
    # (These checks depend on the structure of the example configuration)
    assert hasattr(config, "keys") or hasattr(config, "__getitem__")
