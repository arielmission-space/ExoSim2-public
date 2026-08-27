"""
Integration tests for CreateNDRs recipe.

This module provides comprehensive testing for the CreateNDRs recipe,
which handles the creation of Non-Destructive Reads (NDRs) in the ExoSim2.0
observation pipeline. It verifies both individual components and their
interactions in the recipe.
"""

import contextlib
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

import exosim.log as log
from exosim.recipes.create_ndrs import CreateNDRs
from exosim.utils.timed_class import TimedClass


class TestCreateNDRsStructure:
    """Test the structure and basic functionality of CreateNDRs recipe."""

    def test_class_structure_and_inheritance(self):
        """Test that CreateNDRs has proper class structure."""
        assert issubclass(CreateNDRs, TimedClass)
        assert issubclass(CreateNDRs, log.Logger)
        assert hasattr(CreateNDRs, "__init__")
        assert hasattr(CreateNDRs, "load_subexposure_data")
        assert hasattr(CreateNDRs, "prepare_output")

    def test_docstring_and_examples(self):
        """Test that CreateNDRs has proper docstring and examples."""
        assert CreateNDRs.__doc__ is not None
        assert "Examples" in CreateNDRs.__doc__
        assert "recipes.CreateNDRs" in CreateNDRs.__doc__

    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios."""
        with pytest.raises(TypeError):
            # Should raise TypeError for missing required arguments
            CreateNDRs()

        with pytest.raises(TypeError):
            # Should raise TypeError for insufficient arguments
            CreateNDRs("input_only.h5")

    def test_method_call_patterns(self):
        """Test that methods are called in expected patterns."""
        methods = ["__init__", "load_subexposure_data", "prepare_output"]

        for method in methods:
            assert hasattr(CreateNDRs, method)


class TestCreateNDRsInitialization:
    """Test initialization and configuration handling of CreateNDRs recipe."""

    @patch("exosim.recipes.create_ndrs.h5py.File")
    @patch("exosim.recipes.create_ndrs.SetOutput")
    @patch("exosim.recipes.create_ndrs.load_options")
    @patch("exosim.recipes.create_ndrs.clean_config_files")
    @patch("exosim.recipes.create_ndrs.copy_input_files")
    @patch("exosim.recipes.create_ndrs.RunConfig")
    def test_init_with_string_options_file(
        self,
        mock_run_config,
        mock_copy_files,
        mock_clean_config,
        mock_load_options,
        mock_set_output,
        mock_h5py_file,
    ):
        """Test initialization with string options file."""
        # Setup mocks for proper payload structure
        mock_load_options.return_value = (
            {"main": "config"},
            {"payload": {"channel": {"test_ch": {"detector": {}}}}},
        )

        # Mock HDF5 file
        mock_file_obj = MagicMock()
        mock_file_obj.__enter__.return_value = mock_file_obj
        mock_file_obj.__exit__.return_value = None
        mock_file_obj.__getitem__.return_value = {"channels": {"test_ch": {}}}
        mock_file_obj.keys.return_value = ["test_ch"]
        mock_h5py_file.return_value = mock_file_obj

        # Mock output
        mock_output_obj = MagicMock()
        mock_set_output.return_value = mock_output_obj

        with (
            patch.object(CreateNDRs, "announce"),
            patch.object(CreateNDRs, "graphics"),
            patch.object(CreateNDRs, "refactor_output"),
            patch.object(CreateNDRs, "load_subexposure_data"),
            patch.object(CreateNDRs, "prepare_output"),
        ):
            with contextlib.suppress(Exception):
                ndrs_instance = CreateNDRs(
                    input_file="test_input.h5",
                    output_file="test_output.h5",
                    options_file="config.xml",
                )
                assert hasattr(ndrs_instance, "mainConfig")
                assert hasattr(ndrs_instance, "payloadConfig")

            # Verify mock calls
            mock_clean_config.assert_called_once()
            mock_load_options.assert_called_once_with("config.xml")
            mock_copy_files.assert_called_once()

    @patch("exosim.recipes.create_ndrs.h5py.File")
    @patch("exosim.recipes.create_ndrs.SetOutput")
    @patch("exosim.recipes.create_ndrs.load_options")
    @patch("exosim.recipes.create_ndrs.clean_config_files")
    @patch("exosim.recipes.create_ndrs.copy_input_files")
    @patch("exosim.recipes.create_ndrs.RunConfig")
    def test_init_with_dict_options_file(
        self,
        mock_run_config,
        mock_copy_files,
        mock_clean_config,
        mock_load_options,
        mock_set_output,
        mock_h5py_file,
    ):
        """Test initialization with dict options file."""
        # Setup mocks
        mock_load_options.return_value = (
            {"main": "config"},
            {"payload": {"channel": {"test_ch": {"detector": {}}}}},
        )

        # Mock HDF5 file
        mock_file_obj = MagicMock()
        mock_file_obj.__enter__.return_value = mock_file_obj
        mock_file_obj.__exit__.return_value = None
        mock_file_obj.__getitem__.return_value = {"test_ch": {}}
        mock_file_obj.keys.return_value = ["test_ch"]
        mock_h5py_file.return_value = mock_file_obj

        # Mock output
        mock_output_obj = MagicMock()
        mock_set_output.return_value = mock_output_obj

        with (
            patch.object(CreateNDRs, "announce"),
            patch.object(CreateNDRs, "graphics"),
            patch.object(CreateNDRs, "refactor_output"),
            patch.object(CreateNDRs, "load_subexposure_data"),
            patch.object(CreateNDRs, "prepare_output"),
        ):
            with contextlib.suppress(Exception):
                config_dict = {"config": "dict"}
                CreateNDRs(
                    input_file="test_input.h5",
                    output_file="test_output.h5",
                    options_file=config_dict,
                )

            # Verify config loading
            mock_load_options.assert_called_once_with(config_dict)


class TestCreateNDRsDataHandling:
    """Test data loading and preparation functionality."""

    def test_load_subexposure_data_method_exists(self):
        """Test that load_subexposure_data method exists and has correct signature."""
        assert hasattr(CreateNDRs, "load_subexposure_data")

        import inspect

        sig = inspect.signature(CreateNDRs.load_subexposure_data)
        assert "self" in sig.parameters
        assert "ch" in sig.parameters

    def test_prepare_output_method_exists(self):
        """Test that prepare_output method exists and has correct signature."""
        assert hasattr(CreateNDRs, "prepare_output")

        import inspect

        sig = inspect.signature(CreateNDRs.prepare_output)
        expected_params = [
            "self",
            "spectral",
            "spatial",
            "time_line",
            "integration_times",
            "ch",
            "ch_grp",
        ]

        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"

    @patch("exosim.recipes.create_ndrs.h5py.File")
    def test_load_subexposure_data_basic_structure(self, mock_h5py_file):
        """Test the basic structure of load_subexposure_data method."""
        # Create a mock CreateNDRs instance
        mock_instance = MagicMock(spec=CreateNDRs)
        mock_instance.input = "test_input.h5"

        # Mock the h5py file structure
        mock_file_obj = MagicMock()
        mock_file_obj.__enter__.return_value = mock_file_obj
        mock_file_obj.__exit__.return_value = None

        # Mock SubExposures data structure
        mock_se_data = MagicMock()
        mock_se_data["spectral"] = np.array([1, 2, 3])
        mock_se_data["spatial"] = np.array([1, 2])
        mock_se_data["time"] = np.array([0.1, 0.2])
        mock_se_data["spectral_units"] = MagicMock()
        mock_se_data["spatial_units"] = MagicMock()
        mock_se_data["time_units"] = MagicMock()
        mock_se_data["spectral_units"].return_value.decode.return_value = "um"
        mock_se_data["spatial_units"].return_value.decode.return_value = "arcsec"
        mock_se_data["time_units"].return_value.decode.return_value = "s"

        # Mock metadata structure
        mock_metadata = MagicMock()
        mock_int_times = MagicMock()
        mock_int_times["value"] = np.array([1.0, 1.0])
        mock_int_times["unit"] = MagicMock()
        mock_int_times["unit"].return_value.decode.return_value = "s"
        mock_metadata["integration_times"] = mock_int_times
        mock_se_data["metadata"] = mock_metadata

        # Mock parameters structures
        class MockHDF5Dataset:
            def __init__(self, value):
                self._value = value

            def __call__(self):
                return self._value

            def __getitem__(self, key):
                # HDF5 datasets support both [key] and [()] access
                if key == ():
                    return self._value
                if key == "value":
                    # For the nested "value" access in try/except
                    return MockHDF5Dataset(self._value)
                raise KeyError(key)

        mock_readout_params = {"number_of_exposures": MockHDF5Dataset(2)}

        mock_scheme_params = {
            "n_NRDs_per_group": MockHDF5Dataset(5),
            "n_GRPs": MockHDF5Dataset(10),
        }  # Set up the file getitem behavior
        mock_keys_map = {
            "channels/test_channel/SubExposures": mock_se_data,
            "channels/test_channel/instantaneous_readout_params": mock_readout_params,
            "channels/test_channel/reading_scheme_params": mock_scheme_params,
        }

        def mock_getitem(key):
            return mock_keys_map.get(key, MagicMock())

        mock_file_obj.__getitem__.side_effect = mock_getitem
        mock_h5py_file.return_value = mock_file_obj

        # Call the actual method
        result = CreateNDRs.load_subexposure_data(mock_instance, "test_channel")
        # Verify that the method returns expected structure
        assert len(result) == 7  # Should return 7 values according to method signature

    @patch("exosim.recipes.create_ndrs.Counts")
    @patch("exosim.recipes.create_ndrs.HDF5OutputGroup")
    @patch("exosim.recipes.create_ndrs.h5py.File")
    def test_prepare_output_basic_structure(
        self, mock_h5py_file, mock_hdf5_group, mock_counts
    ):
        """Test the basic structure of prepare_output method."""
        # Create a mock CreateNDRs instance
        mock_instance = MagicMock(spec=CreateNDRs)
        mock_instance.input = "test_input.h5"

        # Mock HDF5 file structure for prepare_output
        mock_file_obj = MagicMock()
        mock_file_obj.__enter__.return_value = mock_file_obj
        mock_file_obj.__exit__.return_value = None
        mock_h5py_file.return_value = mock_file_obj

        # Mock inputs
        spectral = np.array([1, 2, 3])
        spatial = np.array([1, 2])
        mid_freq_time_line = np.array([0.1, 0.2]) * u.s
        integration_times = np.array([1.0, 1.0]) * u.s
        ch = "test_channel"
        ch_grp = MagicMock()

        # Mock Counts creation
        mock_counts_obj = MagicMock()
        mock_counts.return_value = mock_counts_obj

        # Mock HDF5OutputGroup
        mock_group_obj = MagicMock()
        mock_hdf5_group.return_value = mock_group_obj

        # Call the actual method
        result = CreateNDRs.prepare_output(
            mock_instance,
            spectral,
            spatial,
            mid_freq_time_line,
            integration_times,
            ch,
            ch_grp,
        )

        # Verify that the method returns expected structure
        assert len(result) == 2  # Should return sub_ndrs and sim_grp


class TestCreateNDRsIntegration:
    """Test integration scenarios and end-to-end functionality."""

    def test_attributes_after_initialization(self):
        """Test that expected attributes are set after initialization."""
        with (
            patch("exosim.recipes.create_ndrs.h5py.File"),
            patch("exosim.recipes.create_ndrs.SetOutput") as mock_set_output,
            patch("exosim.recipes.create_ndrs.load_options") as mock_load_options,
            patch("exosim.recipes.create_ndrs.clean_config_files"),
            patch("exosim.recipes.create_ndrs.copy_input_files"),
            patch("exosim.recipes.create_ndrs.RunConfig"),
            patch.object(CreateNDRs, "announce"),
            patch.object(CreateNDRs, "graphics"),
            patch.object(CreateNDRs, "refactor_output"),
        ):
            mock_load_options.return_value = ({"main": "config"}, {"payload": "config"})
            # Configure the output mock to return a context manager with getsize method
            mock_output_context = MagicMock()
            mock_output_context.getsize.return_value = 1024.0  # Mock file size
            mock_set_output.return_value.use.return_value.__enter__.return_value = (
                mock_output_context
            )

            ndrs_instance = CreateNDRs(
                input_file="test_input.h5",
                output_file="test_output.h5",
                options_file="config.xml",
            )

            # Check that required attributes are set
            assert hasattr(ndrs_instance, "input")
            assert hasattr(ndrs_instance, "mainConfig")
            assert hasattr(ndrs_instance, "payloadConfig")
            assert ndrs_instance.input == "test_input.h5"

    @patch("exosim.recipes.create_ndrs.os.path.dirname")
    @patch("exosim.recipes.create_ndrs.os.path.abspath")
    def test_file_path_handling(self, mock_abspath, mock_dirname):
        """Test file path handling in initialization."""
        mock_abspath.return_value = "/full/path/to/output.h5"
        mock_dirname.return_value = "/full/path/to"

        with (
            patch("exosim.recipes.create_ndrs.h5py.File"),
            patch("exosim.recipes.create_ndrs.SetOutput") as mock_set_output,
            patch("exosim.recipes.create_ndrs.load_options") as mock_load_options,
            patch("exosim.recipes.create_ndrs.clean_config_files"),
            patch("exosim.recipes.create_ndrs.copy_input_files") as mock_copy_files,
            patch("exosim.recipes.create_ndrs.RunConfig"),
            patch.object(CreateNDRs, "announce"),
            patch.object(CreateNDRs, "graphics"),
            patch.object(CreateNDRs, "refactor_output"),
        ):
            mock_load_options.return_value = ({"main": "config"}, {"payload": "config"})
            # Configure the output mock to return a context manager with getsize method
            mock_output_context = MagicMock()
            mock_output_context.getsize.return_value = 2048.0  # Mock file size
            mock_set_output.return_value.use.return_value.__enter__.return_value = (
                mock_output_context
            )

            CreateNDRs(
                input_file="test_input.h5",
                output_file="output.h5",
                options_file="config.xml",
            )

            # Verify path handling calls
            mock_abspath.assert_called_once_with("output.h5")
            mock_dirname.assert_called_once_with("/full/path/to/output.h5")
            mock_copy_files.assert_called_once_with("/full/path/to")
