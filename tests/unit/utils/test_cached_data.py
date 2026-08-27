"""
Unit tests for cached data functionality and HDF5 operations.

This module contains comprehensive tests for the CachedData class,
including memory management, file operations, dataset handling,
and integration with ExoSim's output system.
"""

import contextlib
import logging
import os
import platform
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from exosim.log import set_log_level
from exosim.models.utils.cached_data import CachedData
from exosim.output import SetOutput

set_log_level(logging.DEBUG)


class TestCachedDataSize:
    """Test suite for basic CachedData size and memory management."""

    @pytest.mark.skipif(platform.system() == "Windows", reason="Windows delete issue")
    def test_create_delete_tmp_file(self):
        """
        Test creation and deletion of temporary cache files.

        This test verifies that CachedData properly manages temporary files,
        creating them when needed and cleaning them up on deletion.
        Note: Skipped on Windows due to file deletion issues.
        """
        cached_data = CachedData(10, 1, 10)
        path = Path(deepcopy(cached_data.fname))
        assert path.is_file() is True

        del cached_data
        assert path.is_file() is False

    def test_data_assign_value(self):
        """
        Test data assignment and retrieval from cached dataset.

        This test verifies that data can be properly assigned to and
        retrieved from the cached dataset with correct values.
        """
        cached_data = CachedData(10, 1, 10)
        data = np.ones((10, 1, 10))
        cached_data.chunked_dataset = data
        np.testing.assert_array_equal(cached_data.chunked_dataset[()], data)


class TestCachedDataNamed:
    """Test suite for named CachedData with file output functionality."""

    def test_create_data(self, test_data_dir):
        """
        Test creation of named cached data with file output.

        This test verifies that CachedData can create named datasets
        in HDF5 files and that the file structure is correct.
        """
        fname = os.path.join(test_data_dir, "test_0.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(fname)

        cached_data = CachedData(10, 1, 10, output=fname, dataset_name="test")
        path = Path(cached_data.fname)

        with h5py.File(fname, "r") as f:
            assert next(iter(f.keys())) == "test"

        assert path.is_file() is True

        # Cleanup
        with contextlib.suppress(Exception):
            os.remove(fname)

    def test_rename_dataset(self, test_data_dir):
        """
        Test renaming of cached datasets.

        This test verifies that cached datasets can be renamed
        and that the HDF5 file structure is updated correctly.
        """
        fname = os.path.join(test_data_dir, "test_1.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(fname)

        cached_data = CachedData(10, 1, 10, output=fname, dataset_name="test")
        new_name = "test_new"
        cached_data.rename_dataset(new_name)

        with h5py.File(fname, "r") as f:
            assert next(iter(f.keys())) == new_name

        with contextlib.suppress(OSError):
            os.remove(fname)

    def test_create_multiple_datasets(self, test_data_dir):
        """
        Test creation of multiple datasets in the same HDF5 file.

        This test verifies that multiple CachedData instances can
        create separate datasets in the same HDF5 file without conflicts.
        """
        fname = os.path.join(test_data_dir, "test_2.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(fname)

        CachedData(10, 1, 10, output=fname, dataset_name="test")
        CachedData(10, 1, 10, output=fname, dataset_name="test1")
        CachedData(10, 1, 10, output=fname, dataset_name="test2")

        with h5py.File(fname, "r") as f:
            assert list(f.keys()) == ["test", "test1", "test2"]

        with contextlib.suppress(OSError):
            os.remove(fname)

    def test_set_value(self, test_data_dir):
        """
        Test setting values in cached datasets and persisting to file.

        This test verifies that data can be written to cached datasets
        and properly persisted to the HDF5 file on disk.
        """
        fname = os.path.join(test_data_dir, "test_3.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(fname)

        cached_data = CachedData(10, 1, 10, output=fname, dataset_name="test")
        data = np.ones((10, 1, 10))

        cached_data.chunked_dataset[:] = data
        cached_data.output.flush()

        with h5py.File(fname, "r") as f:
            np.testing.assert_array_equal(f["test/data"][()], data)

        with contextlib.suppress(OSError):
            os.remove(fname)

    @pytest.mark.skipif(
        "Windows" in os.environ.get("OS", ""), reason="skipped on windows machine"
    )
    def test_use_output(self, test_data_dir):
        """
        Test integration with SetOutput context manager.

        This test verifies that CachedData works properly with ExoSim's
        SetOutput system, including cache management and error handling
        for invalid configurations.
        """
        fname = os.path.join(test_data_dir, "test_4.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(fname)

        # Test successful creation with cached output
        output = SetOutput(fname)
        with output.use(append=True, cache=True) as out:
            cached_data = CachedData(10, 1, 10, output=out, dataset_name="test")

        path = Path(cached_data.fname)
        with h5py.File(fname, "r") as f:
            assert "test" in list(f.keys())

        assert path.is_file() is True

        with contextlib.suppress(OSError):
            os.remove(fname)

        # Test that CachedData works even when cache is disabled
        # (no longer raises an error; logs a warning via structlog instead)
        output = SetOutput(fname)
        cached_data = CachedData(
            10,
            1,
            10,
            output=output.use(append=True, cache=False),
            dataset_name="test",
        )
        # Verify the file was still created with the expected dataset
        with h5py.File(fname, "r") as f:
            assert "test" in list(f.keys())

        # Test error with wrong output class
        output = SetOutput(fname)
        with pytest.raises(IOError, match=r".*"):
            cached_data = CachedData(10, 1, 10, output=output, dataset_name="test")

        # Test successful creation with context manager
        output = SetOutput(fname)
        with output.use(append=True, cache=True) as out:
            cached_data = CachedData(10, 1, 10, output=out, dataset_name="test")
            path = Path(cached_data.fname)
            with h5py.File(fname, "r") as f:
                assert "test" in list(f.keys())
            assert path.is_file() is True

        output.delete()

    @pytest.mark.skipif(
        "Windows" in os.environ.get("OS", ""), reason="skipped on windows machine"
    )
    def test_create_data_path(self, test_data_dir):
        """
        Test creation of cached data with custom HDF5 group paths.

        This test verifies that CachedData can create datasets within
        nested HDF5 group structures using custom output paths.
        """
        fname = os.path.join(test_data_dir, "test_5.h5")
        with contextlib.suppress(FileNotFoundError):
            os.remove(fname)

        cached_data = CachedData(
            10,
            1,
            10,
            output=fname,
            output_path="path/to/data",
            dataset_name="test",
        )
        path = Path(cached_data.fname)

        with h5py.File(fname, "r") as f:
            # Check if dataset exists in nested path structure
            check = False
            try:
                if "test" in list(f["path/to/data"].keys()):
                    check = True
            except KeyError:
                # Alternative nested structure
                if "test" in list(f["path"]["to"]["data"].keys()):
                    check = True

            assert check
            assert path.is_file() is True

        with contextlib.suppress(OSError):
            os.remove(fname)


# Note: The following commented code represents arithmetic operations tests
# that were disabled in the original file. These tests would verify
# CachedData's ability to perform mathematical operations (+, -, *, /, //, etc.)
# with other CachedData instances, numpy arrays, and scalars.
# They are preserved here for potential future implementation.

# class TestCachedDataOperations:
#     """Test suite for arithmetic operations on CachedData objects."""
#
#     def test_sum(self):
#         """Test addition operations with CachedData objects."""
#         # Implementation for testing +=, +, etc.
#         pass
#
#     def test_mul(self):
#         """Test multiplication operations with CachedData objects."""
#         # Implementation for testing *=, *, etc.
#         pass
#
#     def test_sub(self):
#         """Test subtraction operations with CachedData objects."""
#         # Implementation for testing -=, -, etc.
#         pass
#
#     def test_truediv(self):
#         """Test division operations with CachedData objects."""
#         # Implementation for testing /=, /, etc.
#         pass
#
#     def test_floordiv(self):
#         """Test floor division operations with CachedData objects."""
#         # Implementation for testing //=, //, etc.
#         pass
