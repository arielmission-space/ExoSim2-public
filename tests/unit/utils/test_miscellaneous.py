"""
Unit tests for miscellaneous utility functions.

This module contains tests for various utility functions including:
- Dictionary key checks and lookups
- Class factory and dynamic loading
- Unit handling and validation
- Timing functionality
- Array operations
- Grid generation
"""

import logging
import os.path
import time

import astropy.units as u
import numpy as np
import pytest

from exosim.log import Logger
from exosim.utils.checks import check_units, find_key, look_for_key
from exosim.utils.grids import time_grid, wl_grid
from exosim.utils.klass_factory import (
    find_and_run_task,
    find_klass_in_file,
    find_task,
    load_klass,
)
from exosim.utils.operations import operate_over_axis
from exosim.utils.timed_class import TimedClass


class TestChecks:
    """Test suite for dictionary and data validation utilities."""

    def test_lookfor_key(self):
        """Test look_for_key function with simple dictionary."""
        dict_ = {"key1": 0, "key2": 2}
        assert look_for_key(dict_, "key2", 2) is True
        assert look_for_key(dict_, "key2", 0) is False

    def test_lookfor_key_nested(self):
        """Test look_for_key function with nested dictionary."""
        dict_ = {"key1": {"key2": 2}}
        assert look_for_key(dict_, "key2", 2) is True
        assert look_for_key(dict_, "key2", 0) is False

    def test_find_key(self):
        """Test find_key function for finding first match."""
        dict_ = {"key1": 1, "key2": 2}
        found = find_key(list(dict_.keys()), ["key1", "key2"])
        assert found == "key1"

    def test_find_key_error(self):
        """Test find_key function error handling."""
        dict_ = {"key1": 1, "key2": 2}
        with pytest.raises(KeyError):
            find_key(list(dict_.keys()), "key3")


class TestKlassFactory:
    """Test suite for dynamic class loading and factory utilities."""

    def test_find_klass(self, test_data_dir):
        """Test finding a class in a file."""
        from exosim.tasks.load import LoadOpticalElement

        file_name = os.path.join(test_data_dir, "loadKlass.py")
        klass = find_klass_in_file(file_name, LoadOpticalElement)
        assert klass.__name__ == "LoadOpticalElementDefault"

    def test_find_klass_error(self, test_data_dir):
        """Test find_klass_in_file error handling."""
        from exosim.tasks.load.load_options import LoadOptions

        file_name = os.path.join(test_data_dir, "loadKlass.py")
        with pytest.raises(ImportError):
            find_klass_in_file(file_name, LoadOptions)

    def test_extract_klass(self):
        """Test extracting a class that inherits from base class."""
        from exosim.tasks.load import LoadOptions
        from exosim.tasks.task import Task

        klass = find_task(LoadOptions, Task)
        assert klass.__name__ == "LoadOptions"

    def test_extract_klass_error(self):
        """Test extract_klass error handling for invalid inheritance."""
        from exosim.tasks.task import Task

        with pytest.raises(TypeError):
            find_task(Logger, Task)

    def test_load_klass_error(self):
        """Test load_klass error handling."""
        from exosim.tasks.task import Task

        with pytest.raises(TypeError):
            load_klass(Logger, Task)

        from exosim.tasks.load import LoadOptions

        with pytest.raises(TypeError):
            load_klass(LoadOptions, Task)

    def test_find_and_run_task_error(self):
        """Test find_and_run_task error handling."""
        from exosim.tasks.load import LoadOpticalElement

        param = {"test": "custom"}
        with pytest.raises(TypeError):
            find_and_run_task(param, "test", LoadOpticalElement)


class TestUnits:
    """Test suite for unit handling utilities."""

    def test_no_unit(self):
        """Test check_units function with dimensionless quantities."""
        res = check_units(3, "", force=True)
        np.testing.assert_equal(res.value, 3)
        np.testing.assert_equal(res.unit, u.Unit(""))


class TestTimedClass:
    """Test suite for timing functionality in base classes."""

    class TimedForTesting(TimedClass):
        """Mock class for testing TimedClass functionality."""

        def __init__(self):
            super().__init__()

    def test_logger(self, caplog):
        """Test logging functionality in TimedClass."""

        # Use standard logging for testing caplog compatibility. The "exosim"
        # logger keeps whatever level an earlier test left it at, so force it to
        # DEBUG (and restore it) rather than relying on suite ordering.
        root_logger = logging.getLogger("exosim")
        original_propagate = root_logger.propagate
        original_level = root_logger.level
        root_logger.propagate = True
        root_logger.setLevel(logging.DEBUG)

        try:
            with caplog.at_level(logging.DEBUG, logger="exosim"):
                test_timed = self.TimedForTesting()
                test_timed.log_runtime_complete("", "info")
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "INFO"
                assert "exosim.TimedForTesting" in caplog.records[0].name
                assert ": 00h00m00s" in caplog.records[0].message

                caplog.clear()

                test_timed.log_runtime("", "debug")
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "DEBUG"
                assert "exosim.TimedForTesting" in caplog.records[0].name
                assert ": 00h00m00s" in caplog.records[0].message
        finally:
            root_logger.propagate = original_propagate
            root_logger.setLevel(original_level)

    def test_timing_attributes_initialization(self):
        """Test that TimedClass initializes timing attributes correctly."""
        before = time.time()
        test_timed = self.TimedForTesting()
        after = time.time()

        # Check that start_time_gen and start_time are initialized
        assert hasattr(test_timed, "start_time_gen")
        assert hasattr(test_timed, "start_time")

        # Should be initialized to approximately current time
        assert before <= test_timed.start_time_gen <= after
        assert before <= test_timed.start_time <= after

        # Both should be approximately equal initially (allow for small timing differences)
        assert abs(test_timed.start_time_gen - test_timed.start_time) < 0.01

    def test_log_runtime_updates_start_time(self, caplog):
        """Test that log_runtime updates start_time but not start_time_gen."""

        # Use standard logging for testing caplog compatibility
        root_logger = logging.getLogger("exosim")
        original_propagate = root_logger.propagate
        root_logger.propagate = True

        try:
            test_timed = self.TimedForTesting()
            original_start_time = test_timed.start_time
            original_start_time_gen = test_timed.start_time_gen

            # Small delay
            time.sleep(0.01)

            with caplog.at_level(logging.INFO):
                test_timed.log_runtime("test operation", "info")

            # start_time should be updated, start_time_gen should not
            assert test_timed.start_time > original_start_time
            assert test_timed.start_time_gen == original_start_time_gen

        finally:
            root_logger.propagate = original_propagate

    def test_log_runtime_complete_uses_start_time_gen(self, caplog):
        """Test that log_runtime_complete uses start_time_gen."""

        # Use standard logging for testing caplog compatibility
        root_logger = logging.getLogger("exosim")
        original_propagate = root_logger.propagate
        root_logger.propagate = True

        try:
            test_timed = self.TimedForTesting()

            # Small delay then log_runtime (which updates start_time)
            time.sleep(0.01)
            test_timed.log_runtime("intermediate", "info")

            # Another delay then log_runtime_complete
            time.sleep(0.01)

            with caplog.at_level(logging.INFO):
                test_timed.log_runtime_complete("complete", "info")

                # Should show total time from initialization, not from last log_runtime
                # May include previous log_runtime message, so check for at least one
                assert len(caplog.records) >= 1
                # Check the last log message for the complete message
                message = caplog.records[-1].message
                assert "complete" in message
                # Total time should be longer than just the last sleep

        finally:
            root_logger.propagate = original_propagate

    def test_invalid_log_level_handling(self, caplog):
        """Test handling of invalid log levels."""

        # Use standard logging for testing caplog compatibility
        root_logger = logging.getLogger("exosim")
        original_propagate = root_logger.propagate
        root_logger.propagate = True

        try:
            test_timed = self.TimedForTesting()

            with caplog.at_level(logging.WARNING):
                test_timed.log_runtime("test", "nonexistent_level")

                # Should generate a warning about missing method
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "WARNING"
                assert (
                    "calling class has no Logger's methods" in caplog.records[0].message
                )

        finally:
            root_logger.propagate = original_propagate

    def test_inheritance_from_logger(self):
        """Test that TimedClass properly inherits from Logger."""
        test_timed = self.TimedForTesting()

        # Should have all Logger methods
        assert hasattr(test_timed, "info")
        assert hasattr(test_timed, "debug")
        assert hasattr(test_timed, "warning")
        assert hasattr(test_timed, "error")

        # Should be instance of Logger
        assert isinstance(test_timed, Logger)


class TestOperations:
    """Test suite for array operation utilities."""

    def test_sum(self):
        """Test operate_over_axis function with addition."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0])
        c = operate_over_axis(a, b, 0, "+")
        np.testing.assert_equal(c, np.array([[2, 3, 4], [4, 5, 6]]))

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0, 1])
        c = operate_over_axis(a, b, 1, "+")
        np.testing.assert_equal(c, np.array([[2, 2, 4], [5, 5, 7]]))

    def test_prod(self):
        """Test operate_over_axis function with multiplication."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0])
        c = operate_over_axis(a, b, 0, "*")
        np.testing.assert_equal(c, np.array([[1, 2, 3], [0, 0, 0]]))

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0, 1])
        c = operate_over_axis(a, b, 1, "*")
        np.testing.assert_equal(c, np.array([[1, 0, 3], [4, 0, 6]]))

    def test_wrong_axis(self):
        """Test operate_over_axis error handling for invalid axis."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0])
        with pytest.raises(IndexError):
            operate_over_axis(a, b, 2, "+")

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0])
        with pytest.raises(ValueError, match=r".*"):
            operate_over_axis(a, b, 1, "+")


class TestGrids:
    """Test suite for grid generation utilities."""

    def test_time_grid_with_resolution(self):
        """Test time grid generation with specified resolution."""
        grid = time_grid(0 * u.hr, 2 * u.hr, 0.5 * u.hr)
        expected = np.array([0, 0.5, 1.0, 1.5]) * u.hr
        np.testing.assert_allclose(grid.value, expected.value)
        assert grid.unit == u.hr

    def test_time_grid_without_resolution(self):
        """Test time grid generation without resolution (single point)."""
        grid = time_grid(1 * u.hr, 3 * u.hr, None)
        expected = np.array([1]) * u.hr
        np.testing.assert_allclose(grid.value, expected.value)
        assert grid.unit == u.hr

    def test_time_grid_accepts_float(self):
        """Test time grid generation with float inputs."""
        grid = time_grid(0, 1, 0.25)
        expected = np.array([0, 0.25, 0.5, 0.75]) * u.hr
        np.testing.assert_allclose(grid.value, expected.value)
        assert grid.unit == u.hr

    def test_wl_grid_no_bin_width(self):
        """Test wavelength grid generation without returning bin widths."""
        wl_min = 1.0 * u.um
        wl_max = 2.0 * u.um
        R = 100
        grid = wl_grid(wl_min, wl_max, R)
        assert isinstance(grid, u.Quantity)
        assert grid.unit == u.um
        assert np.all(grid.value > 0)
        assert np.all(np.diff(grid.value) > 0)  # must be increasing

    def test_wl_grid_with_bin_width(self):
        """Test wavelength grid generation with bin width output."""
        wl_min = 1.0 * u.um
        wl_max = 2.0 * u.um
        R = 100
        grid, bin_width = wl_grid(wl_min, wl_max, R, return_bin_width=True)
        assert isinstance(grid, u.Quantity)
        assert isinstance(bin_width, u.Quantity)
        assert grid.unit == u.um
        assert bin_width.unit == u.um
        assert len(grid) == len(bin_width)

    def test_wl_grid_accepts_float(self):
        """Test wavelength grid generation with float inputs."""
        grid = wl_grid(1.0, 2.0, 50)
        assert isinstance(grid, u.Quantity)
        assert grid.unit == u.um
        assert np.all(np.diff(grid.value) > 0)
