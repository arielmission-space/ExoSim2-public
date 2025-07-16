import logging
import os.path
import random
from re import search

import astropy.units as u
import numpy as np
import pytest

from exosim.log import Logger, setLogLevel
from exosim.utils import RunConfig
from exosim.utils.binning import rebin
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

setLogLevel(logging.DEBUG)

class TestRandom:
    def test_seed(self):
        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        print(a, b)
        RunConfig.random_seed = 1
        print(RunConfig.random_seed)

        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        assert a == 0.417022004702574
        assert b == 0.13436424411240122

        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        assert a == 0.7203244934421581
        assert b == 0.8474337369372327


class TestRunConfig:
    def test_info(self):
        RunConfig.stats()


class TestBinning:
    def test_duplicates(self):
        xp = [0, 0, 1, 2, 3, 4]
        f = [0, 0, 1, 2, 3, 4]
        x = [0, 1, 1, 2, 3]
        new_f = rebin(x, xp, f)

        xp = [0, 1, 2, 3, 4]
        f = [0, 1, 2, 3, 4]
        x = [0, 1, 2, 3]
        new_f_1 = rebin(x, xp, f)
        np.testing.assert_equal(new_f_1, new_f)

    def test_NaNs(self):
        xp = [0, np.nan, 1, 2, 3, 4]
        f = [0, 0, 1, 2, 3, 4]
        x = [0, 1, np.nan, 2, 3]
        new_f = rebin(x, xp, f)

        xp = [0, 1, 2, 3, 4]
        f = [0, 1, 2, 3, 4]
        x = [0, 1, 2, 3]
        new_f_1 = rebin(x, xp, f)
        np.testing.assert_equal(new_f_1, new_f)


class TestChecks:
    def test_lookfor_key(self):
        dict_ = {"key1": 0, "key2": 2}
        assert look_for_key(dict_, "key2", 2) is True
        assert look_for_key(dict_, "key2", 0) is False

    def test_lookfor_key_nested(self):
        dict_ = {"key1": {"key2": 2}}
        assert look_for_key(dict_, "key2", 2) is True
        assert look_for_key(dict_, "key2", 0) is False

    def test_find_key(self):
        dict_ = {"key1": 1, "key2": 2}
        found = find_key(list(dict_.keys()), ["key1", "key2"])
        assert found == "key1"

    def test_find_key_error(self):
        dict_ = {"key1": 1, "key2": 2}
        with pytest.raises(KeyError):
            find_key(list(dict_.keys()), "key3")


class TestKlassFactory:
    def test_find_klass(self, test_data_dir):
        from exosim.tasks.load import LoadOpticalElement

        file_name = os.path.join(test_data_dir, "loadKlass.py")
        klass = find_klass_in_file(file_name, LoadOpticalElement)
        assert klass.__name__ == "LoadOpticalElementDefault"

    def test_find_klass_error(self, test_data_dir):
        from exosim.tasks.load.loadOptions import LoadOptions

        file_name = os.path.join(test_data_dir, "loadKlass.py")
        with pytest.raises(Exception):
            find_klass_in_file(file_name, LoadOptions)

    def test_extract_klass(self):
        from exosim.tasks.load import LoadOptions
        from exosim.tasks.task import Task

        klass = find_task(LoadOptions, Task)
        assert klass.__name__ == "LoadOptions"

    def test_extract_klass_error(self):
        from exosim.log import Logger
        from exosim.tasks.task import Task

        with pytest.raises(TypeError):
            find_task(Logger, Task)

    def test_load_klass_error(self):
        from exosim.log import Logger
        from exosim.tasks.task import Task

        with pytest.raises(TypeError):
            load_klass(Logger, Task)

        from exosim.tasks.load import LoadOptions

        with pytest.raises(TypeError):
            load_klass(LoadOptions, Task)

    def test_find_and_run_task_error(self):
        from exosim.tasks.load import LoadOpticalElement

        param = {"test": "custom"}
        with pytest.raises(Exception):
            find_and_run_task(param, "test", LoadOpticalElement)


class TestUnits:
    def test_no_unit(self):
        res = check_units(3, "", force=True)
        np.testing.assert_equal(res.value, 3)
        np.testing.assert_equal(res.unit, u.Unit(""))


class TestTimedClass:
    class TestTimed(TimedClass):
        def __init__(self):
            super().__init__()

    def test_logger(self, caplog):
        import logging

        from exosim.log.logger import root_logger

        original_propagate = root_logger.propagate
        root_logger.propagate = True

        try:
            with caplog.at_level(logging.DEBUG):
                test_timed = self.TestTimed()
                test_timed.log_runtime_complete("", "info")
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "INFO"
                assert "exosim.TestTimed" in caplog.records[0].name
                assert ": 00h00m00s" in caplog.records[0].message

                caplog.clear()

                test_timed.log_runtime("", "debug")
                assert len(caplog.records) == 1
                assert caplog.records[0].levelname == "DEBUG"
                assert "exosim.TestTimed" in caplog.records[0].name
                assert ": 00h00m00s" in caplog.records[0].message
        finally:
            root_logger.propagate = original_propagate




class TestOperations:
    def test_sum(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0])
        c = operate_over_axis(a, b, 0, "+")
        np.testing.assert_equal(c, np.array([[2, 3, 4], [4, 5, 6]]))

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0, 1])
        c = operate_over_axis(a, b, 1, "+")
        np.testing.assert_equal(c, np.array([[2, 2, 4], [5, 5, 7]]))

    def test_prod(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0])
        c = operate_over_axis(a, b, 0, "*")
        np.testing.assert_equal(c, np.array([[1, 2, 3], [0, 0, 0]]))

        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 0, 1])
        c = operate_over_axis(a, b, 1, "*")
        np.testing.assert_equal(c, np.array([[1, 0, 3], [4, 0, 6]]))

    def test_wrong_axis(self):
        with pytest.raises(IndexError):
            a = np.array([[1, 2, 3], [4, 5, 6]])
            b = np.array([1, 0])
            operate_over_axis(a, b, 2, "+")

        with pytest.raises(ValueError):
            a = np.array([[1, 2, 3], [4, 5, 6]])
            b = np.array([1, 0])
            operate_over_axis(a, b, 1, "+")




class TestGrids:
    def test_time_grid_with_resolution(self):
        grid = time_grid(0 * u.hr, 2 * u.hr, 0.5 * u.hr)
        expected = np.array([0, 0.5, 1.0, 1.5]) * u.hr
        np.testing.assert_allclose(grid.value, expected.value)
        assert grid.unit == u.hr

    def test_time_grid_without_resolution(self):
        grid = time_grid(1 * u.hr, 3 * u.hr, None)
        expected = np.array([1]) * u.hr
        np.testing.assert_allclose(grid.value, expected.value)
        assert grid.unit == u.hr

    def test_time_grid_accepts_float(self):
        grid = time_grid(0, 1, 0.25)
        expected = np.array([0, 0.25, 0.5, 0.75]) * u.hr
        np.testing.assert_allclose(grid.value, expected.value)
        assert grid.unit == u.hr

    def test_wl_grid_no_bin_width(self):
        wl_min = 1.0 * u.um
        wl_max = 2.0 * u.um
        R = 100
        grid = wl_grid(wl_min, wl_max, R)
        assert isinstance(grid, u.Quantity)
        assert grid.unit == u.um
        assert np.all(grid.value > 0)
        assert np.all(np.diff(grid.value) > 0)  # must be increasing

    def test_wl_grid_with_bin_width(self):
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
        grid = wl_grid(1.0, 2.0, 50)
        assert isinstance(grid, u.Quantity)
        assert grid.unit == u.um
        assert np.all(np.diff(grid.value) > 0)
