import logging
import os.path
import random
import unittest
from re import search

import astropy.units as u
import numpy as np
from inputs import test_dir

from exosim.log import Logger
from exosim.log import setLogLevel
from exosim.utils import RunConfig
from exosim.utils.binning import rebin
from exosim.utils.checks import check_units
from exosim.utils.checks import find_key
from exosim.utils.checks import look_for_key
from exosim.utils.klass_factory import find_and_run_task
from exosim.utils.klass_factory import find_klass_in_file
from exosim.utils.klass_factory import find_task
from exosim.utils.klass_factory import load_klass
from exosim.utils.operations import operate_over_axis
from exosim.utils.timed_class import TimedClass

setLogLevel(logging.DEBUG)


class RandomTest(unittest.TestCase):
    def test_seed(self):
        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        print(a, b)
        RunConfig.random_seed = 1
        print(RunConfig.random_seed)

        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        self.assertEqual(a, 0.417022004702574)
        self.assertEqual(b, 0.13436424411240122)

        a = np.random.uniform(0, 1)
        b = random.uniform(0, 1)

        self.assertEqual(a, 0.7203244934421581)
        self.assertEqual(b, 0.8474337369372327)


class SystemInfoTest(unittest.TestCase):
    def test_info(self):
        RunConfig.stats()


class BinningTest(unittest.TestCase):
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


class KeyValTest(unittest.TestCase):
    def test_lookfor_key(self):
        dict_ = {"key1": 0, "key2": 2}
        self.assertTrue(look_for_key(dict_, "key2", 2))
        self.assertFalse(look_for_key(dict_, "key2", 0))

    def test_lookfor_key_nested(self):
        dict_ = {"key1": {"key2": 2}}
        self.assertTrue(look_for_key(dict_, "key2", 2))
        self.assertFalse(look_for_key(dict_, "key2", 0))

    def test_find_key(self):
        dict_ = {"key1": 1, "key2": 2}
        found = find_key(list(dict_.keys()), ["key1", "key2"])
        self.assertEqual(found, "key1")

    def test_find_key_error(self):
        dict_ = {"key1": 1, "key2": 2}

        with self.assertRaises(KeyError):
            find_key(list(dict_.keys()), "key3")


class KlassTest(unittest.TestCase):
    def test_find_klass(self):
        from exosim.tasks.load import LoadOpticalElement

        file_name = os.path.join(test_dir, "loadKlass.py")
        klass = find_klass_in_file(file_name, LoadOpticalElement)
        self.assertEqual(klass.__name__, "LoadOpticalElementDefault")

    def test_find_klass_error(self):
        with self.assertRaises(Exception):
            from exosim.tasks.load.loadOptions import LoadOptions

            file_name = os.path.join(test_dir, "loadKlass.py")
            find_klass_in_file(file_name, LoadOptions)

    def test_extract_klass(self):
        from exosim.tasks.load import LoadOptions
        from exosim.tasks.task import Task

        klass = find_task(LoadOptions, Task)
        self.assertEqual(klass.__name__, "LoadOptions")

    def test_extract_klass_error(self):
        from exosim.log import Logger
        from exosim.tasks.task import Task

        with self.assertRaises(TypeError):
            find_task(Logger, Task)

    def test_load_klass_error(self):
        from exosim.log import Logger
        from exosim.tasks.task import Task

        with self.assertRaises(TypeError):
            load_klass(Logger, Task)

        from exosim.tasks.load import LoadOptions

        with self.assertRaises(TypeError):
            load_klass(LoadOptions, Task)

    def test_find_and_run_task_error(self):
        from exosim.tasks.load import LoadOpticalElement

        param = {"test": "custom"}
        with self.assertRaises(Exception):
            find_and_run_task(param, "test", LoadOpticalElement)


class CheckUnitsTest(unittest.TestCase):
    def test_no_unit(self):
        res = check_units(3, "", force=True)
        np.testing.assert_equal(res.value, 3)
        np.testing.assert_equal(res.unit, u.Unit(""))


class TestTimed(TimedClass, Logger):
    def __init__(self):
        super().__init__()


class TestTimedFalse(TimedClass):
    def __init__(self):
        super().__init__()


class TimedClassTest(unittest.TestCase):
    def test_logger(self):
        with self.assertLogs("exosim", level="DEBUG") as cm:
            test_timed = TestTimed()
            test_timed.log_runtime_complete("", "info")
            self.assertTrue(search("INFO:exosim.TestTimed", cm.output[0]))

            test_timed.log_runtime("", "debug")
            self.assertTrue(search("DEBUG:exosim.TestTimed", cm.output[1]))

    def test_no_logger(self):
        with self.assertLogs("exosim", level="DEBUG") as cm:
            test_timed = TestTimedFalse()
            test_timed.log_runtime_complete("", "info")
            self.assertTrue(
                search(
                    "WARNING:exosim.utils.timed_class.TimedClass", cm.output[0]
                )
            )


class OperationTest(unittest.TestCase):
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
        with self.assertRaises(IndexError):
            a = np.array([[1, 2, 3], [4, 5, 6]])
            b = np.array([1, 0])
            c = operate_over_axis(a, b, 2, "+")

        with self.assertRaises(ValueError):
            a = np.array([[1, 2, 3], [4, 5, 6]])
            b = np.array([1, 0])
            c = operate_over_axis(a, b, 1, "+")
