import logging
import os
import platform
import unittest
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from inputs import test_dir

from exosim.log import setLogLevel
from exosim.models.utils.cachedData import CachedData
from exosim.output import SetOutput

setLogLevel(logging.DEBUG)


class CachedDataSizeTest(unittest.TestCase):
    @unittest.skipIf(platform.system(), "Windows")
    def test_create_delete_tmp_file(self):
        # TODO to solve the file delete issue for windows
        from pathlib import Path

        cachedData = CachedData(10, 1, 10)
        path = Path(deepcopy(cachedData.fname.name))
        self.assertEqual(path.is_file(), True)
        del cachedData
        self.assertEqual(path.is_file(), False)

    def test_data_assign_value(self):
        cachedData = CachedData(10, 1, 10)
        data = np.ones((10, 1, 10))
        cachedData.chunked_dataset = data
        np.testing.assert_array_equal(cachedData.chunked_dataset[()], data)


# class CachedDataOperationsTest(unittest.TestCase):
#
#     def test_sum(self):
#         cachedData1 = CachedData(10, 1, 10)
#         cachedData2 = CachedData(10, 1, 10)
#         data = np.ones((10, 1, 10))
#         cachedData1.dataset = data
#         cachedData2.dataset = data
#
#         cachedData1 += cachedData2
#         np.testing.assert_array_equal(cachedData1.dataset[()], data * 2)
#
#         cachedData3 = cachedData1 + cachedData2
#         np.testing.assert_array_equal(cachedData3.dataset[()], data * 3)
#
#         cachedData4 = cachedData3 + data
#         np.testing.assert_array_equal(cachedData4.dataset[()], data * 4)
#
#         cachedData5 = cachedData4 + 1
#         np.testing.assert_array_equal(cachedData5.dataset[()], data * 5)
#
#         cachedData4 = data + cachedData3
#         np.testing.assert_array_equal(cachedData4.dataset[()], data * 4)
#
#         cachedData5 = 1 + cachedData4
#         np.testing.assert_array_equal(cachedData5.dataset[()], data * 5)
#
#     def test_mul(self):
#         cachedData1 = CachedData(10, 1, 10)
#         cachedData2 = CachedData(10, 1, 10)
#         data = np.ones((10, 1, 10))
#         cachedData1.dataset = data
#         cachedData2.dataset = data + 1
#
#         cachedData1 *= cachedData2
#         np.testing.assert_array_equal(cachedData1.dataset[()], data * 2)
#
#         cachedData3 = cachedData1 * cachedData2
#         np.testing.assert_array_equal(cachedData3.dataset[()], data * 4)
#
#         cachedData4 = cachedData3 * (data + 1)
#         np.testing.assert_array_equal(cachedData4.dataset[()], data * 8)
#
#         cachedData5 = cachedData4 * 2
#         np.testing.assert_array_equal(cachedData5.dataset[()], data * 16)
#
#         cachedData4 = (data + 1) * cachedData3
#         np.testing.assert_array_equal(cachedData4.dataset[()], data * 8)
#
#         cachedData5 = 2 * cachedData4
#         np.testing.assert_array_equal(cachedData5.dataset[()], data * 16)
#
#     def test_sub(self):
#         cachedData1 = CachedData(10, 1, 10)
#         cachedData2 = CachedData(10, 1, 10)
#         data = np.ones((10, 1, 10))
#         cachedData1.dataset = data
#         cachedData2.dataset = data
#
#         cachedData1 -= cachedData2
#         np.testing.assert_array_equal(cachedData1.dataset[()], data * 0)
#
#         cachedData3 = cachedData1 - cachedData2
#         np.testing.assert_array_equal(cachedData3.dataset[()], data * (-1))
#
#         cachedData4 = cachedData3 - data
#         np.testing.assert_array_equal(cachedData4.dataset[()], data * (-2))
#
#         cachedData5 = cachedData4 - 1
#         np.testing.assert_array_equal(cachedData5.dataset[()], data * (-3))
#
#         cachedData4 = data - cachedData2
#         np.testing.assert_array_equal(cachedData4.dataset[()], data * (0))
#
#         cachedData5 = 1 - cachedData2
#         np.testing.assert_array_equal(cachedData5.dataset[()], data * (0))
#
#     def test_truediv(self):
#         cachedData1 = CachedData(10, 1, 10)
#         cachedData2 = CachedData(10, 1, 10)
#         data = np.ones((10, 1, 10))
#         cachedData1.dataset = data * 3
#         cachedData2.dataset = data * 2
#
#         cachedData1 /= cachedData2
#         np.testing.assert_array_equal(cachedData1.dataset[()], data * 3 / 2)
#
#         cachedData1.dataset = data * 3
#         cachedData3 = cachedData1 / cachedData2
#         np.testing.assert_array_equal(cachedData3.dataset[()], data * 3 / 2)
#
#         cachedData4 = cachedData3 / (data * 3)
#         np.testing.assert_array_equal(cachedData4.dataset[()], data / 2)
#
#         cachedData5 = cachedData4 / 2
#         np.testing.assert_array_equal(cachedData5.dataset[()], data / 4)
#
#         cachedData4 = data / cachedData2
#         np.testing.assert_array_equal(cachedData4.dataset[()], data / 2)
#
#         cachedData5 = 1 / cachedData2
#         np.testing.assert_array_equal(cachedData5.dataset[()], data / 2)
#
#     def test_floordiv(self):
#         cachedData1 = CachedData(10, 1, 10)
#         cachedData2 = CachedData(10, 1, 10)
#         data = np.ones((10, 1, 10))
#         cachedData1.dataset = data * 3
#         cachedData2.dataset = data * 2
#
#         cachedData1 //= cachedData2
#         np.testing.assert_array_equal(cachedData1.dataset[()], data * 3 // 2)
#
#         cachedData1.dataset = data * 3
#         cachedData3 = cachedData1 // cachedData2
#         np.testing.assert_array_equal(cachedData3.dataset[()], data * 3 // 2)
#
#         cachedData4 = cachedData3 // (data * 3)
#         np.testing.assert_array_equal(cachedData4.dataset[()], data // 2)
#
#         cachedData5 = cachedData4 // 2
#         np.testing.assert_array_equal(cachedData5.dataset[()], data // 4)
#
#         cachedData4 = data // cachedData2
#         np.testing.assert_array_equal(cachedData4.dataset[()], data // 2)
#
#         cachedData5 = 1 // cachedData2
#         np.testing.assert_array_equal(cachedData5.dataset[()], data // 2)


class CachedDataNamedTest(unittest.TestCase):
    def test_create_data(self):
        fname = os.path.join(test_dir, "test_0.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        path = Path(cachedData.fname)
        f = h5py.File(fname, "r")
        self.assertEqual(list(f.keys())[0], "test")
        f.close()
        self.assertEqual(path.is_file(), True)

        # TODO deleter remove it when it close the tasks.
        # del cachedData
        # self.assertEquals(path.is_file(), False)
        try:
            os.remove(fname)
        except:
            pass

    def test_rename_dataset(self):
        fname = os.path.join(test_dir, "test_1.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        new_name = "test_new"
        cachedData.rename_dataset(new_name)
        f = h5py.File(fname, "r")
        self.assertEqual(list(f.keys())[0], new_name)
        f.close()
        try:
            os.remove(fname)
        except:
            pass

    def test_create_moredataset(self):
        fname = os.path.join(test_dir, "test_2.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        cachedData1 = CachedData(10, 1, 10, output=fname, dataset_name="test1")
        cachedData2 = CachedData(10, 1, 10, output=fname, dataset_name="test2")

        f = h5py.File(fname, "r")
        self.assertListEqual(list(f.keys()), ["test", "test1", "test2"])
        f.close()
        try:
            os.remove(fname)
        except:
            pass

    def test_set_value(self):
        fname = os.path.join(test_dir, "test_3.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        data = np.ones((10, 1, 10))

        cachedData.chunked_dataset[:] = data
        cachedData.output.flush()

        f = h5py.File(fname, "r")
        np.testing.assert_array_equal(f["test/data"][()], data)
        f.close()
        try:
            os.remove(fname)
        except:
            pass

    @unittest.skipIf(
        "Windows" in os.environ.get("OS", ""), "skipped on windows machine"
    )
    def test_use_Output(self):
        fname = os.path.join(test_dir, "test_4.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        output = SetOutput(fname)
        with output.use(append=True, cache=True) as out:
            cachedData = CachedData(10, 1, 10, output=out, dataset_name="test")
        path = Path(cachedData.fname)
        f = h5py.File(fname, "r")
        self.assertTrue("test" in list(f.keys()))
        f.close()
        self.assertEqual(path.is_file(), True)

        # TODO deleter remove it when it close the tasks.
        #        del cachedData
        #        self.assertEquals(path.is_file(), False)

        try:
            os.remove(fname)
        except:
            pass

        # test not cachable file
        with self.assertRaises(IOError):
            output = SetOutput(fname)
            cachedData = CachedData(
                10,
                1,
                10,
                output=output.use(append=True, cache=False),
                dataset_name="test",
            )

        # test wrong class outut file
        with self.assertRaises(IOError):
            output = SetOutput(fname)
            cachedData = CachedData(
                10, 1, 10, output=output, dataset_name="test"
            )

        output = SetOutput(fname)
        with output.use(append=True, cache=True) as out:
            cachedData = CachedData(10, 1, 10, output=out, dataset_name="test")
            path = Path(cachedData.fname)
            f = h5py.File(fname, "r")
            self.assertTrue("test" in list(f.keys()))
            f.close()
            self.assertEqual(path.is_file(), True)

            # TODO deleter remove it when it close the tasks.
        #            del cachedData
        #            self.assertEquals(path.is_file(), False)
        # TODO to solve the file delete issue for windows

        output.delete()

    @unittest.skipIf(
        "Windows" in os.environ.get("OS", ""), "skipped on windows machine"
    )
    def test_create_data_path(self):
        fname = os.path.join(test_dir, "test_5.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(
            10,
            1,
            10,
            output=fname,
            output_path="path/to/data",
            dataset_name="test",
        )
        path = Path(cachedData.fname)
        print(path)
        # TODO to solve the path error for windows

        with h5py.File(fname, "r") as f:
            check = False
            if "test" in list(f["path/to/data"].keys()):
                check = True
            if "test" in list(f["path"]["to"]["data"].keys()):
                check = True
            self.assertTrue(check)
            f.close()
            self.assertEqual(path.is_file(), True)

        try:
            os.remove(fname)
        except:
            pass
