import logging
import os
import platform
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from exosim.log import setLogLevel
from exosim.models.utils.cachedData import CachedData
from exosim.output import SetOutput

setLogLevel(logging.DEBUG)


class TestCachedDataSize:
    @pytest.mark.skipif(platform.system() == "Windows", reason="Windows delete issue")
    def test_create_delete_tmp_file(self):
        # TODO to solve the file delete issue for windows
        cachedData = CachedData(10, 1, 10)
        path = Path(deepcopy(cachedData.fname.name))
        assert path.is_file() is True
        del cachedData
        assert path.is_file() is False

    def test_data_assign_value(self):
        cachedData = CachedData(10, 1, 10)
        data = np.ones((10, 1, 10))
        cachedData.chunked_dataset = data
        np.testing.assert_array_equal(cachedData.chunked_dataset[()], data)


# class TestCachedDataOperations:
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


class TestCachedDataNamed:
    def test_create_data(self, test_data_dir):
        fname = os.path.join(test_data_dir, "test_0.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        path = Path(cachedData.fname)
        f = h5py.File(fname, "r")
        assert list(f.keys())[0] == "test"
        f.close()
        assert path.is_file() is True

        # TODO deleter remove it when it close the tasks.
        # del cachedData
        # assert path.is_file() is False
        try:
            os.remove(fname)
        except:
            pass

    def test_rename_dataset(self, test_data_dir):
        fname = os.path.join(test_data_dir, "test_1.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        new_name = "test_new"
        cachedData.rename_dataset(new_name)
        f = h5py.File(fname, "r")
        assert list(f.keys())[0] == new_name
        f.close()
        try:
            os.remove(fname)
        except:
            pass

    def test_create_moredataset(self, test_data_dir):
        fname = os.path.join(test_data_dir, "test_2.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        cachedData = CachedData(10, 1, 10, output=fname, dataset_name="test")
        cachedData1 = CachedData(10, 1, 10, output=fname, dataset_name="test1")
        cachedData2 = CachedData(10, 1, 10, output=fname, dataset_name="test2")

        f = h5py.File(fname, "r")
        assert list(f.keys()) == ["test", "test1", "test2"]
        f.close()
        try:
            os.remove(fname)
        except:
            pass

    def test_set_value(self, test_data_dir):
        fname = os.path.join(test_data_dir, "test_3.h5")
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

    @pytest.mark.skipif(
        "Windows" in os.environ.get("OS", ""), reason="skipped on windows machine"
    )
    def test_use_Output(self, test_data_dir):
        fname = os.path.join(test_data_dir, "test_4.h5")
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

        output = SetOutput(fname)
        with output.use(append=True, cache=True) as out:
            cachedData = CachedData(10, 1, 10, output=out, dataset_name="test")
        path = Path(cachedData.fname)
        f = h5py.File(fname, "r")
        assert "test" in list(f.keys())
        f.close()
        assert path.is_file() is True

        # TODO deleter remove it when it close the tasks.
        #        del cachedData
        #        assert path.is_file() is False

        try:
            os.remove(fname)
        except:
            pass

        # test not cachable file
        with pytest.raises(IOError):
            output = SetOutput(fname)
            cachedData = CachedData(
                10,
                1,
                10,
                output=output.use(append=True, cache=False),
                dataset_name="test",
            )

        # test wrong class outut file
        with pytest.raises(IOError):
            output = SetOutput(fname)
            cachedData = CachedData(
                10, 1, 10, output=output, dataset_name="test"
            )

        output = SetOutput(fname)
        with output.use(append=True, cache=True) as out:
            cachedData = CachedData(10, 1, 10, output=out, dataset_name="test")
            path = Path(cachedData.fname)
            f = h5py.File(fname, "r")
            assert "test" in list(f.keys())
            f.close()
            assert path.is_file() is True

            # TODO deleter remove it when it close the tasks.
        #            del cachedData
        #            assert path.is_file() is False
        # TODO to solve the file delete issue for windows

        output.delete()

    @pytest.mark.skipif(
        "Windows" in os.environ.get("OS", ""), reason="skipped on windows machine"
    )
    def test_create_data_path(self, test_data_dir):
        fname = os.path.join(test_data_dir, "test_5.h5")
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
            assert check
            f.close()
            assert path.is_file() is True

        try:
            os.remove(fname)
        except:
            pass
