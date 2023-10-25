import logging
import os
import unittest

import astropy.units as u
import h5py
import numpy as np
from astropy.io.misc.hdf5 import read_table_hdf5
from inputs import test_dir

from exosim.log import setLogLevel
from exosim.models.signal import Adu
from exosim.models.signal import Counts
from exosim.models.signal import CountsPerSecond
from exosim.models.signal import Dimensionless
from exosim.models.signal import Radiance
from exosim.models.signal import Sed
from exosim.models.signal import Signal
from exosim.output.hdf5.hdf5 import HDF5Output
from exosim.output.hdf5.utils import load_signal
from exosim.output.hdf5.utils import recursively_read_dict_contents

setLogLevel(logging.DEBUG)


class HDF5Test(unittest.TestCase):
    fname = os.path.join(test_dir, "output_test.h5")

    def test_attributes(self):
        with HDF5Output(self.fname) as o:
            o.close()

        f = h5py.File(self.fname, "r")
        print(dict(f.attrs))
        self.assertEqual(f.attrs["program_name"], "ExoSim2")
        self.assertEqual(f.attrs["creator"], "HDF5Output")
        self.assertEqual(f.attrs["file_name"], self.fname)
        f.close()
        os.remove(self.fname)

    def test_group(self):
        with HDF5Output(self.fname) as o:
            o.create_group("test_group")

        f = h5py.File(self.fname, "r")
        # info is always there because is produced during the first opening
        self.assertListEqual(list(f.keys()), ["info", "test_group"])
        f.close()
        os.remove(self.fname)

    def test_array(self):
        arr = np.ones((10, 1, 10))
        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_array("test_array", arr)

        f = h5py.File(self.fname, "r")
        np.testing.assert_array_equal(f["test_group"]["test_array"][()], arr)
        f.close()
        os.remove(self.fname)

    def test_string_array(self):
        list = ["test"] * 4
        arr = np.array(list)
        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_string_array("test_array", arr)

        f = h5py.File(self.fname, "r")
        np.testing.assert_array_equal(
            f["test_group"]["test_array"][()].astype("<U4"), arr
        )
        f.close()
        os.remove(self.fname)

    def test_scalar(self):
        sc = 1
        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_scalar("test_scalar", sc)

        f = h5py.File(self.fname, "r")
        np.testing.assert_array_equal(f["test_group"]["test_scalar"][()], sc)
        f.close()
        os.remove(self.fname)

    def test_string(self):
        st = "test"
        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_scalar("test_string", st)

        f = h5py.File(self.fname, "r")
        np.testing.assert_array_equal(
            f["test_group"]["test_string"][()], st.encode("utf-8")
        )
        f.close()
        os.remove(self.fname)

    def test_quantity(self):
        q = 1 * u.m
        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_quantity("test_quantity", q)

        f = h5py.File(self.fname, "r")
        retrieved = f["test_group"]["test_quantity"]["value"][()] * u.Unit(
            f["test_group"]["test_quantity"]["unit"][()]
        )
        np.testing.assert_array_equal(retrieved, q)
        f.close()
        os.remove(self.fname)

    def test_table(self):
        from astropy.table import Table

        table = Table()
        table["strings"] = ["test1", "test2"]
        table["floats"] = [1.0, 2.0]
        table["quantities"] = [3.0 * u.m, 4.0 * u.m]

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_table("test_table", table)

        f = h5py.File(self.fname, "r")

        retrieved = read_table_hdf5(f["test_group"], "test_table")

        np.testing.assert_array_equal(retrieved["floats"], table["floats"])
        np.testing.assert_array_equal(
            retrieved["quantities"], table["quantities"]
        )
        f.close()
        os.remove(self.fname)

    def test_dictionary(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(spectral=wl, data=data, time=time_grid, cached=False)

        from astropy.table import Table

        table = Table()
        table["strings"] = ["test1", "test2"]
        table["floats"] = [1.0, 2.0]
        table["quantities"] = [3.0 * u.m, 4.0 * u.m]

        dictionary = {
            "test1": 1,
            "test2": [2, 3],
            "signal": signal,
            "table": table,
        }

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.store_dictionary(dictionary, group_name="test_dict")

        f = h5py.File(self.fname, "r")
        self.assertEqual(f["test_group"]["test_dict"]["test1"][()], 1)
        np.testing.assert_array_equal(
            f["test_group"]["test_dict"]["test2"][()], [2, 3]
        )

        f.close()
        os.remove(self.fname)

    def test_wrong_object(self):
        foo = Foo()

        dictionary = {"test1": foo}
        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            with self.assertRaises(ValueError):
                g.store_dictionary(dictionary, group_name="test_dict")

        os.remove(self.fname)


class Foo:
    def __init__(self):
        self.a = 1
        self.b = 2


class LoadHDF5Test(unittest.TestCase):
    fname = os.path.join(test_dir, "input_test.h5")

    def test_dictionary(self):
        from astropy.table import Table

        dictionary = {"test1": 1, "test2": 2}

        table = Table()
        table["strings"] = ["test1", "test2"]
        table["floats"] = [1.0, 2.0]
        table["quantities"] = [3.0 * u.m, 4.0 * u.m]

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            g.write_table("test_table", table)
            g.store_dictionary(dictionary, group_name="test_dict")

        f = h5py.File(self.fname, "r")
        input = recursively_read_dict_contents(f)
        self.assertEqual(input["test_group"]["test_dict"]["test1"][()], 1)
        self.assertEqual(input["test_group"]["test_dict"]["test2"][()], 2)

        # self.assertIsInstance(input['test_group']['test_table'], QTable)
        np.testing.assert_array_equal(
            input["test_group"]["test_table"]["floats"], table["floats"]
        )
        np.testing.assert_array_equal(
            input["test_group"]["test_table"]["quantities"],
            table["quantities"],
        )
        f.close()
        os.remove(self.fname)

    def test_load_signal(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(spectral=wl, data=data, time=time_grid, cached=False)
        signal.metadata = {"test": 1}

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_signal")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_signal"])

        self.assertIsInstance(loaded_signal, Signal)
        np.testing.assert_array_equal(
            loaded_signal.spectral * loaded_signal.spectral_units, wl
        )
        np.testing.assert_array_equal(loaded_signal.data, data)
        np.testing.assert_array_equal(
            loaded_signal.time * loaded_signal.time_units, time_grid
        )
        self.assertEqual(loaded_signal.metadata["test"], 1)

        f.close()
        os.remove(self.fname)

    def test_load_Sed(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Sed(spectral=wl, data=data, time=time_grid, cached=False)

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_sed")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_sed"])

        self.assertIsInstance(loaded_signal, Sed)
        self.assertEqual(loaded_signal.data_units, u.W / (u.m**2 * u.um))

        f.close()
        os.remove(self.fname)

    def test_load_Radiance(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Radiance(spectral=wl, data=data, time=time_grid, cached=False)

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_radiance")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_radiance"])

        self.assertIsInstance(loaded_signal, Radiance)
        self.assertEqual(
            loaded_signal.data_units, u.W / (u.m**2 * u.um * u.sr)
        )

        f.close()
        os.remove(self.fname)

    def test_load_CountsPerSecond(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = CountsPerSecond(
            spectral=wl, data=data, time=time_grid, cached=False
        )

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_countsPerSecond")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_countsPerSecond"])

        self.assertIsInstance(loaded_signal, CountsPerSecond)
        self.assertEqual(loaded_signal.data_units, u.count / u.s)

        f.close()
        os.remove(self.fname)

    def test_load_Counts(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Counts(spectral=wl, data=data, time=time_grid, cached=False)

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_counts")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_counts"])

        self.assertIsInstance(loaded_signal, Counts)
        self.assertEqual(loaded_signal.data_units, u.count)

        f.close()
        os.remove(self.fname)

    def test_load_Adu(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Adu(spectral=wl, data=data, time=time_grid, cached=False)

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_adu")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_adu"])

        self.assertIsInstance(loaded_signal, Adu)
        self.assertEqual(loaded_signal.data_units, u.adu)

        f.close()
        os.remove(self.fname)

    def test_load_Dimensionless(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Dimensionless(
            spectral=wl, data=data, time=time_grid, cached=False
        )

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_diensionless")

        f = h5py.File(self.fname, "r")
        loaded_signal = load_signal(f["test_group"]["test_diensionless"])

        self.assertIsInstance(loaded_signal, Dimensionless)

        f.close()
        os.remove(self.fname)

    def test_load_cached(self):
        wl = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        signal = Signal(spectral=wl, data=data, time=time_grid, cached=True)

        with HDF5Output(self.fname) as o:
            g = o.create_group("test_group")
            signal.write(g, "test_cached")

        f = h5py.File(self.fname, "r")
        with self.assertRaises(OSError):
            load_signal(f["test_group"]["test_cached"])

        f.close()
        os.remove(self.fname)
