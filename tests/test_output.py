import os

import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import Table

from exosim.models.signal import (
    Adu,
    Counts,
    CountsPerSecond,
    Dimensionless,
    Radiance,
    Sed,
    Signal,
)
from exosim.output.hdf5.hdf5 import HDF5Output
from exosim.output.hdf5.utils import load_signal, recursively_read_dict_contents


@pytest.fixture
def temp_hdf5_file(test_data_dir):
    """Temporary HDF5 file fixture."""
    fname = os.path.join(test_data_dir, "test_output.h5")
    yield fname
    if os.path.exists(fname):
        os.remove(fname)


def test_attributes(temp_hdf5_file):
    with HDF5Output(temp_hdf5_file) as o:
        pass

    with h5py.File(temp_hdf5_file, "r") as f:
        assert f.attrs["program_name"] == "ExoSim2"
        assert f.attrs["creator"] == "HDF5Output"
        assert f.attrs["file_name"] == temp_hdf5_file


def test_group_creation(temp_hdf5_file):
    with HDF5Output(temp_hdf5_file) as o:
        o.create_group("test_group")

    with h5py.File(temp_hdf5_file, "r") as f:
        assert "test_group" in f.keys()


def test_array_write_read(temp_hdf5_file):
    arr = np.ones((10, 1, 10))
    with HDF5Output(temp_hdf5_file) as o:
        g = o.create_group("test_group")
        g.write_array("test_array", arr)

    with h5py.File(temp_hdf5_file, "r") as f:
        np.testing.assert_array_equal(f["test_group"]["test_array"][()], arr)


def test_quantity_write_read(temp_hdf5_file):
    q = 1 * u.m
    with HDF5Output(temp_hdf5_file) as o:
        g = o.create_group("test_group")
        g.write_quantity("test_quantity", q)

    with h5py.File(temp_hdf5_file, "r") as f:
        retrieved = f["test_group"]["test_quantity"]["value"][()] * u.Unit(
            f["test_group"]["test_quantity"]["unit"][()]
        )
        np.testing.assert_array_equal(retrieved, q)


def test_table_write_read(temp_hdf5_file):
    table = Table()
    table["strings"] = ["test1", "test2"]
    table["floats"] = [1.0, 2.0]
    table["quantities"] = [3.0 * u.m, 4.0 * u.m]

    with HDF5Output(temp_hdf5_file) as o:
        g = o.create_group("test_group")
        g.write_table("test_table", table)

    with h5py.File(temp_hdf5_file, "r") as f:
        retrieved = read_table_hdf5(f["test_group"], "test_table")
        np.testing.assert_array_equal(retrieved["floats"], table["floats"])
        np.testing.assert_array_equal(retrieved["quantities"], table["quantities"])


@pytest.mark.parametrize(
    "signal_class, expected_unit",
    [
        (Adu, u.adu),
        (Counts, u.count),
        (CountsPerSecond, u.count / u.s),
        (Radiance, u.W / (u.m**2 * u.um * u.sr)),
        (Sed, u.W / (u.m**2 * u.um)),
        (Dimensionless, None),
    ],
)
def test_signal_write_read(temp_hdf5_file, signal_class, expected_unit):
    wl = np.linspace(0.1, 1, 10) * u.um
    data = np.random.random_sample((10, 1, 10))
    time_grid = np.linspace(1, 5, 10) * u.hr

    signal = signal_class(spectral=wl, data=data, time=time_grid, cached=False)

    with HDF5Output(temp_hdf5_file) as o:
        g = o.create_group("test_group")
        signal.write(g, f"test_{signal_class.__name__}")

    with h5py.File(temp_hdf5_file, "r") as f:
        loaded_signal = load_signal(f["test_group"][f"test_{signal_class.__name__}"])
        assert isinstance(loaded_signal, signal_class)

        if expected_unit:
            assert loaded_signal.data_units == expected_unit

        np.testing.assert_array_equal(loaded_signal.data, data)
        np.testing.assert_array_equal(
            loaded_signal.spectral * loaded_signal.spectral_units, wl
        )
        np.testing.assert_array_equal(
            loaded_signal.time * loaded_signal.time_units, time_grid
        )


def test_cached_signal_error(temp_hdf5_file):
    wl = np.linspace(0.1, 1, 10) * u.um
    data = np.random.random_sample((10, 1, 10))
    time_grid = np.linspace(1, 5, 10) * u.hr

    signal = Signal(spectral=wl, data=data, time=time_grid, cached=True)

    with HDF5Output(temp_hdf5_file) as o:
        g = o.create_group("test_group")
        signal.write(g, "test_cached_signal")

    with h5py.File(temp_hdf5_file, "r") as f:
        with pytest.raises(OSError):
            load_signal(f["test_group"]["test_cached_signal"])


def test_dictionary_write_read(temp_hdf5_file):
    wl = np.linspace(0.1, 1, 10) * u.um
    data = np.random.random_sample((10, 1, 10))
    time_grid = np.linspace(1, 5, 10) * u.hr

    signal = Signal(spectral=wl, data=data, time=time_grid, cached=False)

    dictionary = {
        "value": 42,
        "list": [1, 2, 3],
        "signal": signal,
    }

    with HDF5Output(temp_hdf5_file) as o:
        g = o.create_group("test_group")
        g.store_dictionary(dictionary, group_name="test_dict")

    with h5py.File(temp_hdf5_file, "r") as f:
        input_dict = recursively_read_dict_contents(f)
        assert input_dict["test_group"]["test_dict"]["value"] == 42
        np.testing.assert_array_equal(
            input_dict["test_group"]["test_dict"]["list"], [1, 2, 3]
        )
