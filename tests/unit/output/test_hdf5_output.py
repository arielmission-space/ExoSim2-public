"""
Unit tests for HDF5 output functionality.

Tests the HDF5Output class and related utilities for writing
and reading various data types to/from HDF5 files.
"""

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
from exosim.output.hdf5.utils import (
    load_signal,
    recursively_read_dict_contents,
)


@pytest.fixture
def temp_hdf5_file(test_data_dir):
    """Fixture providing temporary HDF5 file with cleanup."""
    os.makedirs(test_data_dir, exist_ok=True)
    fname = os.path.join(test_data_dir, "test_hdf5_output.h5")
    yield fname
    if os.path.exists(fname):
        os.remove(fname)


class TestHDF5OutputFileOperations:
    """Test HDF5 file creation and metadata."""

    def test_file_attributes_creation(self, temp_hdf5_file):
        """Test that HDF5 file is created with proper attributes."""
        with HDF5Output(temp_hdf5_file):
            pass

        with h5py.File(temp_hdf5_file, "r") as f:
            assert f.attrs["program_name"] == "ExoSim 2"
            assert f.attrs["creator"] == "HDF5Output"
            assert f.attrs["file_name"] == temp_hdf5_file

    def test_group_creation(self, temp_hdf5_file):
        """Test creating groups in HDF5 file."""
        with HDF5Output(temp_hdf5_file) as output:
            output.create_group("test_group")

        with h5py.File(temp_hdf5_file, "r") as f:
            assert "test_group" in f


class TestBasicDataTypes:
    """Test writing and reading basic data types."""

    def test_array_write_and_read(self, temp_hdf5_file):
        """Test writing and reading numpy arrays."""
        test_array = np.ones((10, 1, 10))

        with HDF5Output(temp_hdf5_file) as output:
            group = output.create_group("arrays")
            group.write_array("test_array", test_array)

        with h5py.File(temp_hdf5_file, "r") as f:
            stored_array = f["arrays"]["test_array"][()]
            np.testing.assert_array_equal(stored_array, test_array)

    def test_quantity_write_and_read(self, temp_hdf5_file):
        """Test writing and reading astropy quantities."""
        test_quantity = 42.5 * u.m

        with HDF5Output(temp_hdf5_file) as output:
            group = output.create_group("quantities")
            group.write_quantity("test_quantity", test_quantity)

        with h5py.File(temp_hdf5_file, "r") as f:
            value = f["quantities"]["test_quantity"]["value"][()]
            unit_str = f["quantities"]["test_quantity"]["unit"][()]
            retrieved = value * u.Unit(unit_str)

            np.testing.assert_array_equal(retrieved, test_quantity)

    def test_table_write_and_read(self, temp_hdf5_file):
        """Test writing and reading astropy tables."""
        table = Table()
        table["strings"] = ["test1", "test2", "test3"]
        table["floats"] = [1.0, 2.5, 3.7]
        table["quantities"] = [10.0 * u.m, 20.0 * u.m, 30.0 * u.m]

        with HDF5Output(temp_hdf5_file) as output:
            group = output.create_group("tables")
            group.write_table("test_table", table)

        with h5py.File(temp_hdf5_file, "r") as f:
            retrieved_table = read_table_hdf5(f["tables"], "test_table")

            np.testing.assert_array_equal(retrieved_table["floats"], table["floats"])
            np.testing.assert_array_equal(
                retrieved_table["quantities"], table["quantities"]
            )
            assert list(retrieved_table["strings"]) == list(table["strings"])


class TestSignalPersistence:
    """Test writing and reading Signal objects."""

    @pytest.mark.parametrize(
        ("signal_class", "expected_unit"),
        [
            (Adu, u.adu),
            (Counts, u.count),
            (CountsPerSecond, u.count / u.s),
            (Radiance, u.W / (u.m**2 * u.um * u.sr)),
            (Sed, u.W / (u.m**2 * u.um)),
            (Dimensionless, None),
        ],
    )
    def test_signal_roundtrip_by_type(
        self, temp_hdf5_file, signal_class, expected_unit
    ):
        """Test writing and reading different Signal types."""
        # Create test data
        wavelength = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        # Create signal
        original_signal = signal_class(
            spectral=wavelength, data=data, time=time_grid, cached=False
        )

        # Write to HDF5
        with HDF5Output(temp_hdf5_file) as output:
            group = output.create_group("signals")
            original_signal.write(group, f"test_{signal_class.__name__}")

        # Read back and verify
        with h5py.File(temp_hdf5_file, "r") as f:
            loaded_signal = load_signal(f["signals"][f"test_{signal_class.__name__}"])

            assert isinstance(loaded_signal, signal_class)

            if expected_unit:
                assert loaded_signal.data_units == expected_unit

            np.testing.assert_array_equal(loaded_signal.data, data)
            np.testing.assert_array_equal(
                loaded_signal.spectral * loaded_signal.spectral_units, wavelength
            )
            np.testing.assert_array_equal(
                loaded_signal.time * loaded_signal.time_units, time_grid
            )

    def test_cached_signal_load_error(self, temp_hdf5_file):
        """Test that loading cached signals raises appropriate error."""
        wavelength = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr

        # Create cached signal
        cached_signal = Signal(
            spectral=wavelength, data=data, time=time_grid, cached=True
        )

        # Write to HDF5
        with HDF5Output(temp_hdf5_file) as output:
            group = output.create_group("cached_signals")
            cached_signal.write(group, "cached_signal")

        # Attempt to read should raise error
        with (
            h5py.File(temp_hdf5_file, "r") as f,
            pytest.raises(OSError, match="Cannot load cached signal"),
        ):
            load_signal(f["cached_signals"]["cached_signal"])


class TestDictionaryPersistence:
    """Test writing and reading dictionary data."""

    def test_dictionary_storage_and_retrieval(self, temp_hdf5_file):
        """Test storing and retrieving complex dictionary structures."""
        # Create test data including a Signal
        wavelength = np.linspace(0.1, 1, 10) * u.um
        data = np.random.random_sample((10, 1, 10))
        time_grid = np.linspace(1, 5, 10) * u.hr
        test_signal = Signal(
            spectral=wavelength, data=data, time=time_grid, cached=False
        )

        # Create complex dictionary
        test_dictionary = {
            "scalar_value": 42,
            "list_values": [1, 2, 3, 4, 5],
            "nested_dict": {"inner_value": 100, "inner_list": ["a", "b", "c"]},
            "signal_object": test_signal,
        }

        # Write dictionary to HDF5
        with HDF5Output(temp_hdf5_file) as output:
            group = output.create_group("dictionaries")
            group.store_dictionary(test_dictionary, group_name="test_dict")

        # Read back and verify
        with h5py.File(temp_hdf5_file, "r") as f:
            retrieved_dict = recursively_read_dict_contents(f)

            assert retrieved_dict["dictionaries"]["test_dict"]["scalar_value"] == 42
            np.testing.assert_array_equal(
                retrieved_dict["dictionaries"]["test_dict"]["list_values"],
                [1, 2, 3, 4, 5],
            )

            # Verify nested dictionary structure was preserved
            nested = retrieved_dict["dictionaries"]["test_dict"]["nested_dict"]
            assert nested["inner_value"] == 100


class TestWriteMethods:
    """The individual write_* helpers of HDF5OutputGroup, with metadata."""

    def test_write_array_list_and_metadata(self, temp_hdf5_file):
        with HDF5Output(temp_hdf5_file) as output:
            g = output.create_group("g")
            g.write_array("stack", [np.ones(3), np.zeros(3)], metadata={"unit": "ct"})
            g.write_scalar("answer", 42, metadata={"note": "meaning"})
            g.write_string("label", "hello", metadata={"lang": "en"})
            g.write_string_array("names", np.array(["a", "bb", "ccc"]))
        with h5py.File(temp_hdf5_file, "r") as f:
            assert "stack0" in f["g"]
            assert "stack1" in f["g"]
            assert f["g"]["stack0"].attrs["unit"] == "ct"
            assert f["g"]["answer"][()] == 42
            assert f["g"]["answer"].attrs["note"] == "meaning"
            assert f["g"]["label"][()].decode() == "hello"
            assert f["g"]["names"].shape == (3,)

    def test_write_table_replace_overwrites_cleanly(self, temp_hdf5_file):
        t1 = Table({"a": [1, 2, 3]})
        t1.meta["origin"] = "first"
        t2 = Table({"a": [9, 9]})
        with HDF5Output(temp_hdf5_file) as output:
            g = output.create_group("g")
            g.write_table("tab", t1)
            g.write_table("tab", t2, metadata={"pass": 2}, replace=True)
        with h5py.File(temp_hdf5_file, "r") as f:
            assert f["g"]["tab_to_group"]["a"]["value"].shape == (2,)
            assert f["g"]["tab_to_group"].attrs["pass"] == 2

    def test_write_quantity_scalar_and_group(self, temp_hdf5_file):
        with HDF5Output(temp_hdf5_file) as output:
            g = output.create_group("g")
            g.write_quantity("temperature", 300 * u.K, metadata={"src": "spec"})
        with h5py.File(temp_hdf5_file, "r") as f:
            assert f["g"]["temperature"]["value"][()] == 300
            assert f["g"]["temperature"]["unit"][()].decode() == "K"

    def test_delete_data_missing_key_raises_keyerror(self, temp_hdf5_file):
        with HDF5Output(temp_hdf5_file) as output:
            g = output.create_group("g")
            g.write_scalar("x", 1)
            with pytest.raises(KeyError):
                g.delete_data("not_here")
            g.delete_data("x")


class TestOutputFileHelpers:
    def test_getsize_returns_a_byte_quantity(self, temp_hdf5_file):
        with HDF5Output(temp_hdf5_file) as output:
            output.create_group("g").write_array("d", np.ones((32, 32)))
        with HDF5Output(temp_hdf5_file, append=True) as output:
            size = output.getsize()
        assert size.unit.physical_type == "data quantity"
        assert size.to_value(u.b) > 0

    def test_delete_data_missing_key_on_output_raises(self, temp_hdf5_file):
        with HDF5Output(temp_hdf5_file) as output:
            output.create_group("keep")
            with pytest.raises(KeyError):
                output.delete_data("ghost")


class TestStoreDictionaryDispatch:
    """``store_dictionary`` routes each value type to the right write_* helper."""

    def test_every_supported_type_round_trips(self, temp_hdf5_file):
        payload = {
            "a_quantity": 3.0 * u.m,
            "a_scalar": 7,
            "a_string": "hi",
            "a_string_array": np.array(["x", "yy"]),
            "a_table": Table({"c": [1, 2, 3]}),
            "a_str_tuple": ("p", "q"),
            "a_nested": {"inner": 1},
            "a_none": None,
        }
        with HDF5Output(temp_hdf5_file) as output:
            output.create_group("bag").store_dictionary(payload, group_name="d")
        with h5py.File(temp_hdf5_file, "r") as f:
            g = f["bag"]["d"]
            assert g["a_scalar"][()] == 7
            assert g["a_string_array"].shape == (2,)
            assert "a_table_to_group" in g
            assert g["a_nested"]["inner"][()] == 1

    def test_unsupported_type_raises_value_error(self, temp_hdf5_file):
        with HDF5Output(temp_hdf5_file) as output:
            g = output.create_group("bag")
            with pytest.raises(ValueError, match="Cannot write"):
                g.store_dictionary({"bad": object()}, group_name="d")
