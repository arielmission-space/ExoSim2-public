"""
Unit tests for output cleaning utilities.

Tests the prune_output function which recursively searches for 'channels'
groups in HDF5 files and removes unnecessary data to reduce file size.
"""

import os

import h5py
import numpy as np
import pytest

from exosim.output.hdf5.hdf5 import HDF5Output
from exosim.utils.output_cleaners import prune_output


@pytest.fixture
def temp_hdf5_file(test_data_dir):
    """Fixture providing temporary HDF5 file with cleanup."""
    os.makedirs(test_data_dir, exist_ok=True)
    fname = os.path.join(test_data_dir, "test_output_cleaners.h5")
    yield fname
    if os.path.exists(fname):
        os.remove(fname)


@pytest.fixture
def simple_output_structure(temp_hdf5_file):
    """
    Create a simple HDF5 file with channels at root level.

    Structure:
    /
    ├── info/
    ├── configuration/
    ├── sky/
    ├── telescope/
    └── channels/
        ├── channel1/
        │   ├── focal_plane/
        │   ├── efficiency/
        │   ├── frg_focal_plane/
        │   ├── bkg_focal_plane/
        │   ├── responsivity/
        │   ├── extra_data_1/  (to be deleted)
        │   └── extra_data_2/  (to be deleted)
        └── channel2/
            ├── focal_plane/
            ├── efficiency/
            └── extra_data_3/  (to be deleted)
    """
    with h5py.File(temp_hdf5_file, "w") as f:
        # Create root-level groups
        f.create_group("info")
        f.create_group("configuration")
        f.create_group("sky")
        f.create_group("telescope")

        # Create channels with essential and extra data
        channels = f.create_group("channels")

        # Channel 1
        ch1 = channels.create_group("channel1")
        ch1.create_dataset("focal_plane", data=np.ones((10, 10)))
        ch1.create_dataset("efficiency", data=np.ones(10))
        ch1.create_dataset("frg_focal_plane", data=np.ones((10, 10)))
        ch1.create_dataset("bkg_focal_plane", data=np.ones((10, 10)))
        ch1.create_dataset("responsivity", data=np.ones(10))
        ch1.create_dataset("extra_data_1", data=np.ones(10))
        ch1.create_dataset("extra_data_2", data=np.ones(10))

        # Channel 2
        ch2 = channels.create_group("channel2")
        ch2.create_dataset("focal_plane", data=np.ones((10, 10)))
        ch2.create_dataset("efficiency", data=np.ones(10))
        ch2.create_dataset("extra_data_3", data=np.ones(10))

    return temp_hdf5_file


@pytest.fixture
def nested_output_structure(temp_hdf5_file):
    """
    Create an HDF5 file with channels nested in subdirectories.

    Structure:
    /
    ├── info/
    ├── configuration/
    ├── target_list/
    │   ├── target1/
    │   │   ├── some_data/
    │   │   └── channels/
    │   │       └── channel1/
    │   │           ├── focal_plane/
    │   │           ├── efficiency/
    │   │           └── extra_nested_data/  (to be deleted)
    │   └── target2/
    │       └── channels/
    │           └── channel1/
    │               ├── focal_plane/
    │               └── extra_nested_data_2/  (to be deleted)
    └── extra_root_group/  (to be deleted, but contains channels)
    """
    with h5py.File(temp_hdf5_file, "w") as f:
        # Root level groups
        f.create_group("info")
        f.create_group("configuration")

        # Nested structure with channels
        target_list = f.create_group("target_list")

        # Target 1
        target1 = target_list.create_group("target1")
        target1.create_group("some_data")
        channels1 = target1.create_group("channels")
        ch1 = channels1.create_group("channel1")
        ch1.create_dataset("focal_plane", data=np.ones((10, 10)))
        ch1.create_dataset("efficiency", data=np.ones(10))
        ch1.create_dataset("extra_nested_data", data=np.ones(10))

        # Target 2
        target2 = target_list.create_group("target2")
        channels2 = target2.create_group("channels")
        ch2 = channels2.create_group("channel1")
        ch2.create_dataset("focal_plane", data=np.ones((10, 10)))
        ch2.create_dataset("extra_nested_data_2", data=np.ones(10))

        # Extra root group that should be kept because it contains channels
        f.create_group("extra_root_group")

    return temp_hdf5_file


@pytest.fixture
def no_channels_structure(temp_hdf5_file):
    """
    Create an HDF5 file without any channels group.

    Structure:
    /
    ├── info/
    ├── configuration/
    ├── data1/
    └── data2/
    """
    with h5py.File(temp_hdf5_file, "w") as f:
        f.create_group("info")
        f.create_group("configuration")
        f.create_group("data1")
        f.create_group("data2")

    return temp_hdf5_file


class TestPruneOutputSimpleStructure:
    """Test pruning with channels at root level."""

    def test_keeps_essential_channel_data(self, simple_output_structure):
        """Test that essential channel data is preserved."""
        # Open and prune
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        # Verify essential data is kept
        with h5py.File(simple_output_structure, "r") as f:
            assert "channels" in f
            assert "channel1" in f["channels"]
            assert "channel2" in f["channels"]

            # Check channel1 has only essential data
            ch1_keys = set(f["channels"]["channel1"].keys())
            expected_keys = {
                "focal_plane",
                "efficiency",
                "frg_focal_plane",
                "bkg_focal_plane",
                "responsivity",
            }
            assert ch1_keys == expected_keys

            # Check channel2 has only essential data
            ch2_keys = set(f["channels"]["channel2"].keys())
            expected_keys_ch2 = {"focal_plane", "efficiency"}
            assert ch2_keys == expected_keys_ch2

    def test_deletes_extra_channel_data(self, simple_output_structure):
        """Test that extra channel data is deleted."""
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        with h5py.File(simple_output_structure, "r") as f:
            # Verify extra data is removed
            assert "extra_data_1" not in f["channels"]["channel1"]
            assert "extra_data_2" not in f["channels"]["channel1"]
            assert "extra_data_3" not in f["channels"]["channel2"]

    def test_deletes_non_essential_root_groups(self, simple_output_structure):
        """Test that non-essential root groups are deleted."""
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        with h5py.File(simple_output_structure, "r") as f:
            # Check that non-essential root groups are deleted
            assert "sky" not in f
            assert "telescope" not in f

            # Check that essential root groups are kept
            assert "info" in f
            assert "configuration" in f
            assert "channels" in f

    def test_preserves_data_values(self, simple_output_structure):
        """Test that data values in kept datasets are unchanged."""
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        with h5py.File(simple_output_structure, "r") as f:
            # Verify data integrity
            focal_plane = f["channels"]["channel1"]["focal_plane"][()]
            assert focal_plane.shape == (10, 10)
            np.testing.assert_array_equal(focal_plane, np.ones((10, 10)))


class TestPruneOutputNestedStructure:
    """Test pruning with nested channels groups."""

    def test_finds_nested_channels(self, nested_output_structure):
        """Test that nested channels groups are found and processed."""
        with HDF5Output(nested_output_structure, append=True) as output:
            prune_output(output)

        with h5py.File(nested_output_structure, "r") as f:
            # Verify nested channels are found and processed
            assert "target_list" in f
            assert "target1" in f["target_list"]
            assert "channels" in f["target_list"]["target1"]
            assert "channel1" in f["target_list"]["target1"]["channels"]

            # Check that extra nested data is removed
            ch1 = f["target_list"]["target1"]["channels"]["channel1"]
            assert "extra_nested_data" not in ch1
            assert "focal_plane" in ch1
            assert "efficiency" in ch1

    def test_processes_multiple_nested_channels(self, nested_output_structure):
        """Test that multiple nested channels groups are all processed."""
        with HDF5Output(nested_output_structure, append=True) as output:
            prune_output(output)

        with h5py.File(nested_output_structure, "r") as f:
            # Check target1 channels
            ch1 = f["target_list"]["target1"]["channels"]["channel1"]
            assert "focal_plane" in ch1
            assert "efficiency" in ch1
            assert "extra_nested_data" not in ch1

            # Check target2 channels
            ch2 = f["target_list"]["target2"]["channels"]["channel1"]
            assert "focal_plane" in ch2
            assert "extra_nested_data_2" not in ch2

    def test_keeps_parent_groups_with_nested_channels(self, nested_output_structure):
        """Test that parent groups containing nested channels are preserved."""
        with HDF5Output(nested_output_structure, append=True) as output:
            prune_output(output)

        with h5py.File(nested_output_structure, "r") as f:
            # target_list should be kept even though it's not in the root keep list
            # because it contains nested channels
            assert "target_list" in f
            assert "target1" in f["target_list"]
            assert "target2" in f["target_list"]


class TestPruneOutputEdgeCases:
    """Test edge cases and error handling."""

    def test_no_channels_group(self, no_channels_structure):
        """Test handling of files without channels groups."""
        with HDF5Output(no_channels_structure, append=True) as output:
            # Should not raise an error
            prune_output(output)

        with h5py.File(no_channels_structure, "r") as f:
            # Essential root groups should still be kept
            assert "info" in f
            assert "configuration" in f
            # Non-essential should be deleted
            assert "data1" not in f
            assert "data2" not in f

    def test_empty_channels_group(self, temp_hdf5_file):
        """Test handling of empty channels group."""
        with h5py.File(temp_hdf5_file, "w") as f:
            f.create_group("info")
            f.create_group("channels")  # Empty channels group

        with HDF5Output(temp_hdf5_file, append=True) as output:
            prune_output(output)

        with h5py.File(temp_hdf5_file, "r") as f:
            assert "channels" in f
            assert len(f["channels"].keys()) == 0

    def test_with_logger(self, simple_output_structure):
        """Test that function works with a logger provided."""
        from exosim.log import Logger

        logger = Logger()
        logger.set_log_name()

        with HDF5Output(simple_output_structure, append=True) as output:
            # Should not raise an error with logger
            prune_output(output, logger=logger)

        with h5py.File(simple_output_structure, "r") as f:
            assert "channels" in f
            ch1_keys = set(f["channels"]["channel1"].keys())
            assert "extra_data_1" not in ch1_keys

    def test_without_logger(self, simple_output_structure):
        """Test that function works without a logger (uses logging module)."""
        with HDF5Output(simple_output_structure, append=True) as output:
            # Should not raise an error without logger
            prune_output(output, logger=None)

        with h5py.File(simple_output_structure, "r") as f:
            assert "channels" in f


class TestPruneOutputIntegration:
    """Integration tests for complete pruning scenarios."""

    def test_file_size_reduction(self, simple_output_structure):
        """Test that pruning reduces file size (or keeps it similar due to HDF5 structure)."""
        # Get original size
        original_size = os.path.getsize(simple_output_structure)

        # Count items before pruning
        with h5py.File(simple_output_structure, "r") as f:
            items_before = len(list(f["channels"]["channel1"].keys()))

        # Prune the output
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        # Count items after pruning
        with h5py.File(simple_output_structure, "r") as f:
            items_after = len(list(f["channels"]["channel1"].keys()))

        # Get new size
        pruned_size = os.path.getsize(simple_output_structure)

        # Should have fewer items after pruning
        assert items_after < items_before

        # File size might not always decrease due to HDF5 internal structure,
        # but it shouldn't increase significantly
        assert pruned_size <= original_size * 1.5  # Allow some overhead

    def test_multiple_prune_calls_idempotent(self, simple_output_structure):
        """Test that calling prune multiple times has the same result."""
        # First prune
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        # Get structure after first prune
        with h5py.File(simple_output_structure, "r") as f:
            keys_after_first = set(f["channels"]["channel1"].keys())

        # Second prune
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output)

        # Get structure after second prune
        with h5py.File(simple_output_structure, "r") as f:
            keys_after_second = set(f["channels"]["channel1"].keys())

        # Should be identical
        assert keys_after_first == keys_after_second

    def test_complex_nested_structure(self, temp_hdf5_file):
        """Test with a complex nested structure mimicking real usage."""
        # Create complex structure
        with h5py.File(temp_hdf5_file, "w") as f:
            f.create_group("info")
            f.create_group("configuration")
            f.create_group("radiometric")

            # Create deeply nested structure
            obs = f.create_group("observations")
            for i in range(3):
                target = obs.create_group(f"target_{i}")
                target.create_group("metadata")
                channels = target.create_group("channels")

                for j in range(2):
                    ch = channels.create_group(f"channel_{j}")
                    ch.create_dataset("focal_plane", data=np.ones((5, 5)))
                    ch.create_dataset("efficiency", data=np.ones(5))
                    ch.create_dataset("frg_focal_plane", data=np.ones((5, 5)))
                    ch.create_dataset("bkg_focal_plane", data=np.ones((5, 5)))
                    ch.create_dataset("responsivity", data=np.ones(5))
                    # Extra data to be deleted
                    ch.create_dataset("intermediate_1", data=np.ones(5))
                    ch.create_dataset("intermediate_2", data=np.ones(5))
                    ch.create_dataset("intermediate_3", data=np.ones(5))

        # Prune
        with HDF5Output(temp_hdf5_file, append=True) as output:
            prune_output(output)

        # Verify
        with h5py.File(temp_hdf5_file, "r") as f:
            # Check all channels are processed
            for i in range(3):
                for j in range(2):
                    ch = f["observations"][f"target_{i}"]["channels"][f"channel_{j}"]
                    ch_keys = set(ch.keys())

                    # Should have only essential data
                    expected = {
                        "focal_plane",
                        "efficiency",
                        "frg_focal_plane",
                        "bkg_focal_plane",
                        "responsivity",
                    }
                    assert ch_keys == expected

                    # Extra data should be removed
                    assert "intermediate_1" not in ch_keys
                    assert "intermediate_2" not in ch_keys
                    assert "intermediate_3" not in ch_keys


class TestPruneOutputParameters:
    """The folders_to_remove / folders_to_keep / compact_file options."""

    def test_folders_to_remove_deletes_matching_groups(self, simple_output_structure):
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output, folders_to_remove=["sky", "telescope"])
        with h5py.File(simple_output_structure, "r") as f:
            assert "sky" not in f
            assert "telescope" not in f
            assert "channels" in f

    def test_compact_file_false_skips_recompaction(self, simple_output_structure):
        with HDF5Output(simple_output_structure, append=True) as output:
            prune_output(output, compact_file=False)
        # the file is still valid and pruned
        with h5py.File(simple_output_structure, "r") as f:
            assert "extra_data_1" not in f["channels"]["channel1"]


class TestCompactHdf5File:
    def test_compacting_removes_deleted_data(self, temp_hdf5_file):
        from exosim.utils.output_cleaners import _compact_hdf5_file

        with h5py.File(temp_hdf5_file, "w") as f:
            f.create_dataset("keep", data=np.ones((50, 50)))
            f.create_dataset("drop", data=np.ones((500, 500)))
        with h5py.File(temp_hdf5_file, "a") as f:
            del f["drop"]

        _compact_hdf5_file(temp_hdf5_file)

        with h5py.File(temp_hdf5_file, "r") as f:
            assert "keep" in f
            assert "drop" not in f
            np.testing.assert_array_equal(f["keep"][()], np.ones((50, 50)))

    def test_missing_file_is_handled(self, tmp_path):
        from exosim.utils.output_cleaners import _compact_hdf5_file

        # should not raise, just return
        _compact_hdf5_file(str(tmp_path / "does_not_exist.h5"))

    def test_missing_file_logs_when_a_logger_is_given(self, tmp_path):
        from unittest.mock import MagicMock

        from exosim.utils.output_cleaners import _compact_hdf5_file

        logger = MagicMock()
        _compact_hdf5_file(str(tmp_path / "nope.h5"), logger=logger)
        logger.error.assert_called_once()

    def test_attributes_are_carried_over(self, tmp_path):
        from unittest.mock import MagicMock

        from exosim.utils.output_cleaners import _compact_hdf5_file

        fname = str(tmp_path / "attrs.h5")
        with h5py.File(fname, "w") as f:
            f.attrs["root_attr"] = "r"
            g = f.create_group("grp")
            g.attrs["group_attr"] = 7
            d = g.create_dataset("data", data=np.arange(10))
            d.attrs["ds_attr"] = 1.5
            f.create_dataset("drop", data=np.ones((200, 200)))
        with h5py.File(fname, "a") as f:
            del f["drop"]

        logger = MagicMock()
        _compact_hdf5_file(fname, logger=logger)

        with h5py.File(fname, "r") as f:
            assert f.attrs["root_attr"] == "r"
            assert f["grp"].attrs["group_attr"] == 7
            assert f["grp"]["data"].attrs["ds_attr"] == 1.5
            assert "drop" not in f
        logger.info.assert_called()

    def test_unreadable_file_raises_and_cleans_the_temp(self, tmp_path):
        from exosim.utils.output_cleaners import _compact_hdf5_file

        bad = tmp_path / "bad.h5"
        bad.write_bytes(b"not an hdf5 file at all")
        with pytest.raises(OSError, match=r"[Uu]nable to (synchronously )?open"):
            _compact_hdf5_file(str(bad))
        assert not (tmp_path / "bad.h5.tmp").exists()


class TestPruneOutputResolution:
    def test_no_valid_group_returns_quietly(self):
        from unittest.mock import MagicMock

        from exosim.utils.output_cleaners import prune_output

        out = MagicMock()
        out._group = None
        out.fd = None
        logger = MagicMock()
        prune_output(out, logger=logger)
        logger.error.assert_called_with("No valid HDF5 group or file found")

    def test_compact_warns_when_no_filename_can_be_found(self, tmp_path):
        from unittest.mock import MagicMock

        from exosim.utils.output_cleaners import prune_output

        fname = str(tmp_path / "anon.h5")
        with h5py.File(fname, "w") as f:
            f.create_group("info")
            ch = f.create_group("channels").create_group("c")
            ch.create_dataset("focal_plane", data=np.ones((4, 4)))
            ch.create_dataset("junk", data=np.ones(4))

        # an output-like object that carries the open file handle but no
        # discoverable filename -> the compaction step must bail out with a warning
        class _Anon:
            pass

        out = _Anon()
        out.fd = h5py.File(fname, "a")
        logger = MagicMock()
        try:
            prune_output(out, logger=logger)
        finally:
            out.fd.close()
        assert any(
            "could not determine filename" in str(c)
            for c in logger.warning.call_args_list
        )
