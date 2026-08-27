import os
import shutil

import h5py

from exosim.log import with_logger


def _compact_hdf5_file(filename, logger=None):
    """
    Compact an HDF5 file by creating a new file with only the existing data.
    This physically removes deleted data and reduces file size.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to compact
    logger : logger object, optional
        Logger instance for logging messages
    """
    if not os.path.exists(filename):
        if logger:
            logger.error(f"File {filename} does not exist")
        return

    original_size = os.path.getsize(filename) / (1024**2)
    if logger:
        logger.info(
            f"Compacting HDF5 file: {filename} (original size: {original_size:.1f} MB)"
        )

    # Create temporary file for compacted version
    temp_file = filename + ".tmp"

    try:
        with h5py.File(filename, "r") as source, h5py.File(temp_file, "w") as dest:

            def copy_item(name, obj):
                if isinstance(obj, h5py.Group):
                    dest.create_group(name)
                    # Copy group attributes
                    for attr_name, attr_value in obj.attrs.items():
                        dest[name].attrs[attr_name] = attr_value
                elif isinstance(obj, h5py.Dataset):
                    # Copy dataset with same properties
                    dest.create_dataset_like(name, obj)
                    dest[name][...] = obj[...]
                    # Copy dataset attributes
                    for attr_name, attr_value in obj.attrs.items():
                        dest[name].attrs[attr_name] = attr_value

            # Copy all items
            source.visititems(copy_item)

            # Copy root attributes
            for attr_name, attr_value in source.attrs.items():
                dest.attrs[attr_name] = attr_value

        # Replace original with compacted version
        shutil.move(temp_file, filename)

        compacted_size = os.path.getsize(filename) / (1024**2)
        reduction = original_size - compacted_size
        reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0

        if logger:
            logger.info(
                f"File compacted: {compacted_size:.1f} MB (reduced by {reduction:.1f} MB, {reduction_pct:.1f}%)"
            )

    except Exception as e:
        # Clean up temp file if something went wrong
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if logger:
            logger.error(f"Error compacting file {filename}: {e}")
        raise


@with_logger
def prune_output(
    out,
    logger=None,
    folders_to_remove=None,
    folders_to_keep=None,
    root_groups_to_keep=None,
    compact_file=True,
) -> None:
    """
    It prunes the output file, removing unnecessary data to reduce its size.
    After pruning, the file is automatically compacted to physically remove deleted data.

    Parameters
    ----------
    out: :class:`~exosim.output.output.OutputGroup`
        output group
    logger: logger object (optional)
        logger instance for logging messages
    folders_to_remove: list of str (optional)
        list of folder names to remove entirely from the output (searched recursively)
    folders_to_keep: list of str (optional)
        list of folder names to keep entirely in the output
    root_groups_to_keep: list of str (optional)
        list of root-level groups to keep (defaults to standard groups)
    compact_file: bool (optional)
        if True, recompacts the file after pruning to physically reduce file size (default: True)
    """

    logger.info("pruning output to reduce size")

    # Get the target group to work on
    if hasattr(out, "_group") and out._group is not None:
        target_group = out._group
        group_path = (
            target_group.name if hasattr(target_group, "name") else "target_group"
        )
    else:
        target_group = out.fd if hasattr(out, "fd") and out.fd is not None else None
        group_path = "root"

    if target_group is None or not target_group.id.valid:
        logger.error("No valid HDF5 group or file found")
        return

    logger.debug(f"Working on group: {group_path}")

    # STEP 1: Remove folders_to_remove recursively throughout the entire structure
    def remove_folders_recursively(group, folders_to_delete, path=""):
        """Recursively remove specified folders from entire HDF5 structure"""
        deleted_count = 0  # Initialize the counter for deleted folders

        if not isinstance(group, h5py.Group) or not folders_to_delete:
            return deleted_count

        # Validate group before accessing keys
        if not group.id.valid:
            logger.error(f"Invalid group ID for path: {path}")
            return deleted_count

        keys_to_check = list(group.keys())

        for key in keys_to_check:
            current_path = f"{path}/{key}" if path else key

            # Delete if key matches any folder to remove
            if key in folders_to_delete:
                try:
                    logger.debug(f"Deleting folder: {current_path}")
                    del group[key]
                    deleted_count += 1
                    continue  # Skip recursion since we deleted it
                except KeyError:
                    logger.warning(f"Folder '{current_path}' does not exist. Skipping.")
                    continue
                except Exception as e:
                    logger.error(f"Error deleting folder '{current_path}': {e}")

            # Recurse into subgroups
            try:
                item = group[key]
                if isinstance(item, h5py.Group):
                    deleted_count += remove_folders_recursively(
                        item, folders_to_delete, current_path
                    )
            except KeyError:
                logger.warning(f"Subgroup '{current_path}' does not exist. Skipping.")
            except Exception as e:
                logger.debug(f"Could not process subgroup {current_path}: {e}")

        return deleted_count

    # Remove unwanted folders first
    total_deleted = 0
    if folders_to_remove:
        folders_to_delete = set(folders_to_remove)
        total_deleted = remove_folders_recursively(
            target_group, folders_to_delete, group_path
        )
        if total_deleted > 0:
            logger.info(
                f"Deleted {total_deleted} folders matching: {folders_to_remove}"
            )

    # STEP 2: Apply standard pruning rules (optional - can be simplified or removed)
    # Define what to keep at root level
    if root_groups_to_keep is None:
        root_groups_to_keep = {"info", "configuration", "radiometric", "channels"}
    else:
        root_groups_to_keep = set(root_groups_to_keep)

    if folders_to_keep:
        root_groups_to_keep.update(folders_to_keep)

    # Helper function to check if a group contains channels (directly or nested)
    def contains_channels(group):
        """Recursively check if a group contains a 'channels' subgroup"""
        if not isinstance(group, h5py.Group):
            return False

        # Check direct children
        if "channels" in group:
            return True

        # Check nested groups
        for key in group:
            item = group.get(key)
            if isinstance(item, h5py.Group) and contains_channels(item):
                return True

        return False

    # Remove any remaining unwanted root-level groups
    # But keep groups that contain channels subgroups
    current_keys = list(target_group.keys())
    deleted_groups = []

    for key in current_keys:
        if key not in root_groups_to_keep:
            # Check if this group contains channels before deleting
            item = target_group.get(key)
            if isinstance(item, h5py.Group) and contains_channels(item):
                logger.debug(f"Keeping root group '{key}' because it contains channels")
                continue

            try:
                logger.debug(f"Deleting root group: {key}")
                del target_group[key]
                deleted_groups.append(key)
            except Exception as e:
                logger.error(f"Error deleting root group '{key}': {e}")

    if deleted_groups:
        logger.debug(f"Deleted {len(deleted_groups)} root groups: {deleted_groups}")

    # STEP 3: Clean up channel datasets - keep only essential data
    def clean_channel_datasets(group, path=""):
        """Remove non-essential datasets from channel groups"""
        essential_datasets = {
            "focal_plane",
            "efficiency",
            "frg_focal_plane",
            "bkg_focal_plane",
            "responsivity",
        }

        if not isinstance(group, h5py.Group):
            return 0

        deleted_count = 0

        # Check if this is a channel group (has datasets that match essential ones)
        keys = list(group.keys())
        datasets = [k for k in keys if isinstance(group.get(k), h5py.Dataset)]

        # If we have some essential datasets, assume this is a channel and clean it
        has_essential = any(d in essential_datasets for d in datasets)

        if has_essential and datasets:
            for key in keys:
                item = group.get(key)
                if isinstance(item, h5py.Dataset) and key not in essential_datasets:
                    try:
                        logger.debug(f"Deleting channel dataset: {path}/{key}")
                        del group[key]
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting dataset '{path}/{key}': {e}")

        # Recurse into subgroups
        for key in list(group.keys()):
            item = group.get(key)
            if isinstance(item, h5py.Group):
                current_path = f"{path}/{key}" if path else key
                deleted_count += clean_channel_datasets(item, current_path)

        return deleted_count

    # Clean channel datasets
    deleted_datasets = clean_channel_datasets(target_group, group_path)
    if deleted_datasets > 0:
        logger.debug(f"Deleted {deleted_datasets} non-essential channel datasets")

    # Flush changes
    if hasattr(out, "fd") and out.fd is not None:
        out.fd.flush()

    # STEP 4: Compact the file if requested
    if compact_file:
        filename = None
        if hasattr(out, "fname"):
            filename = out.fname
        elif hasattr(out, "filename"):
            filename = out.filename

        if filename:
            logger.info(f"Starting file compaction for: {filename}")
            # Close the file first if it's open
            if hasattr(out, "fd") and out.fd is not None:
                out.fd.close()
            _compact_hdf5_file(filename, logger)
        else:
            logger.warning("Cannot compact file: could not determine filename")
    else:
        logger.debug("File compaction skipped (compact_file=False)")

    logger.info(f"Output pruning completed for {group_path}")
