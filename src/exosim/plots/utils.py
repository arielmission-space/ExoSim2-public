from copy import deepcopy

import h5py
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from astropy.table import Table


def _create_ordered_cmap(
    map_name: str,
    roll: int | None = None,
    delete: int | None = None,
    change: list[tuple[int, int]] | None = None,
) -> mcolors.ListedColormap:
    """
    Create an ordered colormap based on a given colormap name.

    Parameters:
    - map_name (str): The name of the colormap to use.
    - roll (int, optional): The number of positions to roll the colormap colors. Defaults to None.
    - delete (int, optional): The index of the color to delete from the colormap. Defaults to None.
    - change (list[tuple[int, int]], optional): A list of pairs of indices to swap colors in the colormap. Defaults to None.

    Returns:
    - cmap (matplotlib.colors.ListedColormap): The created ordered colormap.
    """
    pastel1_cmap = mpl.colormaps.get_cmap(map_name)
    pastel1_colors = pastel1_cmap(np.linspace(0, 1, pastel1_cmap.N))

    pastel1_colors_hsv = np.array(
        [mcolors.rgb_to_hsv(color[:3]) for color in pastel1_colors]
    )
    sorted_indices = np.argsort(pastel1_colors_hsv[:, 0])
    sorted_colors = pastel1_colors[sorted_indices]

    reversed_colors = sorted_colors[::-1]
    if roll is not None:
        reversed_colors = np.roll(reversed_colors, roll, axis=0)
    if delete is not None:
        if isinstance(delete, list):
            for del_ in delete:
                reversed_colors = np.delete(reversed_colors, del_, axis=0)
        else:
            reversed_colors = np.delete(reversed_colors, delete, axis=0)

    if change is not None:
        for couple in change:
            val1 = deepcopy(reversed_colors[couple[0]])
            reversed_colors[couple[0]] = deepcopy(reversed_colors[couple[1]])
            reversed_colors[couple[1]] = val1

    return mcolors.ListedColormap(reversed_colors)


def prepare_channels_list(
    input_table,
) -> tuple[np.ndarray, mpl.colors.Normalize]:
    """
    Prepare the list of channels and the normalization object for plotting,
    using a table with 'ch_name' and 'wavelength' columns.

    Parameters
    ----------
    input_table : astropy.table.Table or QTable
        Table containing at least 'ch_name' and 'wavelength' columns.

    Returns
    -------
    Tuple[np.ndarray, matplotlib.colors.Normalize]
        The sorted array of channel names and the normalization object.
    """
    # Check if input_table is actually a table
    if isinstance(input_table, str):
        with h5py.File(input_table, "r") as f:
            if "channels" in f:
                channels_path = f["channels"]
            elif "targets" in f:
                targets = list(f["targets"].keys())
                channels_path = f["targets"][targets[0]]["channels"]
            else:
                raise ValueError("No channels or targets found in the HDF5 file.")

            channels_wl = []
            channels = np.array(list(channels_path.keys()))
            for channel_name in channels:
                metadata = channels_path[channel_name]["focal_plane/metadata"]
                wl_min = metadata["wl_min"]["value"][()]
                channels_wl.append(wl_min)
            # Sort channels by their minimum wavelength
            id_ = np.argsort(channels_wl)
            channels_sorted = channels[id_]
            norm = mpl.colors.Normalize(vmin=0.0, vmax=len(channels_sorted))

    if isinstance(input_table, Table):
        # Get unique channel names
        channels = np.unique(input_table["ch_name"])
        # For each channel, get the minimum wavelength (or another representative value)
        channels_wl = []
        # Try both 'wavelength' and 'Wavelength' column names
        wl_col = "wavelength" if "wavelength" in input_table.colnames else "Wavelength"
        for channel_name in channels:
            mask = input_table["ch_name"] == channel_name
            wl_min = np.min(input_table[wl_col][mask])
            channels_wl.append(wl_min.value)
        # Sort channels by their minimum wavelength
        id_ = np.argsort(channels_wl)
        channels_sorted = channels[id_]
        norm = mpl.colors.Normalize(vmin=0.0, vmax=len(channels_sorted))

    return channels_sorted, norm


def find_channels_position(input_file: str) -> str:
    """
    Find the positions of the specified channels in the HDF5 file.

    Parameters
    ----------
    input_file : str
        Path to the HDF5 file.

    Returns
    -------
    str
        The path to the channels in the HDF5 file.
    """
    with h5py.File(input_file, "r") as f:
        if "channels" in f:
            channels_path = "channels"
        elif "targets" in f:
            targets = list(f["targets"].keys())
            channels_path = f"targets/{targets[0]}/channels"

    return channels_path
