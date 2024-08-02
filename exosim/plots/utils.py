import os
from copy import deepcopy
from typing import List
from typing import Tuple

import h5py
import matplotlib
import matplotlib.colors as mcolors
import numpy as np


def _create_ordered_cmap(
    map_name: str,
    roll: int = None,
    delete: int = None,
    change: List[Tuple[int, int]] = None,
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
    pastel1_cmap = matplotlib.cm.get_cmap(map_name)
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
        if isinstance(delete, List):
            for del_ in delete:
                reversed_colors = np.delete(reversed_colors, del_, axis=0)
        else:
            reversed_colors = np.delete(reversed_colors, delete, axis=0)

    if change is not None:
        for couple in change:
            val1 = deepcopy(reversed_colors[couple[0]])
            reversed_colors[couple[0]] = deepcopy(reversed_colors[couple[1]])
            reversed_colors[couple[1]] = val1

    cmap = mcolors.ListedColormap(reversed_colors)
    return cmap


def prepare_channels_list(
    input_file,
) -> Tuple[np.ndarray, matplotlib.colors.Normalize]:
    """
    Prepare the list of channels and the normalization object for plotting.

    Returns
    -------
    Tuple[np.ndarray, matplotlib.colors.Normalize]
        The sorted array of channel names and the normalization object.
    """
    with h5py.File(input_file, "r") as f:
        channels = np.array(list(f["channels"].keys()))
        channels_wl = []
        for channel_name in channels:
            file_path = os.path.join("channels", channel_name)
            file_path = os.path.join(file_path, "focal_plane/metadata")
            channels_wl.append(f[os.path.join(file_path, "wl_min/value")][()])
        id_ = np.argsort(np.array(channels_wl))
        channels = channels[id_]
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=len(channels))
    return channels, norm
