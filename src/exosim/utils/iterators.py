from collections import OrderedDict

import numpy as np
from tqdm.auto import tqdm


def iterate_over_opticalElements(input, key, last_key, val):
    """
    It iterates over the optical element of a given class and returns the edited dictionary

    Parameters
    ----------
    input: dict
        input dictionaru
    key:
        optical element class
    last_key:
        optical element key to edit
    val:
        value to edit

    Returns
    -------
    dict
    """

    if key in input.keys():
        if "optical_path" in input[key].keys():
            _nested(input[key]["optical_path"], last_key, val)
        elif isinstance(input[key], OrderedDict):
            for ch in input[key].keys():
                if "optical_path" in input[key][ch].keys():
                    _nested(input[key][ch]["optical_path"], last_key, val)
                else:
                    _nested(input[key][ch], last_key, val)
        else:
            _nested(input[key], last_key, val)
    return input


def _nested(input, key, val):
    if isinstance(input["opticalElement"], OrderedDict):
        for opt in input["opticalElement"].keys():
            input["opticalElement"][opt][key] = val
    else:
        input["opticalElement"][key] = val


def iterate_over_chunks(dataset, **kwargs):
    """
    Iterates over the dataset chunks using tqdm to produce an adaptive progress bass
    Parameters
    ----------
    dataset: :class:`h5py.Dataset`
        h5py chunked dataset used to store the data

    Returns
    -------
        :class:`tqdm.tqdm`

    """
    total = int(np.ceil(dataset.shape[0] / dataset.chunks[0]))
    return tqdm(dataset.iter_chunks(), total=total, unit="chunk", **kwargs)


# from https://stackoverflow.com/a/20780569 on StackOverflow
# by cyborg (https://stackoverflow.com/users/907578/cyborg)
# this is compliant to StackOverflow's CC BY-SA 3.0
# and its attribution requirement
# (https://stackoverflow.blog/2009/06/25/attribution-required/)
def searchsorted(known_array, test_array):
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
    known_array_middles = (
        known_array_sorted[1:] - np.diff(known_array_sorted.astype("f")) / 2
    )
    idx1 = np.searchsorted(known_array_middles, test_array)
    indices = index_sorted[idx1]
    return indices
