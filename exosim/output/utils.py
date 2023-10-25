import astropy.units as u
import h5py
import numpy as np
from astropy.table import QTable
from astropy.table import Table

import exosim.models.signal as signal


def recursively_save_dict_contents_to_output(output, dic):
    """
    Will recursive write a dictionary into output.

    Parameters
    ----------
    :class:`~exosim.output.output.Output` :
        Group (or root) in output file to write to

    dic : :obj:`dict`
        Dictionary we want to write

    """

    for key, item in dic.items():
        try:
            store_thing(output, key, item)
        except TypeError:
            raise ValueError("Cannot write %s type" % type(item))
    return


def store_thing(output, key, item):
    """
    It stores one thing into the :class:`~exosim.output.output.Output`

    Parameters
    ----------
    :class:`~exosim.output.output.Output`:
        Group (or root) in output file to write to
    key: str
        name for the stored item
    item: obj
     item to store
    """
    if isinstance(item, u.Quantity):
        output.write_quantity(key, item)
    elif isinstance(
        item,
        (
            float,
            int,
            np.int64,
            np.float64,
        ),
    ):
        output.write_scalar(key, item)
    elif isinstance(item, np.ndarray):
        if True in [isinstance(x, str) for x in item]:
            output.write_string_array(key, item)
        else:
            output.write_array(key, item)
    elif isinstance(item, (str,)):
        output.write_string(key, item)
    elif isinstance(item, (Table, QTable)):
        output.write_table(key, item)
    elif isinstance(item, signal.Signal):
        item.write(output, key)
    #        group = output.create_group(key)
    #        recursively_save_dict_contents_to_output(group, item.to_dict())
    elif isinstance(
        item,
        (
            list,
            tuple,
        ),
    ):
        if isinstance(item, tuple):
            item = list(item)
        if True in [isinstance(x, str) for x in item]:
            output.write_string_array(key, np.array(item))
        else:
            try:
                output.write_array(key, np.array(item))

            except (TypeError, ValueError):
                for idx, val in enumerate(item):
                    new_key = "{}{}".format(key, idx)
                    store_thing(output, new_key, val)

    elif isinstance(item, dict):
        group = output.create_group(key)
        recursively_save_dict_contents_to_output(group, item)
    elif item is None:
        pass
    else:
        raise TypeError
    return
