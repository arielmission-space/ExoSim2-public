from typing import Union

import astropy.units as u
import h5py
from astropy.io.misc.hdf5 import read_table_hdf5
from h5py import Dataset
from h5py import Datatype
from h5py import Group

import exosim.models.signal as signal


def load_signal(input: h5py.File) -> signal.Signal:
    """
    It loads the appropriate :class:`~exosim.models.signal.Signal` class from an opened hdf5 file.

    Parameters
    ----------
    input: :class:`h5py.File`
        opened hdf5 file

    Returns
    -------
    :class:`~exosim.models.signal.Signal`

    Raises
    -------
    IOError
        if the loaded Dataset is a Cached  :class:`~exosim.models.signal.Signal`.

    """
    metadata = {}
    if "metadata" in input.keys():
        metadata = recursively_read_dict_contents(input["metadata"])

    unit = u.Unit(input["data_units"][()].decode("utf-8"))

    spectral = (
        input["spectral"] * u.Unit(input["spectral_units"][()].decode("utf-8"))
        if "spectral" in input.keys()
        else None
    )

    time = (
        input["time"] * u.Unit(input["time_units"][()].decode("utf-8"))
        if "time" in input.keys()
        else None
    )

    spatial = (
        input["spatial"] * u.Unit(input["spatial_units"][()].decode("utf-8"))
        if "spatial" in input.keys()
        else None
    )

    # if 'datatype' in input.keys():
    #     class_name = input['datatype'][()].decode("utf-8").replace("<class '",
    #                                                                "").replace(
    #         "'>", "")
    #     klass = locate(class_name)
    # given the unit, check the right class to instantiate
    if unit == u.W / u.m**2 / u.um:
        klass = signal.Sed
    elif unit == u.W / u.m**2 / u.um / u.sr:
        klass = signal.Radiance
    elif unit == u.ct / u.s:
        klass = signal.CountsPerSecond
    elif unit == u.ct:
        klass = signal.Counts
    elif unit == u.adu:
        klass = signal.Adu
    elif unit == u.Unit(""):
        klass = signal.Dimensionless
    else:
        klass = signal.Signal

    if not input["cached"][()]:
        return klass(
            spectral=spectral,
            data=input["data"][()],
            data_units=unit,
            time=time,
            spatial=spatial,
            metadata=metadata,
        )
    else:
        raise OSError("impossible to load a cached dataset in a Signal class")


def recursively_read_dict_contents(input_dict: dict) -> dict:
    """
    Will recursive read a dictionary, initializing quantities and table from a dictionary read from an hdf5 file.

    Parameters
    ----------
    input_dict : dict
        dictionary read from hdf5

    Returns
    --------
    dict
        Dictionary we want to use

    """
    new_keys = [k for k in input_dict.keys()]
    output_dict = {}

    if all(elem in new_keys for elem in ["value", "unit"]):
        output_dict = input_dict["value"][()] * u.Unit(
            input_dict["unit"][()].decode("utf-8")
        )
        return output_dict

    elif any(".__table_column_meta__" in elem for elem in new_keys):
        table_keys = [
            elem for elem in new_keys if ".__table_column_meta__" in elem
        ]
        table_keys = (elem.split(".")[0] for elem in table_keys)
        for k in table_keys:
            table = read_table_hdf5(input_dict, k)
            output_dict[k] = table

    for key in new_keys:
        if isinstance(input_dict[key], h5py.Group):
            output_dict[key] = recursively_read_dict_contents(input_dict[key])
        elif isinstance(input_dict[key], h5py.Dataset):
            try:
                output_dict[key] = input_dict[key][()].decode("utf-8")
            except AttributeError:
                output_dict[key] = input_dict[key][()]

    return output_dict


def copy_file(
    in_object: Union[Group, Dataset], out_object: Union[Group, Dataset]
) -> None:
    """
    Recursively copy an HDF5 tree structure from one file to another.

    This function traverses the hierarchy of the input HDF5 object (`in_object`) and
    replicates it in the output HDF5 object (`out_object`). It can copy both groups and datasets,
    and also replicates all attributes.

    Parameters
    ----------
    in_object : Union[h5py.Group, h5py.Dataset]
        The input HDF5 object (either root, a subgroup, or a dataset).
    out_object : Union[h5py.Group, h5py.Dataset]
        The output HDF5 object (either root, a subgroup, or a dataset).

    Raises
    ------
    ValueError
        If an invalid object type is encountered.
    """

    # Copy attributes only once per group or dataset, to avoid overwriting
    for key, value in in_object.attrs.items():
        out_object.attrs[key] = value

    # Check if the input object is a dataset and skip further copying (base case for recursion)
    if isinstance(in_object, Dataset):
        return

    # Iterate through each key-value pair in the input object
    for key, in_obj in in_object.items():
        # Skip HDF5 Datatype objects
        if not isinstance(in_obj, Datatype):
            # Copy group
            if isinstance(in_obj, Group):
                out_obj = out_object.create_group(key)
                copy_file(in_obj, out_obj)  # Recursive call

            # Copy dataset
            elif isinstance(in_obj, Dataset):
                out_object.create_dataset(key, data=in_obj)

            # Raise an exception for unknown object types
            else:
                raise ValueError(f"Invalid object type {type(in_obj)}")
