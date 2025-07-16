import h5py
import numpy as np

from exosim.models import signal as signal
from exosim.tasks.load.loadOpticalElement import LoadOpticalElement
from exosim.utils import checks as checks


class LoadOpticalElementHDF5(LoadOpticalElement):
    """
    Class to load an optical element and return its radiance and efficiency using data from an HDF5 file.

    This class inherits from LoadOpticalElement and overrides the `model` and `_get_data` methods to use an HDF5 file
    for the optical element data.

    Returns
    --------
    :class:`~exosim.models.signal.Radiance`
        optical element radiance
    :class:`~exosim.models.signal.Dimensionless`
        optical element efficiency
    """

    def __init__(self):
        super().__init__()
        self.set_log_name()  # Ensure that the logger is properly initialized

    def _get_data(self, parameters, wl, tt, key, signal_type):
        """
        Load wavelength-dependent data from an HDF5 file, handling multiple structures.

        Parameters
        ----------
        parameters: dict
            dictionary containing the sources parameters.
        wl: :class:`~astropy.units.Quantity`
            wavelength grid.
        tt: :class:`~astropy.units.Quantity`
            time grid.
        key: str
            the key to read the specific dataset in the HDF5 group.
        signal_type: :class:`~exosim.models.signal.Signal`
            type of the signal to create.

        Returns
        -------
        :class:`~exosim.models.signal.Signal`
            extracted data as the specified signal type
        """
        hdf5_file_path = parameters["hdf5_file"]
        group_key = parameters["group_key"]
        read_key = parameters[key]

        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            # Check if the group exists
            if group_key not in hdf5_file:
                raise KeyError(
                    f"Group '{group_key}' not found in HDF5 file '{hdf5_file_path}'"
                )

            group = hdf5_file[group_key]

            # Handle two possible structures
            if read_key in group and isinstance(group[read_key], h5py.Dataset):
                self.debug(
                    f"Reading data from dataset '{read_key}' in group '{group_key}'"
                )
                # Default structure: read_key and wavelength_key directly in group
                parsed_wl = checks.check_units(
                    group[parameters["wavelength_key"]][:], "um", force=True
                )
                parsed_data = group[read_key][:]
            elif read_key in group.keys() and isinstance(
                group[read_key], h5py.Group
            ):
                self.debug(
                    f"Reading data from folder '{read_key}' in group '{group_key}'"
                )
                # Alternative structure: read_key as a folder containing datasets
                subgroup = group[read_key]
                if (
                    parameters["wavelength_key"] not in subgroup
                    or read_key not in subgroup
                ):
                    raise KeyError(
                        f"Datasets '{parameters['wavelength_key']}' and '{read_key}' "
                        f"not found in folder '{read_key}' within group '{group_key}'"
                    )
                parsed_wl = checks.check_units(
                    subgroup[parameters["wavelength_key"]][:], "um", force=True
                )
                parsed_data = subgroup[read_key][:]
            else:
                raise KeyError(
                    f"Dataset or folder '{read_key}' not found in group '{group_key}'"
                )

        # Create and process the signal
        extracted_data = signal_type(spectral=parsed_wl, data=parsed_data)
        extracted_data.spectral_rebin(wl)
        extracted_data.temporal_rebin(tt)

        self.debug(
            f"Extracted data for key '{read_key}': {extracted_data.data}"
        )
        return extracted_data
