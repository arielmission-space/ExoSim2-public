import h5py

import exosim.output as output
from exosim.output.hdf5.utils import recursively_read_dict_contents
from exosim.tasks.task import Task


class LoadPixelsNonLinearityMap(Task):
    """
    Loads the pixels non-linearuty map

    Returns
    --------
    dict
        channel non linearity map

    Raises
    ------
    KeyError:
        if the output do not have the `map` key

    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("loading pixels non-linearity map")
        parameters = self.get_task_param("parameters")

        pnl_dict_map = self.model(parameters)

        # checking output
        if "map" not in pnl_dict_map:
            self.error("map missing in pixel non-linearity map")
            raise KeyError("map missing in pixel non-linearity map")

        self.debug("pnl_variation_map: {}".format(pnl_dict_map["map"]))

        output_file = self.get_task_param("output")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                output_file.store_dictionary(
                    pnl_dict_map, "pixel_non_linearity"
                )

        self.set_output(pnl_dict_map)

    def model(self, parameters: dict) -> dict:
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        dict
            channel pixel non-linearity map

        """
        file_name = parameters["detector"]["pnl_filename"]
        ch_name = parameters["value"]

        with h5py.File(file_name, "r") as f:
            pnl_dict = recursively_read_dict_contents(f[ch_name])

        return pnl_dict
