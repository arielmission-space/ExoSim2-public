import h5py

import exosim.models.signal as signal
import exosim.output as output
from exosim.output.hdf5.utils import load_signal
from exosim.tasks.task import Task


class LoadQeMap(Task):
    """
    Loads the Quantum efficiency map

    Returns
    --------
    :class:`~exosim.models.signal.Signal`
        channel responsivity variation map

    Raises
    ------
    TypeError:
        if the output is not a :class:`~exosim.models.signal.Signal` class


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
        time: :class:`~astropy.units.Quantity`
            time grid
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("parameters", "parameters dict")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("loading quantum efficiency variation map")
        parameters = self.get_task_param("parameters")
        tt = self.get_task_param("time")

        qe_map = self.model(parameters, tt)

        # checking output
        if not isinstance(qe_map, signal.Signal):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.debug("qe_variation_map: {}".format(qe_map.data))

        output_file = self.get_task_param("output")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                qe_map.write(output_file, "qe_variation_map")

        self.set_output(qe_map)

    def model(self, parameters, time):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            channel responsivity

        """
        file_name = parameters["detector"]["qe_map_filename"]

        with h5py.File(file_name, "r") as f:
            qe = load_signal(f[parameters["value"]])

        return qe
