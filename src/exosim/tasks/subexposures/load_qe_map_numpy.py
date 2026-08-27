import numpy as np

import exosim.models.signal as signal

from .load_qe_map import LoadQeMap


class LoadQeMapNumpy(LoadQeMap):
    """
    Loads the Quantum efficiency map from a NPY file (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_).

    Returns
    --------
    :class:`~exosim.models.signal.Signal`
        channel responsivity variation map

    Raises
    ------
    TypeError:
        if the output is not a :class:`~exosim.models.signal.Signal` class

    """

    def model(self, parameters, time):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            channel responsivity

        """
        file_name = parameters["detector"]["qe_map_filename"]
        qe_data = np.load(file_name)

        return signal.Signal(data=qe_data)
