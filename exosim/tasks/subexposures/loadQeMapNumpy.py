import numpy as np

import exosim.models.signal as signal
from exosim.tasks.subexposures import LoadQeMap


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
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            channel responsivity

        """
        file_name = parameters["detector"]["qe_map_filename"]
        qe_data = np.load(file_name)

        qe = signal.Signal(data=qe_data)

        return qe
