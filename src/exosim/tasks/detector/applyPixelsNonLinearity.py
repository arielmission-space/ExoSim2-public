from copy import deepcopy

import h5py
import numpy as np

from exosim.models.signal import Counts
from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.types import HDF5OutputType


class ApplyPixelsNonLinearity(Task):
    r"""
    Given the pixel non-linearity parameters,
    this Task correct the ideal measured signal rate to the pixel linearity.

    .. math::
        Q_{det} = Q \cdot (a + b \cdot Q + c \cdot Q^2 + d \cdot Q^3 + e \cdot Q^4)

    The input is a dictionary with a `map` keyword containg an array with the coefficients for each pixel.
    The map shape is (n_pixels_x, n_pixels_y, coefficient order).

    The user can list any number of coefficients, that will be parsed in the following model

    .. math::
        Q_{det} = Q \cdot (a_0 + \sum_i a_i \cdot Q^i)

    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        """

        self.add_task_param("subexposures", " ")
        self.add_task_param("parameters", "channel non linearity dictionary")

    def execute(self):
        self.info("applying pixel non-linearity map")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")

        self.model(subexposures, parameters)

    def model(self, subexposures: Counts, parameters: dict) -> None:
        pnl_map = parameters["map"][:]

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying pixel non-linearity map"
        ):
            data = deepcopy(subexposures.dataset[chunk])
            npl = 0
            for i in range(0, len(pnl_map)):
                npl += data ** (i + 1) * np.repeat(
                    pnl_map[np.newaxis, i], data.shape[0], axis=0
                )
            subexposures.dataset[chunk] = npl
            subexposures.output.flush()
