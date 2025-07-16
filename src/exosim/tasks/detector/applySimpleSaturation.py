from copy import deepcopy

import numpy as np

from exosim.models.signal import Counts
from exosim.tasks.task import Task
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks


class ApplySimpleSaturation(Task):
    r"""
    This Task applies a simple model of saturation to pixel counts.
    If the counts in a pixel are higher than the well depth, the counts are set to the well capacity.
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
        self.add_task_param("parameters", "channel parameters dictionary")

    def execute(self):
        self.info("applying saturation")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")

        self.model(subexposures, parameters)

    def model(self, subexposures: Counts, parameters: dict) -> None:
        """
        Applies the saturation model to the subexposures.

        Parameters
        ----------
        subexposures : Counts
            The subexposures to be saturated.
        parameters : dict
            Dictionary containing saturation parameters like 'detector' and 'well_depth'.

        Returns
        -------
        None
        """

        sat = parameters["detector"]["well_depth"]
        sat = check_units(sat, "ct", force=True).value

        self.info("saturation: {}".format(sat))
        self.info("max counts: {}".format(np.max(subexposures.dataset)))

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying pixel saturation"
        ):
            data = deepcopy(subexposures.dataset[chunk])
            data[data > sat] = sat

            subexposures.dataset[chunk] = data
            subexposures.output.flush()
