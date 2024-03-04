from copy import deepcopy

import astropy.units as u
import numpy as np

from exosim.models.signal import Signal
from exosim.tasks.task import Task
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.operations import operate_over_axis
from exosim.utils.types import ArrayType


class AddConstantDarkCurrent(Task):
    """
    It adds constant dark current to all the pixel in the array.
    The dark current is loaded from the parameters

    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        integration_times: :class:`~astropy.units.Quantity`
            sub-exposures integration times
        outputs: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("parameters", "channel parameters dictionary")
        self.add_task_param(
            "integration_times",
            "subexposures integration times",
        )
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("adding dark current")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        integration_times = self.get_task_param("integration_times")
        output = self.get_task_param("output")

        self.model(subexposures, parameters, integration_times, output)

    def model(
        self,
        subexposures: Signal,
        parameters: dict,
        integration_times: ArrayType,
        output=None,
    ) -> None:
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        integration_times: :class:`~astropy.units.Quantity`
            sub-exposures integration times
        outputs: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        dc = parameters["detector"]["dc_mean"].astype(np.float64)
        dc = check_units(dc, "ct/s")

        self.info("dark current: {}".format(dc))

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding dark current"
        ):
            dc_map = (
                dc.to(u.ct / u.s).value
                * integration_times[chunk[0]].to(u.s).value
            )
            data = deepcopy(subexposures.dataset[chunk])

            subexposures.dataset[chunk] = operate_over_axis(
                data, dc_map, axis=0, operation="+"
            )
            subexposures.output.flush()
