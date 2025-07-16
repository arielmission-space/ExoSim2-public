from copy import deepcopy

import astropy.units as u
import numpy as np

from exosim.models.signal import Signal
from exosim.tasks.task import Task
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.operations import operate_over_axis
from exosim.utils.types import ArrayType


class AddDarkCurrentMapNumpy(Task):
    """
    It adds a dark current map to the array.
    The map must be indicated under the `dc_map_filename` keyword.
    The dark current is loaded from a NPY format file (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_).
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

        dc_file = parameters["detector"]["dc_map_filename"]
        dc = np.load(dc_file)
        dc = check_units(dc, "ct/s", force=True)

        if dc.shape != subexposures.dataset[0].shape:
            self.error(
                "wrong shape: expected {} but got {}".format(
                    subexposures.dataset[0].shape, dc.shape
                )
            )
            raise OSError()

        self.info("dark current map loaded")

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding dark current"
        ):
            dc_map = (
                dc[np.newaxis, :, :].value
                * integration_times[chunk[0], np.newaxis, np.newaxis]
                .to(u.s)
                .value
            )

            data = deepcopy(subexposures.dataset[chunk])

            subexposures.dataset[chunk] = data + dc_map

            subexposures.output.flush()
