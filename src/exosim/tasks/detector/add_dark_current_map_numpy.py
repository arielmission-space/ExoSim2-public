from copy import deepcopy

import astropy.units as u
import numpy as np

from exosim.models.signal import Signal
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.types import ArrayType

from .add_constant_dark_current import AddConstantDarkCurrent


class AddDarkCurrentMapNumpy(AddConstantDarkCurrent):
    """
    It adds a dark current map to the array.
    The map must be indicated under the `dc_map_filename` keyword.
    The dark current is loaded from a NPY format file (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_.).
    """

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
                f"wrong shape: expected {subexposures.dataset[0].shape} but got {dc.shape}"
            )
            raise OSError("Map dimensions do not match signal shape")

        self.info("dark current map loaded")

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding dark current"
        ):
            dc_map = (
                dc[np.newaxis, :, :].value
                * integration_times[chunk[0], np.newaxis, np.newaxis].to(u.s).value
            )

            data = deepcopy(subexposures.dataset[chunk])

            subexposures.dataset[chunk] = data + dc_map

            subexposures.output.flush()
