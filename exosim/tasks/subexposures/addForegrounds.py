from copy import deepcopy

import astropy.units as u
import numpy as np

from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.iterators import searchsorted


class AddForegrounds(Task):
    """
    It adds the foregrounds the sub exposures.

    Returns
    --------
    :class:`~exosim.models.signal.Counts`
        sub-exposure cached signal class
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal class
        frg_focal_plane: :class:`~exosim.models.signal.Signal`
            channel foreground focal plane
        integration_time: :class:`~astropy.units.Quantity`
            sub-exposures integration times

        """

        self.add_task_param(
            "subexposures", "sub-exposures cached signal class"
        )
        self.add_task_param(
            "frg_focal_plane",
            "channel foreground focal plane",
        )
        self.add_task_param(
            "integration_time",
            "sub-exposures integration times",
        )

    def execute(self):
        self.info("adding foregrounds")
        subexposures = self.get_task_param("subexposures")
        frg_focal_plane = self.get_task_param("frg_focal_plane")
        integration_time = self.get_task_param("integration_time")

        osf = frg_focal_plane.metadata["oversampling"]

        ndrs_time = (subexposures.time * subexposures.time_units).to(u.hr)
        frg = deepcopy(frg_focal_plane.data.astype(np.float64))
        frg_time = (frg_focal_plane.time * frg_focal_plane.time_units).to(u.hr)
        index = searchsorted(frg_time, ndrs_time)
        self.debug("time indexes: {}".format(index))

        # bin down the frg to the subexposure size (integrate over pixels)
        frg = frg.reshape(
            (frg.shape[0], int(frg.shape[1] / osf), osf, frg.shape[2])
        ).sum(axis=2)
        frg = frg.reshape(
            (
                frg.shape[0],
                frg.shape[1],
                int(frg.shape[2] / osf),
                osf,
            )
        ).sum(axis=3)

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding foregrounds"
        ):
            subexposures.dataset[chunk] = self.add_frg(
                subexposures.dataset[chunk],
                frg,
                index[chunk[0]],
                integration_time[chunk[0]].to(u.s).value,
            )
            subexposures.output.flush()
        subexposures.metadata["frg_focal_plane_time_indexes"] = index
        self.set_output(subexposures)

    @staticmethod
    def add_frg(ndrs, frg, index, integration_time):
        for t in range(ndrs.shape[0]):
            ndrs[t] = ndrs[t] + frg[index[t]] * integration_time[t]
        return ndrs
