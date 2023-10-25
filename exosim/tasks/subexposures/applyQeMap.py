import astropy.units as u
import numpy as np

from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.iterators import searchsorted


class ApplyQeMap(Task):
    """
    It applies the quantum efficiency variation map to the sub exposures.

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
            sub-exposure cached signal class
        qe_map: :class:`~exosim.models.signal.Signal`
            channel responsivity variation map
        """

        self.add_task_param("subexposures", "sub-exposure cached signal clas")
        self.add_task_param(
            "qe_map",
            "channel responsivity variation map",
        )

    def execute(self):
        subexposures = self.get_task_param("subexposures")
        qe_map = self.get_task_param("qe_map")

        subexposures_time = (subexposures.time * subexposures.time_units).to(
            u.hr
        )
        qe_map_time = qe_map.time
        if not isinstance(qe_map_time, u.Quantity):
            qe_map_time = qe_map.time * qe_map.time_units
        qe_map_time.to(u.hr)
        qe = qe_map.data.astype(np.float64)
        index = searchsorted(qe_map_time, subexposures_time)

        self.debug("time indexes: {}".format(index))
        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying qe variation map"
        ):
            subexposures.dataset[chunk] = self.apply_qe(
                subexposures.dataset[chunk], qe, index[chunk[0]]
            )
            subexposures.output.flush()
        subexposures.metadata["qe_variation_map_indexes"] = index
        self.set_output(subexposures)

    @staticmethod
    def apply_qe(se, qe, index):
        for t in range(se.shape[0]):
            se[t] *= qe[index[t]]
        return se
