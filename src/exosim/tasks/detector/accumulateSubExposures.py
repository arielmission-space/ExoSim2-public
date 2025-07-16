from typing import List

import numpy as np
from numba import jit

from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks


class AccumulateSubExposures(Task):
    """
    It accumulates sub-exposures of the same ramp.
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        state_machine: :class:`numpy.ndarray`
            array indicating the exposures number of each sub-exposure.
            `
        """

        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("state_machine", "ramp state machine")

    def execute(self) -> None:
        self.info("accumulating sub-exposures")
        subexposures = self.get_task_param("subexposures")
        state_machine = self.get_task_param("state_machine")

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="accumulating sub-exposures"
        ):
            if (
                state_machine[chunk[0].start - 1]
                == state_machine[chunk[0].start]
            ):
                offset = subexposures.dataset[chunk[0].start - 1]
            else:
                offset = np.zeros_like(
                    subexposures.dataset[chunk[0].start - 1]
                )

            subexposures.dataset[chunk] = self.sub_exposures_cumsum(
                subexposures.dataset[chunk], state_machine[chunk[0]], offset
            )
            subexposures.output.flush()

    @staticmethod
    @jit(nopython=True)
    def sub_exposures_cumsum(
        dset: np.ndarray, state_machine: List[int], offset: List[int]
    ) -> np.ndarray:
        dset[0] += offset
        for i in range(1, dset.shape[0]):
            if state_machine[i] == state_machine[i - 1]:
                dset[i] += dset[i - 1]
        return dset
