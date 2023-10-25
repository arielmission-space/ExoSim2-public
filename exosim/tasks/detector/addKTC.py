import numpy as np
from numba import jit

from exosim.output import Output
from exosim.tasks.task import Task
from exosim.utils import RunConfig
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks


class AddKTC(Task):
    """
    It adds the ktc bias to the sub-exposures.
    This Task produces a new random offset map for each ramp,
    and it adds it to the sub-exposures of the same ramp.
    If an output group is provided, it saves all the random seeds used.

    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        state_machine: :class:`numpy.ndarray`
            array indicating the exposures number of each sub-exposure.
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("parameters", "channel parameters dictionary")
        self.add_task_param("state_machine", "ramp state machine")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("adding reset bias")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        state_machine = self.get_task_param("state_machine")
        output = self.get_task_param("output")

        ktc_std = parameters["detector"]["ktc_sigma"].astype(np.float64)
        ktc_std = check_units(ktc_std, "ct").value

        offset = RunConfig.random_generator.normal(
            0,
            ktc_std,
            (subexposures.dataset.shape[1], subexposures.dataset.shape[2]),
        )

        random_seeds = []

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding ktc noise"
        ):
            if (
                state_machine[chunk[0].start - 1]
                != state_machine[chunk[0].start]
            ):
                offset = RunConfig.random_generator.normal(
                    0,
                    ktc_std,
                    (
                        subexposures.dataset.shape[1],
                        subexposures.dataset.shape[2],
                    ),
                )

            subexposures.dataset[chunk], offset = self.add_offset(
                subexposures.dataset[chunk],
                state_machine[chunk[0]],
                offset,
                ktc_std,
            )
            subexposures.output.flush()
            random_seeds.append(RunConfig.random_seed)

        if output:
            if issubclass(output.__class__, Output):
                out_grp = output.create_group("kTC noise")
                out_grp.write_list("random_seed", random_seeds)
                out_grp.write_array(
                    "chunks_index", np.arange(len(random_seeds))
                )

    @staticmethod
    @jit(nopython=True)
    def add_offset(dset, state_machine, offset, bias_std):
        dset[0] += offset
        for i in range(1, dset.shape[0]):
            if state_machine[i] == state_machine[i - 1]:
                dset[i] += offset
            else:
                offset = np.random.normal(0, bias_std, dset[i].shape)
                dset[i] += offset
        return dset, offset
