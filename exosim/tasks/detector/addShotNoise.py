from copy import deepcopy

import numpy as np

from exosim.output import Output
from exosim.tasks.task import Task
from exosim.utils import RunConfig
from exosim.utils.iterators import iterate_over_chunks


class AddShotNoise(Task):
    """
    It adds the shot noise to the sub-exposures.
    The shot noise is added as a Poisson noise to the sub-exposures.
    If an output group is provided, it saves all the random seeds used.
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("adding shot noise")
        subexposures = self.get_task_param("subexposures")
        output = self.get_task_param("output")

        random_seeds = []
        warning_raised = False
        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding shot noise"
        ):
            data = deepcopy(subexposures.dataset[chunk])
            idx = np.where(data <= 0)
            if len(idx[0]) > 0:
                data[idx] = 1e-10
                if not warning_raised:
                    self.warning(
                        "Negative and zero pixels found: values replaced with 1e-10"
                    )
                    warning_raised = True

            subexposures.dataset[chunk] = RunConfig.random_generator.poisson(
                data
            )
            subexposures.output.flush()
            random_seeds.append(RunConfig.random_seed)

        if output:
            if issubclass(output.__class__, Output):
                out_grp = output.create_group("shot noise")
                out_grp.write_list("random_seed", random_seeds)
                out_grp.write_array(
                    "chunks_index", np.arange(len(random_seeds))
                )
