from copy import deepcopy

import numpy as np

from exosim.output import Output
from exosim.tasks.task import Task
from exosim.utils import RunConfig
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks


class AddNormalReadNoise(Task):
    """
    This Task simulates the read noise as a normal distribution which parameters can be defined in the configuration file.

    If it is not explicitly stated, the distribution mean is set to 0.
    A different realisation of the same distribution is added to each pixel of each sub-exposure.
    If an output group is provided, it saves all the random seeds used.

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
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("parameters", "channel parameters dictionary")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("adding read noise")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        output = self.get_task_param("output")

        read_noise_sigma = parameters["detector"]["read_noise_sigma"].astype(
            np.float64
        )
        read_noise_sigma = check_units(read_noise_sigma, "ct").value

        random_seeds = []

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding read noise"
        ):
            data = deepcopy(subexposures.dataset[chunk])
            subexposures.dataset[chunk] = (
                data
                + RunConfig.random_generator.normal(
                    0, read_noise_sigma, data.shape
                ).astype(np.float64)
            )
            subexposures.output.flush()
            random_seeds.append(RunConfig.random_seed)

        if output:
            if issubclass(output.__class__, Output):
                out_grp = output.create_group("read noise")
                out_grp.write_list("random_seed", random_seeds)
                out_grp.write_array(
                    "chunks_index", np.arange(len(random_seeds))
                )
