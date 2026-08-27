from copy import deepcopy

import numpy as np

from exosim.output import Output
from exosim.utils import RunConfig
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks

from .add_read_noise import AddNormalReadNoise


class AddReadNoiseMapNumpy(AddNormalReadNoise):
    """
    This Task simulates the read noise as a normal distribution which parameters can be defined with a map indicated in the configuration file under `read_noise_filename` keyword.
    The input must be a NPY format file (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_) containing the map of the distribution standard deviation for each pixel.

    A different realisations of the same distribution is added to each pixel of each sub-exposure.
    If an output group is provided, it saves all the random seeds used.

    """

    def execute(self):
        self.info("adding read noise")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        output = self.get_task_param("output")

        read_noise_file = parameters["detector"]["read_noise_filename"]
        read_noise_sigma = np.load(read_noise_file)
        read_noise_sigma = check_units(read_noise_sigma, "ct", force=True)

        if read_noise_sigma.shape != subexposures.dataset[0].shape:
            self.error(
                f"wrong shape: expected {subexposures.dataset[0].shape} but got {read_noise_sigma.shape}"
            )
            raise OSError("Map dimensions do not match signal shape")

        random_seeds = []

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="adding read noise"
        ):
            data = deepcopy(subexposures.dataset[chunk])
            subexposures.dataset[chunk] = data + RunConfig.random_generator.normal(
                0, read_noise_sigma, data.shape
            ).astype(np.float64)
            subexposures.output.flush()
            random_seeds.append(RunConfig.random_seed)

        if output and issubclass(output.__class__, Output):
            out_grp = output.create_group("read noise")
            out_grp.write_list("random_seed", random_seeds)
            out_grp.write_array("chunks_index", np.arange(len(random_seeds)))
