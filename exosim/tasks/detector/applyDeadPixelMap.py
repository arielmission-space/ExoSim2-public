import numpy as np
from astropy.io import ascii
from numba import jit
from numba import prange

from exosim.output import Output
from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks


class ApplyDeadPixelsMap(Task):
    """
    It masks the dead pixel in the array given their coordinates.
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        outputs: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("subexposures", " ")
        self.add_task_param("parameters", "channel parameters dictionary")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("applying dead pixel map")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        output = self.get_task_param("output")

        self.model(subexposures, parameters, output)

    def model(self, subexposures, parameters, output):
        dead_pixels_map = np.ones(
            (subexposures.shape[1], subexposures.shape[2])
        )

        dead_coords = ascii.read(parameters["detector"]["dp_map"])

        if output:
            if issubclass(output.__class__, Output):
                out_grp = output.create_group("dead_pixel_map")
                out_grp.write_table("dead_pixel_map", dead_coords)

        for x, y in dead_coords["spectral_coords", "spatial_coords"]:
            dead_pixels_map[y, x] = 0.0

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying dead pixel map"
        ):
            subexposures.dataset[chunk] = self.add_dead_pixels(
                subexposures.dataset[chunk], dead_pixels_map
            )
            subexposures.output.flush()

    @staticmethod
    @jit(nopython=True, parallel=True)
    def add_dead_pixels(ndrs, dead_pixels_map):
        for t in prange(ndrs.shape[0]):
            ndrs[t] = ndrs[t] * dead_pixels_map
        return ndrs
