import numpy as np
from astropy.io import ascii
from numba import jit, prange

from exosim.output import Output
from exosim.tasks.detector import ApplyDeadPixelsMap
from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks


class ApplyDeadPixelMapNumpy(ApplyDeadPixelsMap):
    """
    It masks the dead pixel in the array given a numpy map.
    The input must be a NPY format file (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_) containing a boolean map marking with True the dead pixels.
    The map should be indicated under ``dp_map_filename`` keyword.
    """

    def model(self, subexposures, parameters, output):
        dead_pixels_map = np.ones(
            (subexposures.shape[1], subexposures.shape[2])
        )

        dead_map = np.load(parameters["detector"]["dp_map_filename"])

        if output:
            if issubclass(output.__class__, Output):
                out_grp = output.create_group("dead_pixel_map")
                out_grp.write_array("dead_pixel_map", dead_map)

        dead_pixels_map = np.invert(dead_map.astype(bool)).astype(int)

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying dead pixel map"
        ):
            subexposures.dataset[chunk] = self.add_dead_pixels(
                subexposures.dataset[chunk], dead_pixels_map
            )
            subexposures.output.flush()

    @staticmethod
    @jit(nopython=True, parallel=True)
    def add_dead_pixels(
        ndrs: np.ndarray, dead_pixels_map: np.ndarray
    ) -> np.ndarray:
        ndrs = ndrs.astype(np.float64)
        dead_pixels_map = dead_pixels_map.astype(np.float64)

        output = np.empty_like(ndrs)

        for t in prange(ndrs.shape[0]):
            output[t] = ndrs[t] * dead_pixels_map

        return output
