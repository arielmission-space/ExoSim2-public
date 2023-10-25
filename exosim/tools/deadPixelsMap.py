import os

import numpy as np
from astropy.io import ascii
from astropy.table import Table

from .exosimTool import ExoSimTool
from exosim.utils import RunConfig


class DeadPixelsMap(ExoSimTool):
    """
    Produces the channel dead pixel map

    Returns
    --------
    dict
        channels' dead pixels maps

    Raises
    ------
    TypeError:
        if the output is not a :class:`~astropy.table.Table` class

    Examples
    ----------

    >>> import exosim.tools as tools
    >>>
    >>> tools.DeadPixelsMap(options_file='tools_input_example.xml',
    >>>                     output='./')
    """

    def __init__(self, options_file, output=None):
        """
        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: str (optional)
            output directory
        """
        super().__init__(options_file)

        self.info("creating dead pixel map")

        for ch in self.ch_list:
            dp_map = self.model(self.ch_param[ch])

            self.results.update({ch: dp_map})

            self.debug("dead pixel map: {}".format(dp_map))

            if output:
                fname = os.path.join(output, "dp_map_{}.csv".format(ch))
                ascii.write(
                    dp_map,
                    fname,
                    format="ecsv",
                    overwrite=True,
                    delimiter=",",
                )
                self.info("dead pixels map stored in {}".format(fname))

    def model(self, parameters):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        :class:`~astropy.table.Table`
            channel dead pixel coordinates

        """

        # detector map
        y_size = parameters["detector"]["spatial_pix"]
        x_size = parameters["detector"]["spectral_pix"]

        if "dp_sigma" in parameters["detector"].keys():
            number_dead_pixels = RunConfig.random_generator.normal(
                loc=parameters["detector"]["dp_mean"],
                scale=parameters["detector"]["dp_sigma"],
                size=1,
            ).astype(int)
        else:
            number_dead_pixels = parameters["detector"]["dp_mean"]

        x_coords = RunConfig.random_generator.integers(
            0, x_size - 1, size=number_dead_pixels
        )
        y_coords = RunConfig.random_generator.integers(
            0, y_size - 1, size=number_dead_pixels
        )

        tab = Table()
        tab["spatial_coords"] = y_coords
        tab["spectral_coords"] = x_coords

        return tab
