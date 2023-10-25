import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

import exosim.models.signal as signal
import exosim.utils as utils
from .exosimTool import ExoSimTool
from exosim.output import SetOutput
from exosim.utils import check_units
from exosim.utils import RunConfig


class QuantumEfficiencyMap(ExoSimTool):
    """
    Produces the channels quantum efficiency variation map

    Returns
    --------
    dict:
        channels' quantum efficiency map

    Raises
    ------
    TypeError:
        if the output is not a :class:`~exosim.models.signal.Signal` class

    Examples
    ----------

    >>> import exosim.tools as tools
    >>>
    >>> results = tools.QuantumEfficiencyMap(options_file='tools_input_example.xml',
    >>>                                      output='output_qe_map.h5')
    """

    def __init__(self, options_file, output=None):
        """
        Parameters
        __________
        parameters: str or dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: str (optional)
            output file
        """
        super().__init__(options_file)

        self.info("creating qe map")

        time_grid = utils.grids.time_grid(
            self.options["time_grid"]["start_time"],
            self.options["time_grid"]["end_time"],
            self.options["time_grid"]["low_frequencies_resolution"],
        )
        output = SetOutput(output)

        with output.use() as out:
            for ch in self.ch_list:
                qe_map = self.model(self.ch_param[ch], time_grid)

                self.results.update({ch: qe_map})

                self.debug("qe variation map: {}".format(qe_map.data))

                qe_map.write(out, ch)

    def model(self, parameters, time):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            channel quantum efficiency map

        """

        # variance map
        variance_map = RunConfig.random_generator.normal(
            1,
            parameters["detector"]["qe_sigma"],
            (
                1,
                parameters["detector"]["spatial_pix"],
                parameters["detector"]["spectral_pix"],
            ),
        )
        t = [0] * u.hr
        # temporal variation
        if "qe_aging_factor" in parameters["detector"].keys():
            t0 = parameters["detector"]["qe_aging_time_scale"]
            t0 = check_units(t0, u.hr, force=True)

            # creates the aging effect
            qe_aging = RunConfig.random_generator.normal(
                1,
                parameters["detector"]["qe_aging_factor"],
                (
                    1,
                    parameters["detector"]["spatial_pix"],
                    parameters["detector"]["spectral_pix"],
                ),
            )

            # remove values greater than 1, because QE should become smaller
            qe_aging[qe_aging > 1] = 2 - qe_aging[qe_aging > 1]

            # create the map evolution over time
            variance_map = np.vstack((variance_map, variance_map * qe_aging))
            t = np.hstack((t, t0))
            variance_map = variance_map.reshape(
                t.size, variance_map.shape[1] * variance_map.shape[2]
            )
            f = interp1d(t, variance_map, fill_value="extrapolate", axis=0)
            variance_map = f(time)
            variance_map = variance_map.reshape(
                time.size,
                parameters["detector"]["spatial_pix"],
                parameters["detector"]["spectral_pix"],
            )
            t = time

        qe_map = signal.Dimensionless(
            data=variance_map,
            spectral=np.arange(0, parameters["detector"]["spectral_pix"])
            * u.pix,
            time=t,
        )

        qe_map.temporal_rebin(time)

        return qe_map
