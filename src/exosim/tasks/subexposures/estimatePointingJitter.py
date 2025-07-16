import astropy.units as u
import numpy as np

import exosim.output as output
import exosim.utils as utils
from exosim.tasks.task import Task
from exosim.utils import RunConfig
from exosim.utils.checks import check_units


class EstimatePointingJitter(Task):
    """
    Produces the telescope pointing jitter expressed as deg on the line of sight.

    Returns
    --------
    :class:`~astropy.units.Quantity`,
        pointing jitter in the spatial and spectral direction expressed in units of :math:`deg`.
    :class:`~astropy.units.Quantity`
        pointing jitter in the spatial and spectral direction expressed in units of :math:`deg`.
    :class:`~astropy.units.Quantity`
        pointing jitter in the spatial and spectral direction expressed in units of :math:`deg`.

    Raises
    ------
    TypeError:
        if the output is not a :class:`~astropy.units.Quantity` class

    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """
        self.add_task_param("parameters", "main parameters dict")
        self.add_task_param("output_file", "output file", None)

    def execute(self):
        self.info("estimating pointing jitter")
        main_parameters = self.get_task_param("parameters")

        jitter_spa, jitter_spe, jitter_time = self.model(main_parameters)

        # checking output
        if not isinstance(jitter_spa, u.Quantity):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        if not isinstance(jitter_spe, u.Quantity):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        if not isinstance(jitter_time, u.Quantity):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        # converting into dec
        jitter_spa = utils.check_units(jitter_spa, u.deg)
        jitter_spe = utils.check_units(jitter_spe, u.deg)
        jitter_time = utils.check_units(jitter_time, u.s)

        self.debug("pointing jitter: {} {}".format(jitter_spa, jitter_spe))
        output_file = self.get_task_param("output_file")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                store_dict = {
                    "jitter_spa": jitter_spa,
                    "jitter_spe": jitter_spe,
                    "jitter_time": jitter_time,
                }
                output_file.store_dictionary(store_dict, "pointing_jitter")

        self.set_output([jitter_spa, jitter_spe, jitter_time])

    def model(self, parameters):
        """
        This default model builds the pointing jitter using random values.
        Starting from the spectral and spatial standard deviations, expressed as angles,
        it computes the pointing position as normally distributed around zero in the two direction.
        The input dictionary, under the `jitter` keyword, must contain `spatial` and `spectral` keyword for the standard deviation.
        The time grid is built using the `high_frequencies_resolution` under the `time_grid` keyword.


        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        :class:`~astropy.units.Quantity`
            pointing jitter in the spatial direction expressed in units of :math:`deg`.
         :class:`~astropy.units.Quantity`)
            pointing jitter in the spectral direction expressed in units of :math:`deg`.
         :class:`~astropy.units.Quantity`)
            pointing jitter timeline expressed in units of :math:`s`.
        """
        jitter_time_step = check_units(
            parameters["jitter"]["frequency_resolution"], "s"
        )

        jitter_time = utils.grids.time_grid(
            parameters["time_grid"]["start_time"],
            parameters["time_grid"]["end_time"]
            + 1 * parameters["time_grid"]["end_time"].unit,
            jitter_time_step,
        )

        jitter_spa = (
            RunConfig.random_generator.normal(
                0, parameters["jitter"]["spatial"].value, jitter_time.size
            )
            * parameters["jitter"]["spatial"].unit
        )
        jitter_spe = (
            RunConfig.random_generator.normal(
                0, parameters["jitter"]["spectral"].value, jitter_time.size
            )
            * parameters["jitter"]["spectral"].unit
        )

        return jitter_spa, jitter_spe, jitter_time
