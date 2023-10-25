import numpy as np
from astropy import units as u
from astropy.modeling.physical_models import BlackBody

import exosim.models.signal as signal
from exosim.tasks.task import Task


class CreateCustomSource(Task):
    """
    Creates a custom SED from input parameters.

    Returns
    -------
    :class:`~exosim.models.signal.Sed`
        Star Sed

    Raises
    ------
    TypeError:
        if the output is not a :class:`~exosim.models.signal.Sed` class

    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        -----------
        parameters: dict
            dictionary containing the source parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """
        self.add_task_param("parameters", "source parameters")

    def execute(self):
        self.info("Create custom sed")
        parameters = self.get_task_param("parameters")

        sed = self.model(parameters)

        # checking output
        if not isinstance(sed, signal.Sed):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.debug("sed: {}".format(sed.data))

        self.set_output(sed)

    def model(self, parameters):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the source parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`.

        Returns
        --------
        :class:`~exosim.models.signal.Sed`
            source sed

        """
        R = parameters["R"].si
        D = parameters["D"].si
        T = parameters["T"].si

        wl_min = parameters["wl_min"]
        wl_max = parameters["wl_max"]
        n_points = parameters["n_points"]
        wl = np.linspace(wl_min, wl_max, n_points)

        omega_star = np.pi * (R / D) ** 2 * u.sr
        bb = BlackBody(T)
        bb_ = bb(wl).to(u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl))
        sed = signal.Sed(spectral=wl, data=omega_star * bb_)

        return sed
