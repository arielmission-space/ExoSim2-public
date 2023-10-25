import numpy as np
from astropy import units as u
from astropy.modeling.physical_models import BlackBody

from exosim.models import signal as signal
from exosim.tasks.task import Task
from exosim.utils import checks as checks


class LoadOpticalElement(Task):
    """
    Abstract class to load an optical element and returns the element self emission
    and it optical efficiency.


    Returns
    --------
    :class:`~exosim.models.signal.Radiance`
        optical element radiance
    :class:`~exosim.models.signal.Dimensionless`
        optical element efficiency

    Raises
    ------
    TypeError:
        if the outputs are not the right :class:`~exosim.models.signal.Signal` subclasses


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
            dictionary contained the optical element parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid
        """

        self.add_task_param("parameters", "optical element parameters dict")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")

    def execute(self):
        parameters = self.get_task_param("parameters")
        wl = self.get_task_param("wavelength")
        tt = self.get_task_param("time")

        radiance, efficiency = self.model(
            parameters=parameters, wavelength=wl, time=tt
        )

        # checking output
        if not isinstance(radiance, signal.Radiance):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        if not isinstance(efficiency, signal.Dimensionless):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.set_output([radiance, efficiency])

    def model(self, parameters, wavelength, time):
        r"""
        It loads the indicated columns from the data file.

        If emissivity and temperature are provided the radiance is estimated as

        .. math::

            I_{surf}(\lambda) = \epsilon \cdot BB(\lambda, T)

        where :math:`\epsilon` is the indicated emissivity and :math:`BB(\lambda, T)` is the Planck black body law.


        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Radiance`
            optical element radiance
        :class:`~exosim.models.signal.Dimensionless`
            optical element efficiency

        """

        # load efficiency if provided
        if "efficiency_key" in parameters.keys():
            efficiency = self._get_data(
                parameters,
                wavelength,
                time,
                "efficiency_key",
                signal.Dimensionless,
            )
        # else we set it to 1
        else:
            efficiency = signal.Dimensionless(
                spectral=wavelength, data=np.ones(len(wavelength))
            )
            efficiency.temporal_rebin(time)

        # load radiance if provided
        if "radiance_key" in parameters.keys():
            radiance = self._get_data(
                parameters, wavelength, time, "radiance_key", signal.Radiance
            )
        # else we estimate it
        else:
            try:
                # if temperature is provided, we estimate the emission
                t_key = checks.find_key(
                    parameters.keys(), ["T", "temperature", "Temperature"]
                )
                T = parameters[t_key]
                T = checks.check_units(T, u.K)
                bb = BlackBody(T)
                bb_ = bb(wavelength).to(
                    u.W / u.m**2 / u.sr / u.um,
                    u.spectral_density(wavelength),
                )
            except KeyError:
                # if temperature is not provided, we consider zero emission
                bb_ = np.zeros(wavelength.size)

            radiance = signal.Radiance(spectral=wavelength, data=bb_)

            # load emissivity if provided
            if "emissivity_key" in parameters.keys():
                emissivity = self._get_data(
                    parameters,
                    wavelength,
                    time,
                    "emissivity_key",
                    signal.Dimensionless,
                )
                radiance *= emissivity

        self.debug("radiance: {}".format(radiance.data))

        return radiance, efficiency

    def _get_data(self, parameters, wl, tt, read_key, signal_type):
        # look for wavelength dependent data
        parsed_data = parameters["data"]
        self.debug("found emissivity file")

        wl_k = parameters["wavelength_key"]
        parsed_wl = checks.check_units(parsed_data[wl_k], "um")
        data_key = parameters[read_key]

        extracted_data = signal_type(
            spectral=parsed_wl, data=parsed_data[data_key].data
        )

        extracted_data.spectral_rebin(wl)
        extracted_data.temporal_rebin(tt)

        self.debug("emissivity: {}".format(extracted_data.data))
        return extracted_data
