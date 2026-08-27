import astropy.constants as const
import astropy.units as u

import exosim.models.signal as signal
from exosim.tasks.task import Task


class LoadResponsivity(Task):
    """
    Returns the channel responsivity (in counts/Joule) from the quantum efficiency (QE)

    This default implementation loads the quantum efficiency from the input parameters,
    then converts it to responsivity using the relation:

    .. math::
        R(\\lambda) = \\frac{QE(\\lambda) \\cdot \\lambda}{h \\cdot c}

    where:
    - :math:`R(\\lambda)` is the responsivity as a function of wavelength
    - :math:`QE(\\lambda)` is the quantum efficiency as a function of wavelength
    - :math:`\\lambda` is the wavelength
    - :math:`h` is Planck's constant (:math:`6.626 \\times 10^{-34}` J·s)
    - :math:`c` is the speed of light (:math:`2.998 \\times 10^8` m/s)

    Returns
    --------
    :class:`~exosim.models.signal.Signal`
        channel responsivity

    Raises
    ------
    TypeError:
        if the output is not a :class:`~exosim.models.signal.Signal` class

    UnitConversionError
        if the output has not the right units (counts/Joule)

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
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid
        """

        self.add_task_param("parameters", "parameters dict")
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("time", "time grid")

    def execute(self):
        parameters = self.get_task_param("parameters")
        wl = self.get_task_param("wavelength")
        tt = self.get_task_param("time")

        responsivity = self.model(parameters, wl, tt)

        # checking output
        if not isinstance(responsivity, signal.Signal):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        try:
            responsivity.data_units.to(u.m / const.c / const.h * u.count)
        except u.UnitConversionError:
            self.error(
                f"{responsivity.data_units} are not convertible in {(u.m / const.c / const.h * u.count).unit}"
            )
            raise u.UnitConversionError(
                f"{responsivity.data_units} are not convertible in {(u.m / const.c / const.h * u.count).unit}"
            ) from None

        self.debug(f"responsivity: {responsivity.data}")

        self.set_output(responsivity)

    def model(self, parameters, wavelength, time):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            channel responsivity

        """
        qe_data = parameters["qe"]["data"]
        wl_ = qe_data["Wavelength"]
        qe_ = qe_data[parameters["value"]]
        qe = signal.Dimensionless(data=qe_, spectral=wl_)
        qe.spectral_rebin(wavelength)
        qe.temporal_rebin(time)

        return signal.Signal(
            spectral=wavelength,
            time=time,
            data=qe.data * wavelength.to(u.m) / const.c / const.h * u.count,
        )
