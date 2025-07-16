import numpy as np
from astropy import units as u
from astropy.modeling.physical_models import BlackBody

import exosim.models.signal as signal
import exosim.utils.checks as checks
from exosim.tasks.task import Task


class CreatePlanckStar(Task):
    r"""
    Create a star SED using the Planck function.

    The star emission is simulated by :class:`astropy.modeling.physical_models.BlackBody`.
    The resulting sed is then converted into :math:`W/m^2/sr/\mu m` and scaled by the solid angle :math:`\pi \left( \frac{R}{D} \right)^2`.


    Returns
    -------
    :class:`~exosim.models.signal.Sed`
        Star Sed

    Examples
    --------
    >>> from exosim.tasks.sed import CreatePlanckStar
    >>> import astropy.units as u
    >>> import numpy as np
    >>> createPlanckStar = CreatePlanckStar()
    >>> wl = np.linspace(0.5, 7.8, 10000) * u.um
    >>> T = 6086 * u.K
    >>> R = 1.18 * u.R_sun
    >>> D = 47 * u.au
    >>> sed = createPlanckStar(wavelength=wl, T=T, R=R, D=D)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sed.spectral, sed.data[0,0])
    >>> plt.ylabel(sed.data_units)
    >>> plt.xlabel(sed.spectral_units)
    >>> plt.show()

    .. plot:: mpl_examples/createPlanckStar.py
    """

    def __init__(self):
        """
        Parameters
        __________
        wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            wavelength grid. If no units are attached is considered as expressed in `um`
        T: :class:`~astropy.units.Quantity` or float
            star temperature. If no units are attached is considered as expressed in `K`
        R: :class:`~astropy.units.Quantity` or float
            star radius. If no units are attached is considered as expressed in `m`
        D: :class:`~astropy.units.Quantity` or float
            star distance. If no units are attached is considered as expressed in `m`
        """
        self.add_task_param("wavelength", "wavelength grid")
        self.add_task_param("T", "star temperature")
        self.add_task_param("R", "star radius")
        self.add_task_param("D", "star distannce")

    def execute(self):
        wl = self.get_task_param("wavelength")
        T = self.get_task_param("T")
        R = self.get_task_param("R")
        D = self.get_task_param("D")

        wl = checks.check_units(wl, u.um, self)
        T = checks.check_units(T, u.K, self)
        R = checks.check_units(R, u.m, self)
        D = checks.check_units(D, u.m, self)

        omega_star = np.pi * (R / D) ** 2 * u.sr
        bb = BlackBody(T)
        bb_ = bb(wl).to(u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl))
        sed = signal.Sed(spectral=wl, data=omega_star * bb_)

        self.debug("star plack source created: {}".format(sed.data))

        self.set_output(sed)
