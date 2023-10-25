from astropy import units as u
from astropy.io import ascii

import exosim.models.signal as signal
import exosim.utils.checks as checks
from exosim.tasks.task import Task


class LoadCustom(Task):
    r"""
    Loads a custom SED from a file and scaled it by the solid angle :math:`\pi \left( \frac{R}{D} \right)^2`.

    Returns
    -------
    :class:`~exosim.models.signal.Sed`
        Star Sed
    """

    def __init__(self):
        """
        Parameters
        -----------
        R: :class:`~astropy.units.Quantity` or float
            star radius. If no units are attached is considered as expressed in `m`
        D: :class:`~astropy.units.Quantity` or float
            star distance. If no units are attached is considered as expressed in `m`
        filename: str
            custom sed file path
        """
        self.add_task_param("filename", "custom sed filename")
        self.add_task_param("R", "star radius")
        self.add_task_param("D", "star distannce")

    def execute(self):
        self.info("loading custom sed")
        fname = self.get_task_param("filename")
        R = self.get_task_param("R")
        D = self.get_task_param("D")

        ph = ascii.read(fname, format="ecsv")
        wl_k = checks.find_key(
            ph.keys(), ["Wavelength", "wavelength", "wl"], self
        )
        ph_wl = ph[wl_k].data * ph[wl_k].unit
        sed_k = checks.find_key(ph.keys(), ["Sed", "sed"], self)
        ph_sed = ph[sed_k].data * ph[sed_k].unit

        sed = signal.Sed(spectral=ph_wl, data=ph_sed)

        R = checks.check_units(R, u.m, self)
        D = checks.check_units(D, u.m, self)

        sed *= (R / D) ** 2
        self.debug("custom sed used : {}".format(fname))

        self.set_output(sed)
