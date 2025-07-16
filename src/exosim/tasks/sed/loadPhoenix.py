import glob
import os

import numpy as np
from astropy import units as u
from astropy.io import fits

import exosim.models.signal as signal
import exosim.utils.checks as checks
from exosim.tasks.task import Task


class LoadPhoenix(Task):
    """
    Loads a star SED from a grid of Phoenix spectra or from a specific Phoenix file.

    Returns
    -------
    :class:`~exosim.models.signal.Sed`
        Star Sed

    Examples
    --------
    >>> from exosim.tasks.sed import LoadPhoenix
    >>> loadPhoenix = LoadPhoenix()

    Prepare the star

    >>> from astropy import constants as cc
    >>> import astropy.units as u
    >>> import numpy as np
    >>> D= 12.975 * u.pc
    >>> T= 3016 * u.K
    >>> M= 0.15 * u.Msun
    >>> R= 0.218 * u.Rsun
    >>> z= 0.0
    >>> g = (cc.G * M.si / R.si ** 2).to(u.cm / u.s ** 2)
    >>> logg = np.log10(g.value)

    Load the sed from a directory

    >>> sed = loadPhoenix(path = phoenix_directory,  T=T, D=D, R=R, z=z, logg=logg)

    or load the sed from a file

    >>> sed = loadPhoenix(filename = phoenix_file, D=D, R=R)

    Notes
    --------
    This class can either load the SED from a file or select the most suitable file from the Phoenix spectra path,
    given the proper information on the star. In the former case the `filename` keyword is needed, in the latter are
    required the keywords `path`, `T`, `z`, `logg`. Not providing the right keywords would result in an error.

    Raises
    --------
        InputError
            if neither `filename` or `path` are given.
        KeyError
            if a needed parameter is missing in the user inputs.

    """

    def __init__(self):
        """
        Parameters
        -----------
        R: :class:`~astropy.units.Quantity` or float
            star radius. If no units are attached is considered as expressed in `m`
        D: :class:`~astropy.units.Quantity` or float
            star distance. If no units are attached is considered as expressed in `m`
        filename: str (optional)
            Phoenix file path
        path: str (optional)
            Phoenix directory path
        T: :class:`~astropy.units.Quantity` or float (optional)
            star temperature. If no units are attached is considered as expressed in `K`
        z: float (optional)
            star metallicity.
        logg: float (optional)
            star logG.

        """
        self.add_task_param("R", "star radius")
        self.add_task_param("D", "star distannce")
        self.add_task_param("T", "star temperature", None)
        self.add_task_param("logg", "star logG", None)
        self.add_task_param("z", "star metallicity", None)
        self.add_task_param("path", "phoenix spectra path", None)
        self.add_task_param("filename", "phoenix file name", None)

    def execute(self):
        path = self.get_task_param("path")
        filename = self.get_task_param("filename")
        R = self.get_task_param("R")
        D = self.get_task_param("D")
        T = self.get_task_param("T")
        logg = self.get_task_param("logg")
        z = self.get_task_param("z")

        if not filename and not path:
            # TODO add Phoenix system path option if not path
            path = os.environ.get("PHOENIX_PATH", None)

        if (
            filename
        ):  # if filename is given it ignores anything else and it uses that
            fits_file_name = filename
        else:
            if path:
                if not os.path.exists(path):
                    # TODO test this
                    self.error(
                        "to load a phoenix model indicate a model file name or the phonix path"
                    )
                    raise OSError(
                        "to load a phoenix model indicate a model file name or the phonix path"
                    )
                if z is None:
                    z = 0.0
                    self.debug("star metallicity missing. zero is assumed.")
                if T is None:
                    self.error("star temperature missing")
                    raise KeyError("star temperature missing")
                if hasattr(T, "unit"):
                    T = T.to(u.K)
                else:
                    T = T * u.K
                    self.debug("no units found for T: Kelvin are assumed.")
                if not logg:
                    self.error("star logg missing")
                    raise KeyError("star logg missing")
                fits_file_name = self.get_phoenix_model_filename(
                    path, T, logg, z
                )
            else:
                self.error("phoenix path missing")
                raise OSError("phoenix path missing")
        sed = self.load(fits_file_name)
        sed.metadata["phoenix_file"] = fits_file_name
        if R is None:
            self.error("star radius missing")
            raise KeyError("star radius missing")
        if D is None:
            self.error("star distance missing")
            raise KeyError("star distance missing")

        R = checks.check_units(R, u.m, self)
        D = checks.check_units(D, u.m, self)

        sed *= (R / D) ** 2

        self.debug("phoenix sed loaded: {}".format(sed.data))

        self.set_output(sed)

    def load(self, ph_file):
        """
        Returs a sed given a Phoenix filename
        Parameters
        ----------
        ph_file: str
            filename

        Returns
        ---------
        :class:`~exosim.models.signal.Sed`
            Star Sed
        """
        with fits.open(ph_file) as hdu:
            str_unit = hdu[1].header["TUNIT1"]
            wl_ph = hdu[1].data.field("Wavelength") * u.Unit(str_unit)

            str_unit = hdu[1].header["TUNIT2"]
            sed_ph = hdu[1].data.field("Flux") * u.Unit(str_unit)

            idx = np.nonzero(np.diff(wl_ph))
            wl_ph = wl_ph[idx]
            sed_ph = sed_ph[idx]

        sed = signal.Sed(spectral=wl_ph, data=sed_ph)
        return sed

    def get_phoenix_model_filename(self, path, T, logg, z):
        """
        It returns the name of the phoenix file that best matches the input star parameters.

        Parameters
        ----------
        path: str
            Phoenix directory path
        T: :class:`~astropy.units.Quantity`
            star temperature
        logg: float
            star logG
        z: float
            star metallicity

        Returns
        -------
        str
            Phoenix file name
        """
        sed_name = glob.glob(os.path.join(path, "*.BT-Settl.spec.fits.gz"))

        if len(sed_name) == 0:
            self.error("No stellar SED files found")
            raise OSError("No stellar SED files found")

        sed_T_list = np.array(
            [float(os.path.basename(k)[3:8]) for k in sed_name]
        )
        sed_Logg_list = np.array(
            [float(os.path.basename(k)[9:12]) for k in sed_name]
        )
        sed_Z_list = np.array(
            [float(os.path.basename(k)[13:16]) for k in sed_name]
        )

        idx = np.argmin(
            np.abs(sed_T_list - np.round(T.value / 100.0))
            + np.abs(sed_Logg_list - logg)
            + np.abs(sed_Z_list - z)
        )

        ph_file = sed_name[idx]

        self.debug("phoenix file selected: {}".format(ph_file))
        return ph_file
