import warnings

from exosim.tasks.sed import CreatePlanckStar
from exosim.tasks.sed import LoadCustom
from exosim.tasks.sed import LoadPhoenix
from exosim.tasks.task import Task

warnings.filterwarnings("ignore", category=UserWarning, append=True)


class PrepareSed(Task):
    """
    Returns a source SED.
    The SED depends on the source type selected. This Task analyse the inputs and return the desired SED.

    Returns
    -------
    :class:`~exosim.models.signal.Sed`
        Star Sed

    Examples
    ---------
    We first define all the inputs and then we show how to produce the star sed:

    >>> from exosim.tasks.sed import LoadPhoenix
    >>> wavelength = np.linspace(0.5, 7.8, 10000) * u.um
    >>> T = 5778 * u.K
    >>> R = 1 * u.R_sun
    >>> D = 1 * u.au
    >>> M = 1 * u.Msun
    >>> z = 0.0
    >>> g = (cc.G * M.si / R.si ** 2).to(u.cm / u.s ** 2)
    >>> logg = np.log10(g.value)

    To produce a planck star:

    >>> prepareSed = LoadPhoenix()
    >>> sed = PrepareSed(source_type='planck', wavelength=wavelength, T=T, R=R, D=D)

    To load a Phoenix star:
    >>> sed_l = prepareSed(source_type='phoenix', path=phoenix_stellar_model, T=T, R=R, D=D, logg=logg, z=z)

    """

    def __init__(self):
        """
        Parameters
        __________
        source_type: str
            source type, can be `planck`, `phoenix` or `custom`. Default is `planck`
        wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity` (optional)
            wavelength grid. If no units are attached is considered as expressed in `um`
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
        self.add_task_param("source_type", "source type", "planck")
        self.add_task_param("wavelength", "wavelength grid", None)
        self.add_task_param("R", "star radius")
        self.add_task_param("D", "star distance")
        self.add_task_param("T", "star temperature", None)
        self.add_task_param("logg", "star logG", None)
        self.add_task_param("z", "star metallicity", None)
        self.add_task_param("path", "phoenix spectra path", None)
        self.add_task_param("filename", "phoenix file name", None)

    def execute(self):
        self.info("loading source Sed")

        source_type = self.get_task_param("source_type")
        source_type = source_type.lower()
        if source_type == "planck":
            wl = self.get_task_param("wavelength")
            T = self.get_task_param("T")
            R = self.get_task_param("R")
            D = self.get_task_param("D")

            createPlanckStar = CreatePlanckStar()
            sed = createPlanckStar(wavelength=wl, T=T, R=R, D=D)

        elif source_type == "phoenix":
            path = self.get_task_param("path")
            filename = self.get_task_param("filename")
            R = self.get_task_param("R")
            D = self.get_task_param("D")
            T = self.get_task_param("T")
            logg = self.get_task_param("logg")
            z = self.get_task_param("z")

            loadPhoenix = LoadPhoenix()
            sed = loadPhoenix(
                filename=filename, path=path, T=T, R=R, D=D, logg=logg, z=z
            )

        elif source_type == "custom":
            filename = self.get_task_param("filename")
            R = self.get_task_param("R")
            D = self.get_task_param("D")

            loadCustom = LoadCustom()
            sed = loadCustom(filename=filename, R=R, D=D)

        else:
            self.error("not supported source type")
            raise KeyError("not supported source type")

        self.set_output(sed)
