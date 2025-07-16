import astropy.units as u
from astropy.table import QTable

from exosim.tasks.task import Task


class LoadWavelengthSolution(Task):
    """
    It loads the wavelength solution expressed as a table with wavelength position on spectral and spatial directcions

    Returns
    --------
    :class:`~astropy.table.Qtable`
        wavelength solution table

    Raises
    -------
    TypeError:
        if the output is not a :class:`~astropy.table.QTable` class

    UnitsError
        if the output has not the right units (micron) in every column

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
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """

        self.add_task_param("parameters", "channel parameters dict")

    def execute(self):
        parameters = self.get_task_param("parameters")

        wl_solution = self.model(parameters)

        if not isinstance(wl_solution, QTable):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        for k in ["wavelength", "spectral", "spatial"]:
            if k not in wl_solution.keys():
                self.error("missing {} keyword".format(k))
                raise KeyError("missing {} keyword".format(k))
            elif wl_solution[k].unit != u.um:
                self.error("wrong units: {} units are not micron".format(k))
                raise u.UnitsError(
                    "wrong units: {} units are not micron".format(k)
                )

        self.set_output(wl_solution)

    def model(self, parameters):
        """
        It loads the Wavelength, X and Y columns from a file and it stores them into a class:`~astropy.table.Qtable` class

        Parameters
        ----------
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`


        Returns
        --------
        :class:`~astropy.table.Qtable`
            wavelength solution table

        """
        wl_solution = QTable()
        wl_solution["wavelength"] = parameters["wl_solution"]["data"][
            "Wavelength"
        ]
        wl_solution["spectral"] = parameters["wl_solution"]["data"]["x"]
        wl_solution["spatial"] = parameters["wl_solution"]["data"]["y"]
        return wl_solution
