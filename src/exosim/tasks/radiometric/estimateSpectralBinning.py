import astropy.units as u
import numpy as np
from astropy.table import QTable
from scipy.interpolate import interp1d

import exosim.utils.grids as grids
from exosim.tasks.task import Task


class EstimateSpectralBinning(Task):
    """
    It computes spectral binning useful to produce the radiometric tables

    Returns
    --------
    astropy.table.QTable:
        wavelength bins table

    Raises
    --------
    TypeError:
        if the output is not :class:`~astropy.table.QTable`
    KeyError:
        if a column is missing in the output :class:`~astropy.table.QTable`

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
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        wl_grid: :class:`~astropy.units.Quantity` (optional)
            focal plane wavelength grid. Useful for spectrometers. Default is ``None``.
        """
        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("wl_grid", "focal plane spectral grid", None)

    def execute(self):
        self.debug("spectral binning estimation started")
        parameters = self.get_task_param("parameters")
        wl_grid = self.get_task_param("wl_grid")

        table = self.model(wl_grid, parameters)

        if not isinstance(table, QTable):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        key_list = [
            "ch_name",
            "Wavelength",
            "bandwidth",
            "left_bin_edge",
            "right_bin_edge",
        ]
        for key in key_list:
            if key not in table.keys():
                self.error("missing keyword: {}".format(key))
                raise KeyError("missing keyword: {}".format(key))

        self.set_output(table)

    def model(self, wl_grid, parameters):
        """
        It runs a dedicated method if the channel is a spectrometer or a photometer.

        Parameters
        ----------
        wl_grid: :class:`~astropy.units.Quantity` (optional)
            focal plane wavelength grid. Useful for spectrometers. Default is ``None``.
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        astropy.table.QTable:
            wavelength bins table

        Raises
        --------
        AttributeError:
            if the channel type is not `spectrometer` or `photometer`
        """
        if parameters["type"].lower() == "spectrometer":
            table = self.wavelength_table_spectrometer(parameters, wl_grid)

        elif parameters["type"].lower() == "photometer":
            table = self.wavelength_table_photometer(parameters)

        else:
            self.error(
                "unsupported channel type: {}", format(parameters["type"])
            )
            raise AttributeError(
                "unsupported channel type: {}", format(parameters["type"])
            )
        return table

    def wavelength_table_spectrometer(self, description, wl_grid):
        """
        It returns the wavelength table for a spectrometer.
        The wavelength grid can be estimated in 2 modes:

        - `native` mode. If `targetR` is set to `native` the wavelength grid computed is the pixel level wavelength grid, where each bin is of the size of a pixel;
        - `fixed R` mode. If targetR` is set to a constant value, the wavelength grid is estimated using :func:`~exosim.utils.grids.wl_grid`.

        Parameters
        ----------
        description: dict
            dictionary contained the channel parameters.
        wl_grid: :class:`~astropy.units.Quantity`
            wavelength grid.

        Returns
        -------
        astropy.table.QTable:
            wavelength bins table

        Raises
        ---------
        KeyError
            Channel targetR format unsupported
        """
        table = QTable()
        # check if R is defined
        if "targetR" not in description:
            self.warning("Channel targetR missing: native R is assumed")
            description["targetR"] = "native"

        # native R
        if description["targetR"] == "native":
            wl_bin_c = wl_grid
            wl_sol = interp1d(
                wl_grid, np.arange(0, wl_grid.size), fill_value="extrapolate"
            )
            wl_bin_width = (
                wl_sol(np.arange(0, wl_grid.size) + 0.5)
                - wl_sol(np.arange(0, wl_grid.size) - 0.5)
            ) * u.um

        # fixed R
        elif isinstance(description["targetR"], int):
            wl_bin_c, wl_bin_width = grids.wl_grid(
                description["wl_min"],
                description["wl_max"],
                description["targetR"],
                return_bin_width=True,
            )
        else:
            self.error("Channel targetR format unsupported.")
            raise KeyError("Channel targetR format unsupported.")

        # preparing the table
        table["ch_name"] = [description["value"]] * wl_bin_c.size
        table["Wavelength"] = wl_bin_c
        table["bandwidth"] = wl_bin_width
        table["left_bin_edge"] = table["Wavelength"] - 0.5 * table["bandwidth"]
        table["right_bin_edge"] = (
            table["Wavelength"] + 0.5 * table["bandwidth"]
        )
        self.debug("wavelength table: \n{}".format(table))
        return table

    def wavelength_table_photometer(self, description):
        """
        It returns the wavelength table for a photometer.
        It is estimated as the central wavelength of the photometer
        with a bin width equal to the wavelength band.

        Parameters
        ----------
        description: dict
            dictionary contained the channel parameters.

        Returns
        -------
        astropy.table.QTable:
            wavelength bins table
        """
        table = QTable()

        wl_c = 0.5 * (description["wl_min"] + description["wl_max"])
        bandwidth = description["wl_max"] - description["wl_min"]
        left_bin_edge = wl_c - 0.5 * bandwidth
        right_bin_edge = wl_c + 0.5 * bandwidth
        table["ch_name"] = [description["value"]]
        table["Wavelength"] = [wl_c.value] * wl_c.unit
        table["bandwidth"] = [bandwidth.value] * bandwidth.unit
        table["left_bin_edge"] = [left_bin_edge.value] * left_bin_edge.unit
        table["right_bin_edge"] = [right_bin_edge.value] * right_bin_edge.unit
        self.debug("wavelength table: \n{}".format(table))

        return table
