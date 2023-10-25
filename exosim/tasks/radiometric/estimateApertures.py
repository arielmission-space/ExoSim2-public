import numpy as np
from astropy.table import QTable
from scipy.interpolate import interp1d

import exosim.utils.aperture as aperture
from exosim.tasks.task import Task


class EstimateApertures(Task):
    """
    It returns the sizes of the apertures to perform photometry for :func:`photutils.aperture.aperture_photometry`.
    The details of the apertures depends on the configurations set by the user.

    Returns
    --------
    astropy.table.QTable:

    Raises
    --------
    TypeError:
        if the output is not :class:`~astropy.table.QTable`

    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        --------------
        table: :class:`astropy.table.QTable`
            wavelength table with bin edge
        focal_plane:
            focal plane
            Default is 'wl_solution`.
        wl_grid: (optional)
            focal plane wavelength solution
        description: dict (optional)
            channel description


        """
        self.add_task_param("table", "wavelength table with bin edges")
        self.add_task_param("focal_plane", "focal plane")
        # self.add_task_param('spatial_mode',
        #                     'aperture in spatial direction',
        #                     'column')
        # self.add_task_param('spectral_mode',
        #                     'aperture in spectral direction',
        #                     'wl_solution')
        self.add_task_param("wl_grid", "focal plane wavelength solution", None)
        self.add_task_param("description", "channel description", None)
        self.add_task_param("auto_mode", "automatic aperture mode", None)

    def execute(self):
        self.debug("performing aperture photometry")
        table = self.get_task_param("table")
        wl_grid = self.get_task_param("wl_grid")
        focal_plane = self.get_task_param("focal_plane")
        description = self.get_task_param("description")

        new_tab = self.model(table, focal_plane, description, wl_grid)

        if not isinstance(new_tab, QTable):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.set_output(new_tab)

    def model(self, table, focal_plane, description, wl_grid):
        """
        It returns the sizes of the apertures to perform photometry for :func:`photutils.aperture.aperture_photometry`.
        The default methods to estimate the aperture are the following and these are also the keywords required for the `description` dictionary.

        + spatial_mode: str
            aperture in spatial direction. If `column` the aperture spatial length is the full pixel column in the array.
            Default is `column`.
        + spectral_mode: str
            aperture in spectral direction. If `row` the aperture is spectral length is the full pixel row of the array.
            If `wl_solution` the spectral size is the same of the spectral bin in the input table.
            Default is 'wl_solution`.
        + auto_mode: str
            automatic aperture mode. The dictionary must contains two keywords.
            The `mode` keyword should be a str: if `rectangular` then :func:`~exosim.utils.psf.find_rectangular_aperture` is used;
            if `elliptical` then :func:`~exosim.utils.psf.find_elliptical_aperture` is used;
            if `bin` then :func:`~exosim.utils.psf.find_bin_aperture` is used;
            if `full` then the full array is integrated.
        + Ene: float
            The `EnE` keyword should be a float and represents the Encircled Energy to include in the aperture.


        Parameters
        --------------
        table: :class:`astropy.table.QTable`
            wavelength table with bin edge
        focal_plane:
            focal plane
        description: dict
            dictionary containing the aperture photometry description

        Returns
        --------
        astropy.table.QTable:
        """

        # spectral settings
        if "spectral_mode" in description.keys():
            if description["spectral_mode"] == "row":
                self.debug("spectral mode: row")
                wlc = [focal_plane.shape[1] / 2] * table["Wavelength"].size
                dwl = [focal_plane.shape[1]] * table["Wavelength"].size
                shape = "rectangular"

            elif description["spectral_mode"] == "wl_solution":
                self.debug("computing wavelength solution")

                if wl_grid is None:
                    self.error("focal plane wavelength solution missing")
                    raise OSError("focal plane wavelength solution missing")

                wl_sol = interp1d(
                    wl_grid,
                    np.arange(0, focal_plane.shape[1]),
                    fill_value="extrapolate",
                )
                wlc = wl_sol(table["Wavelength"])
                wld = wl_sol(table["left_bin_edge"])
                wlu = wl_sol(table["right_bin_edge"])
                dwl = abs(wlu - wld)
                shape = "rectangular"

            else:
                self.error("not supported spectral mode")
                raise OSError("not supported spectral mode")

        # spatial settings
        if "spatial_mode" in description.keys():
            if description["spatial_mode"] == "column":
                self.debug("spatial mode: column")
                spc = [focal_plane.shape[0] / 2] * table["Wavelength"].size
                dsp = [focal_plane.shape[0]] * table["Wavelength"].size
                shape = "rectangular"
            else:
                self.error("not supported spatial mode")
                raise OSError("not supported spatial mode")

        # automatic settings
        if "auto_mode" in description.keys():
            if description["auto_mode"] == "full":
                wlc = [focal_plane.shape[1] / 2] * table["Wavelength"].size
                dwl = [focal_plane.shape[1]] * table["Wavelength"].size
                spc = [focal_plane.shape[0] / 2] * table["Wavelength"].size
                dsp = [focal_plane.shape[0]] * table["Wavelength"].size
                shape = "rectangular"

            if description["auto_mode"] == "elliptical":
                ene = description["EnE"]
                sizes, surf, ene = aperture.find_elliptical_aperture(
                    focal_plane, ene
                )
                wlc = [focal_plane.shape[1] / 2] * table["Wavelength"].size
                dwl = [sizes[0]] * table["Wavelength"].size
                spc = [focal_plane.shape[0] / 2] * table["Wavelength"].size
                dsp = [sizes[1]] * table["Wavelength"].size
                shape = "elliptical"

            if description["auto_mode"] == "rectangular":
                ene = description["EnE"]
                sizes, surf, ene = aperture.find_rectangular_aperture(
                    focal_plane, ene
                )
                wlc = [focal_plane.shape[1] / 2] * table["Wavelength"].size
                dwl = [sizes[1]] * table["Wavelength"].size
                spc = [focal_plane.shape[0] / 2] * table["Wavelength"].size
                dsp = [sizes[0]] * table["Wavelength"].size
                shape = "rectangular"

            if description["auto_mode"] == "bin":
                ene = description["EnE"]

                if wl_grid is None:
                    self.error("focal plane wavelength solution missing")
                    raise OSError("focal plane wavelength solution missing")

                wl_sol = interp1d(
                    wl_grid,
                    np.arange(0, focal_plane.shape[1]),
                    fill_value="extrapolate",
                )
                wlc = wl_sol(table["Wavelength"])
                wld = wl_sol(table["left_bin_edge"])
                wlu = wl_sol(table["right_bin_edge"])
                dwl = abs(wlu - wld)
                spc = [focal_plane.shape[0] / 2] * table["Wavelength"].size
                dsp = [1] * table["Wavelength"].size
                shape = "rectangular"

                self.debug("estimating apertures for spectral bins")
                for i in range(wlc.size):
                    size, surf, ene = aperture.find_bin_aperture(
                        focal_plane,
                        ene,
                        spatial_with=dwl[i],
                        center=(wlc[i], spc[i]),
                    )
                    dsp[i] *= size

        new_tab = QTable()
        new_tab["spectral_center"] = wlc
        new_tab["spectral_size"] = dwl
        new_tab["spatial_center"] = spc
        new_tab["spatial_size"] = dsp
        new_tab["aperture_shape"] = shape

        return new_tab
