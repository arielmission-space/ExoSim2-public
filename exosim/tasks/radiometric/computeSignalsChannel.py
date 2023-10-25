import astropy.units as u

from .aperturePhotometry import AperturePhotometry
from exosim.tasks.task import Task


class ComputeSignalsChannel(Task):
    """
    It estimates the radiometric signals on the input focal plane..

    Returns
    --------
    :class:`astropy.units.Quantity`
        photometry

    Raises
    --------
    TypeError:
        if the output is not :class:`~astropy.units.Quantity`
    UnitsError:
        wrong output units

    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.

    """

    def __init__(self):
        """
        Parameters
        ----------
        table: :class:`astropy.table.QTable`
            apertures table
        focal plane: :class:`~exosim.output.output.Output`
            focal plane in the HDF5 file
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """
        self.add_task_param("table", "channel table")
        self.add_task_param("focal_plane", "focal_plane")
        self.add_task_param("parameters", "channel parameters dict", None)

    def execute(self):
        table = self.get_task_param("table")
        focal_plane = self.get_task_param("focal_plane")
        parameters = self.get_task_param("parameters")

        photometry = self.model(table, focal_plane, parameters)
        if not isinstance(photometry, u.Quantity):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        if not photometry.unit == u.ct / u.s:
            self.error("wrong output units")
            raise u.UnitsError("wrong output units")

        self.set_output(photometry)

    def model(self, table, focal_plane, parameters):
        """
        It estimates the radiometric signals on the input focal plane..
        It uses :func:`photutils.aperture.aperture_photometry` with the apertures
        from :class:`~exosim.tasks.radiometric.estimateApertures.EstimateApertures`.

        Parameters
        ----------
        table: :class:`astropy.table.QTable`
            apertures table
        focal plane: :class:`~exosim.output.output.Output`
            focal plane in the HDF5 file
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        :class:`astropy.units.Quantity`
            photometry

        """
        # prepare the focal plane
        self.debug("extracting focal plane. Oversampling removed")
        osf = focal_plane["metadata"]["oversampling"][()]
        focal_plane_units = u.Unit(focal_plane["data_units"][()])
        focal_plane = (
            focal_plane["data"][0, osf // 2 :: osf, osf // 2 :: osf]
            * focal_plane_units
        )

        aperturePhotometry = AperturePhotometry()

        self.debug("aperture photometry for source focal plane")
        signal_in_phot = aperturePhotometry(
            table=table, focal_plane=focal_plane
        )

        return signal_in_phot * focal_plane_units
