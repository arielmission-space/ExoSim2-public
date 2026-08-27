import astropy.units as u

from exosim.tasks.task import Task

from .aperture_photometry import AperturePhotometry


class ComputeSignalsChannel(Task):
    """
    It estimates the radiometric signals on the input focal plane.

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
        focal_plane: :class:`~exosim.output.output.Output`
            focal plane HDF5 dataset
        parameters: dict, optional
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        oversampling: int, optional
            oversampling factor. If not provided, will extract from focal_plane metadata
        """
        self.add_task_param("table", "channel table")
        self.add_task_param("focal_plane", "focal_plane data")
        self.add_task_param("parameters", "channel parameters dict", None)
        self.add_task_param("oversampling", "oversampling factor", None)

    def execute(self):
        table = self.get_task_param("table")
        focal_plane = self.get_task_param("focal_plane")
        parameters = self.get_task_param("parameters")
        oversampling = self.get_task_param("oversampling")

        photometry = self.model(table, focal_plane, parameters, oversampling)
        if not isinstance(photometry, u.Quantity):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        if photometry.unit != u.ct / u.s:
            self.error(f"wrong output units: expected ct / s but got {photometry.unit}")
            raise u.UnitsError(
                f"wrong output units: expected ct / s but got {photometry.unit}"
            )

        self.set_output(photometry)

    def model(self, table, focal_plane, parameters, oversampling=None):
        """
        It estimates the radiometric signals on the input focal plane.
        It uses :func:`photutils.aperture.aperture_photometry` with the apertures
        from :class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures`.

        Parameters
        ----------
        table: :class:`astropy.table.QTable`
            apertures table
        focal_plane: :class:`~exosim.output.output.Output`
            focal plane HDF5 dataset
        parameters: dict, optional
            dictionary contained the channel parameters
        oversampling: int, optional
            oversampling factor. If None, will extract from focal_plane metadata

        Returns
        --------
        :class:`astropy.units.Quantity`
            photometry
        """
        self.debug("extracting focal plane from HDF5 dataset. Oversampling removed")

        # Get oversampling factor - use parameter if provided, otherwise get from metadata
        if oversampling is not None:
            osf = oversampling
            self.debug(f"Using provided oversampling factor: {osf}")
        else:
            osf = focal_plane["metadata"]["oversampling"][()]
            self.debug(f"Using oversampling factor from metadata: {osf}")

        focal_plane_data = focal_plane["data"][0, osf // 2 :: osf, osf // 2 :: osf]
        focal_plane_units = u.Unit(focal_plane["data_units"][()])
        focal_plane_with_units = focal_plane_data * focal_plane_units

        aperturePhotometry = AperturePhotometry()

        self.debug("aperture photometry for source focal plane")
        return aperturePhotometry(table=table, focal_plane=focal_plane_with_units)
