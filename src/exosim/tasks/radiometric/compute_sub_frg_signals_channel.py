import astropy.units as u
from astropy.table import QTable

from exosim.tasks.task import Task
from exosim.utils.checks import check_units

from .aperture_photometry import AperturePhotometry


class ComputeSubFrgSignalsChannel(Task):
    """
    It iteratively estimates the radiometric signals on the foregrounds sub focal planes for a channel
    and returns a table with all the contributions.

    Returns
    --------
    :class:`astropy.table.QTable`
        signal table

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
        ----------
        table: :class:`astropy.table.QTable`
            apertures table
        ch_name: str
            channel name
        input_file: :class:`~exosim.output.output.Output`
            input HDF5 file
        channels_path: str
            channels path in the input file
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        """
        self.add_task_param("table", "channel table")
        self.add_task_param("ch_name", "channel name")
        self.add_task_param("input_file", "input file containing the focal planes")
        self.add_task_param("channels_path", "focal plane path", None)
        self.add_task_param("parameters", "channel parameters dict", None)

    def execute(self):
        table = self.get_task_param("table")
        ch = self.get_task_param("ch_name")
        input_file = self.get_task_param("input_file")
        channels_path = self.get_task_param("channels_path")
        parameters = self.get_task_param("parameters")

        self.debug(f"Computing foreground signals table for {ch}")
        new_table = self.model(ch, table, input_file, channels_path, parameters)

        if not isinstance(new_table, QTable):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.set_output(new_table)

    def model(self, ch, table, input_file, channels_path, parameters):
        """
        It iteratively estimates the radiometric signals on the foregrounds sub focal plane for a channel
        and returns a table with all the contributions.
        It uses :func:`photutils.aperture.aperture_photometry` with the apertures
        from :class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures`.

        Parameters
        ----------
        table: :class:`astropy.table.QTable`
            apertures table
        ch_name: str
            channel name
        input_file: :class:`~exosim.output.output.Output`
            input HDF5 file
        channels_path: str
            channels path in the input file
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`

        Returns
        --------
        :class:`astropy.table.QTable`
            signal table
        """

        new_table = QTable()
        with input_file.open() as f:
            if channels_path is not None:
                f = f[channels_path]
            if "sub_focal_planes" in f["channels"][ch]:
                for frg in f["channels"][ch]["sub_focal_planes"]:
                    # prepare the focal plane
                    self.debug(f"extracting {frg} focal plane. Oversampling removed")
                    sub_f = f["channels"][ch]["sub_focal_planes"][frg]
                    osf = sub_f["metadata"]["oversampling"][()]
                    focal_plane_units = u.Unit(sub_f["data_units"][()])
                    focal_plane = sub_f["data"][0, osf // 2 :: osf, osf // 2 :: osf]
                    focal_plane = check_units(
                        focal_plane, focal_plane_units, force=True
                    )

                    aperturePhotometry = AperturePhotometry()

                    self.debug(f"aperture photometry for {frg} focal plane")
                    signal_in_phot = aperturePhotometry(
                        table=table, focal_plane=focal_plane
                    )

                    name = frg.split("_")[-1]
                    new_table[f"{name}_signal_in_aperture"] = check_units(
                        signal_in_phot, focal_plane_units, force=True
                    )

                    if (
                        "type" in parameters
                        and parameters["type"].lower() == "photometer"
                    ):
                        # for photometer we also compute the total signal on the focal plane
                        total_signal = focal_plane.sum()
                        new_table[f"{name}_total_signal"] = check_units(
                            total_signal, focal_plane_units, force=True
                        )
                    if (
                        "type" in parameters
                        and parameters["type"].lower() == "spectrometer"
                    ):
                        # for spectrometer we also compute the total signal on bin columns
                        _table_extended = table.copy()
                        _table_extended["spatial_size"] = focal_plane.shape[1]
                        signal_in_column = aperturePhotometry(
                            table=_table_extended, focal_plane=focal_plane
                        )
                        new_table[f"{name}_total_signal"] = check_units(
                            signal_in_column, focal_plane_units, force=True
                        )
        return new_table
