import astropy.units as u
from astropy.table import QTable

from .aperturePhotometry import AperturePhotometry
from exosim.tasks.task import Task


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
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """
        self.add_task_param("table", "channel table")
        self.add_task_param("ch_name", "channel name")
        self.add_task_param(
            "input_file", "input file containing the focal planes"
        )
        self.add_task_param("parameters", "channel parameters dict", None)

    def execute(self):
        table = self.get_task_param("table")
        ch = self.get_task_param("ch_name")
        input_file = self.get_task_param("input_file")
        parameters = self.get_task_param("parameters")

        self.debug("Computing foreground signals table for {}".format(ch))
        new_table = self.model(ch, table, input_file, parameters)

        if not isinstance(new_table, QTable):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.set_output(new_table)

    def model(self, ch, table, input_file, parameters):
        """
        It iteratively estimates the radiometric signals on the foregrounds sub focal plane for a channel
        and returns a table with all the contributions.
        It uses :func:`photutils.aperture.aperture_photometry` with the apertures
        from :class:`~exosim.tasks.radiometric.estimateApertures.EstimateApertures`.

        Parameters
        ----------
        table: :class:`astropy.table.QTable`
            apertures table
        ch_name: str
            channel name
        input_file: :class:`~exosim.output.output.Output`
            input HDF5 file
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        :class:`astropy.table.QTable`
            signal table
        """

        new_table = QTable()
        with input_file.open() as f:
            if "sub_focal_planes" in f["channels"][ch].keys():
                for frg in f["channels"][ch]["sub_focal_planes"].keys():
                    # prepare the focal plane
                    self.debug(
                        "extracting {} focal plane. Oversampling removed".format(
                            frg
                        )
                    )
                    sub_f = f["channels"][ch]["sub_focal_planes"][frg]
                    osf = sub_f["metadata"]["oversampling"][()]
                    focal_plane_units = u.Unit(sub_f["data_units"][()])
                    focal_plane = (
                        sub_f["data"][0, osf // 2 :: osf, osf // 2 :: osf]
                        * focal_plane_units
                    )

                    aperturePhotometry = AperturePhotometry()

                    self.debug(
                        "aperture photometry for {} focal plane".format(frg)
                    )
                    signal_in_phot = aperturePhotometry(
                        table=table, focal_plane=focal_plane
                    )

                    name = frg.split("_")[-1]
                    new_table["{}_signal_in_aperture".format(name)] = (
                        signal_in_phot * focal_plane_units
                    )
        return new_table
