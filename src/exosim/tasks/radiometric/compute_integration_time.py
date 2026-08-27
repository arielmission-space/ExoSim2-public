import numpy as np
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeIntegrationTime(Task):
    r"""
    Task to compute the integration time for each entry in a saturation table.

    This class determines the integration time as the minimum integration time found in the input saturation table.
    The computed integration time can be used for all entries or for a specific channel if requested.

    Parameters
    ----------
    saturation_table : astropy.table.QTable or array-like
        Table containing the saturation information. Must include the column:
        - 'integration_time': integration time for each entry (astropy.units.Quantity, in s)
    description : dict
        Dictionary containing the channel description. Must include:
        - 'integration_time': integration time (required key, astropy.units.Quantity, in s)
    channel_name : str, optional
        Name of the channel to filter the table. If None, all entries are used.

    Raises
    ------
    ValueError
        If required keys are missing in the description dictionary or table.
    TypeError
        If the output table is not a QTable.

    Notes
    -----
    The integration time is computed as:

    .. math::

        \mathrm{frame\_time} = \min(\mathrm{integration\_time})

    and is assigned to all relevant entries in the output.
    """

    def __init__(self):
        self.add_task_param(
            "saturation_table", "table containing saturation information"
        )
        self.add_task_param("description", "channel description", None)
        self.add_task_param("channel_name", "name of the channel", None)

    def execute(self):
        self.debug("computing integration time")
        table = self.get_task_param("saturation_table")
        description = self.get_task_param("description")
        channel_name = self.get_task_param("channel_name")

        model = self.model(table, description, channel_name)

        self.set_output(model)

    def model(self, table, description, channel_name=None):
        r"""
        Compute the integration time for each entry in the provided saturation table.

        The integration time is determined as the minimum saturation time found in the input table.
        If a channel name is provided, the minimum is computed only for that channel.
        The resulting integration time is returned as an array with the same length as the number of bins for the selected channel or the whole table.

        Parameters
        ----------
        table : astropy.table.QTable or array-like
            Table containing the saturation information. Must include the column:
            - 'saturation_time': saturation time for each entry (astropy.units.Quantity, in s)
        description : dict
            Dictionary containing the channel description.
        channel_name : str, optional
            Name of the channel to filter the table. If None, all entries are used.

        Returns
        -------
        numpy.ndarray
            Array of integration time values (minimum saturation time) for each entry (or channel bin), in seconds.

        Raises
        ------
        ValueError
            If required keys are missing in the description dictionary or table.

        Notes
        -----
        The integration time is computed as:

        .. math::

            \mathrm{integration\_time} = \min(\mathrm{saturation\_time})

        and is assigned to all relevant entries in the output.
        """

        table = QTable(table)

        if "saturation_time" not in table.colnames:
            raise ValueError("Saturation time is missing in the input.")

        if channel_name:
            saturation_time = table["saturation_time"][table["ch_name"] == channel_name]
            n_bins = len(table[table["ch_name"] == channel_name])
        else:
            saturation_time = table["saturation_time"]
            n_bins = len(table)

        saturation_time = check_units(saturation_time, u.s)

        integration_time_value = min(saturation_time)
        return integration_time_value * np.ones(n_bins, dtype=float)
