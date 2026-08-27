import numpy as np
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeObservationEfficiency(Task):
    r"""
    Task to compute the observation efficiency for each entry in a radiometric table.

    This class determines a constant observation efficiency value that is applied to each
    aperture in the radiometric table. The observation efficiency represents the fraction
    of time that the detector is actively observing (not blocked by shutters,
    choppers, or other mechanisms).


    Parameters
    ----------
    radiometric_table : astropy.table.QTable or array-like
        Table containing the radiometric information. Must include the column:
        - 'integration_time': integration time for each entry (astropy.units.Quantity, in s)
    description : dict
        Dictionary containing the channel description. Must include:
        - 'integration_time': integration time (required key, astropy.units.Quantity, in s)
    channel_name : str, optional
        Name of the channel to filter the table. If None, all entries are used.

    Raises
    ------
    TypeError
        If the output is not a float.

    """

    def __init__(self):
        self.add_task_param(
            "radiometric_table", "table containing radiometric information"
        )
        self.add_task_param("description", "channel description", None)
        self.add_task_param("channel_name", "name of the channel", None)

    def execute(self):
        self.debug("compute observation efficiency")

        radiometric_table = self.get_task_param("radiometric_table")
        description = self.get_task_param("description")
        channel_name = self.get_task_param("channel_name")

        observation_efficiency = self.model(
            radiometric_table, description, channel_name
        )
        if not isinstance(observation_efficiency, float | np.floating):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        self.set_output(observation_efficiency)

    def model(self, radiometric_table, description, channel_name):
        """
        Compute the observation efficiency for each aperture in the radiometric table.

        This method calculates a constant observation efficiency value that is applied to each
        aperture in the radiometric table. The observation efficiency represents the fraction
        of time that the detector is actively observing (not blocked by shutters,
        choppers, or other mechanisms).

        Parameters
        ----------
        radiometric_table : astropy.table.QTable
            Table containing radiometric information for each aperture. Must include
            a ``ch_name`` column if channel_name is specified for filtering.
        description : dict
            Channel description dictionary containing the observation efficiency specification
            under ``description["radiometric"]["observation_efficiency"]``. If not present,
            assumes an observation efficiency of 1.0 (100% observing efficiency).
        channel_name : str or None
            Name of the specific channel to process. If provided, only apertures
            matching this channel name are considered. If None, all apertures
            in the table are processed.

        Returns
        -------
        float
            Constant observation efficiency value (dimensionless) applied to all apertures.

        Notes
        -----
        - The observation efficiency must be dimensionless and typically ranges from 0 to 1
        - An observation efficiency of 1.0 means 100% observing efficiency (no interruptions)
        - An observation efficiency of 0.5 means 50% observing efficiency (half the time blocked)
        - If no observation efficiency is specified in the description, defaults to 1.0 with a warning
        - The number of output values matches the number of apertures for the specified channel
        """

        radiometric_table = QTable(radiometric_table)

        # Check if radiometric section exists and has observation_efficiency
        if (
            "radiometric" not in description
            or "observation_efficiency" not in description["radiometric"]
        ):
            self.warning(
                "No observation efficiency specified in the description. Assuming observation efficiency = 1."
            )
            observation_efficiency = 1.0
        else:
            observation_efficiency = description["radiometric"][
                "observation_efficiency"
            ]
            observation_efficiency = check_units(
                observation_efficiency, u.dimensionless_unscaled
            )
            observation_efficiency = float(observation_efficiency.value)

        return observation_efficiency
