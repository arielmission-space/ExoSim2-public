from astropy import units as u
from astropy.table import QTable

from exosim.utils.checks import check_units

from .compute_observation_efficiency import ComputeObservationEfficiency


class ComputeObservationEfficiencyFromDeadTime(ComputeObservationEfficiency):
    r"""
    Task to compute observation efficiency based on detector dead time.

    This class calculates the observation efficiency for each aperture in the radiometric
    table by considering the detector's dead time. The observation efficiency is computed
    as the ratio of integration time to the sum of integration time and dead time:

    .. math::
        efficiency = \frac{t_{int}}{t_{int} + t_{dead}}

    Where:
    - :math:`t_{int}` is the integration time for each aperture
    - :math:`t_{dead}` is the detector dead time (constant for all apertures)

    The dead time represents the period after each readout during which the detector
    cannot acquire new data, reducing the overall observational efficiency.

    Parameters
    ----------
    radiometric_table : astropy.table.QTable or array-like
        Table containing radiometric information. Must include columns:
        - 'integration_time': integration time for each aperture (astropy.units.Quantity, in s)
        - 'ch_name': channel name for filtering (if channel_name is specified)
    description : dict
        Dictionary containing the channel description. Should include:
        - 'radiometric'/'dead_time': detector dead time (astropy.units.Quantity, in s)
        If not provided, assumes dead_time = 0 s (100% efficiency).
    channel_name : str, optional
        Name of the channel to filter the table. If provided, only apertures
        matching this channel name are processed.

    Returns
    -------
    float
        Array of observation efficiency values (dimensionless) for each aperture
        in the filtered radiometric table. Values range from 0 to 1, where:
        - 1.0 = 100% efficiency (no dead time)
        - 0.5 = 50% efficiency (dead time equals integration time)

    Notes
    -----
    This implementation accounts for detector-specific dead time effects, making it
    more accurate than constant efficiency models for detectors with significant
    readout overhead.

    """

    def model(self, radiometric_table, description, channel_name):
        """
        Compute observation efficiency based on detector dead time.

        This method calculates the observation efficiency for each aperture by
        considering the detector's dead time. The efficiency is computed as:
        efficiency = integration_time / (integration_time + dead_time)

        Parameters
        ----------
        radiometric_table : astropy.table.QTable
            Table containing radiometric information for each aperture. Must include:
            - 'integration_time' column (astropy.units.Quantity, in s)
            - 'ch_name' column if channel_name is specified for filtering
        description : dict
            Channel description dictionary. Should contain the dead time specification
            under ``description["radiometric"]["dead_time"]`` (astropy.units.Quantity, in s).
            If not present, assumes dead_time = 0 s (perfect efficiency).
        channel_name : str or None
            Name of the specific channel to process. If provided, only apertures
            matching this channel name are considered. If None, all apertures
            in the table are processed.

        Returns
        -------
        float
            Array of observation efficiency values (dimensionless) for each aperture.
            Each value represents the fraction of time the detector is actively
            observing for that specific aperture's integration time.

        Notes
        -----
        - Efficiency values range from 0 to 1 (0% to 100%)
        - Longer integration times result in higher efficiency (less impact from dead time)
        - If dead_time = 0, efficiency = 1.0 for all apertures
        - If dead_time equals integration_time, efficiency = 0.5 (50%)
        - Dead time is assumed constant across all apertures for a given detector
        """

        radiometric_table = QTable(radiometric_table)
        radiometric_table = radiometric_table[
            radiometric_table["ch_name"] == channel_name
        ]

        # Check if radiometric section exists and has dead_time
        if (
            "radiometric" not in description
            or "dead_time" not in description["radiometric"]
        ):
            self.warning(
                "No dead time specified in the description. Assuming dead time = 0."
            )
            dead_time = 0.0 * u.s
        else:
            dead_time = description["radiometric"]["dead_time"]
            dead_time = check_units(dead_time, u.s)

        integration_time = check_units(radiometric_table["integration_time"], u.s)

        observation_efficiency = integration_time / (integration_time + dead_time)
        observation_efficiency = check_units(
            observation_efficiency, u.dimensionless_unscaled
        )
        return observation_efficiency.value
