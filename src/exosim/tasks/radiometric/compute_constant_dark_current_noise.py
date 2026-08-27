import numpy as np
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeConstantDarkCurrentNoise(Task):
    r"""
    Computes the dark current noise for each aperture in the provided table.

    The total dark current counts are computed for a fixed exposure time (default: 1 hour), and the noise is given by the square root of the expected counts.

    Returns
    -------
    astropy.table.QTable
        The input table with two additional columns:
        - 'aperture_area': calculated area of the aperture
        - 'darkcurrent_noise': computed dark current noise (standard deviation of counts) for 1 hour exposure
    astropy.units.Quantity
        Array of dark current noise values for each aperture.

    Raises
    ------
    ValueError
        If required keys are missing in the description dictionary.

    Notes
    -----
    The dark current noise is calculated as:
        noise = sqrt(multiaccum_gain * dark_current_rate * aperture_area * exposure_time)
    where:
        - dark_current_rate is in ct/s
        - aperture_area is in pixels (or other consistent units)
        - exposure_time is set to 1 hour
    """

    def __init__(self):
        """
        Parameters
        ----------
        signal : numpy.ndarray or astropy.units.Quantity
            Signal array for normalization.
        aperture_table : astropy.table.QTable or array-like
            Table containing the aperture information. Must include the columns:
            - 'spectral_size': size of the aperture in the spectral direction
            - 'spatial_size': size of the aperture in the spatial direction
            - 'aperture_shape': shape of the aperture ('rectangular' or 'elliptical')
        description : dict, optional
            Dictionary containing the channel description. Must include:
            - 'dark_current': description of the dark current (required key)
            - 'dc_mean': mean dark current value (required key, astropy.units.Quantity, in ct/s)
        multiaccum_gain : numpy.ndarray or float, optional
            Multiaccum gain factor for shot noise calculation.

        """
        self.add_task_param("signal", "signal array")
        self.add_task_param("aperture_table", "table containing aperture information")
        self.add_task_param("description", "channel description", None)
        self.add_task_param("multiaccum_gain", "multiaccum gain factor", None)

    def execute(self):
        self.debug("compute dark current noise")
        signal = self.get_task_param("signal")
        aperture_table = self.get_task_param("aperture_table")
        description = self.get_task_param("description")
        multiaccum_gain = self.get_task_param("multiaccum_gain")

        table, model = self.model(signal, aperture_table, description, multiaccum_gain)

        if not isinstance(table, QTable):
            self.error("wrong output format")
            raise TypeError("wrong output format")
        required_columns = ["darkcurrent_noise"]
        for col in required_columns:
            if col not in table.colnames:
                self.error(f"missing required column: {col}")
                raise KeyError(f"missing required column: {col}")
        self.set_output([table, model])

    def model(self, signal, aperture_table, description, multiaccum_gain):
        """
        Compute the dark current noise for each aperture using the provided parameters.

        This method estimates the dark current noise based on the mean dark current value, the aperture area, and the multiaccum gain.
        The area of each aperture is taken from the ``aperture_size`` column of the input table.
        The expected dark current counts are computed for a fixed exposure time of 1 hour, and the noise is given by the square root of the expected counts.
        The result is normalized by the input signal.

        The total dark current variance is computed as:

        .. math::

            \\mathrm{dark\\_current\\_variance} = G \\cdot \\mu_\\mathrm{DC} \\cdot A

        where

        - :math:`G` is the multiaccum gain,
        - :math:`\\mu_\\mathrm{DC}` is the mean dark current (in ct/s),
        - :math:`A` is the aperture area (in pixels or consistent units).

        The dark current noise for a 1 hour exposure is then:

        .. math::

            \\mathrm{dark\\_current\\_noise} = \\sqrt{\frac{\\mathrm{dark\\_current\\_variance}}{3600} \\cdot 3600}

        The result is normalized by the input signal:

        .. math::

            \\mathrm{dark\\_current\\_noise\\_norm} = \frac{\\mathrm{dark\\_current\\_noise}}{S}

        where :math:`S` is the signal.

        Parameters
        ----------
        signal : astropy.units.Quantity
            Signal array for normalization.
        aperture_table : astropy.table.QTable or array-like
            Table containing the aperture information with 'aperture_size' column.
        description : dict
            Dictionary containing the channel description with 'detector' key containing
            'dark_current' and 'dc_mean' keys.
        multiaccum_gain : numpy.ndarray or float
            Multiaccum gain factor for shot noise calculation.

        Returns
        -------
        astropy.table.QTable
            The input table with additional 'dark_current_variance' and 'darkcurrent_noise' columns.
        astropy.units.Quantity
            Array of dark current noise values for each aperture (normalized by the signal).
        """

        aperture_table = QTable(aperture_table)
        noise_table = QTable()

        if "dark_current" not in description["detector"]:
            raise ValueError("Dark current description is missing in the input.")

        if "dc_mean" not in description["detector"]:
            raise ValueError("Dark current mean value is missing in the input.")
        dark_mean = description["detector"]["dc_mean"]
        dark_mean = check_units(dark_mean, u.ct / u.s)

        dark_current_variance = (
            multiaccum_gain * dark_mean * aperture_table["aperture_size"] * u.ct / u.s
        )
        dark_current_noise = np.sqrt(dark_current_variance / u.hr.to(u.s) * u.hr)

        # Convert dark current noise to per signal unit
        if not isinstance(signal, u.Quantity):
            raise TypeError("Signal must be a Quantity")
        dark_current_noise = dark_current_noise / signal

        noise_table["dark_current_variance"] = dark_current_variance

        noise_table["darkcurrent_noise"] = dark_current_noise

        return noise_table, dark_current_noise
