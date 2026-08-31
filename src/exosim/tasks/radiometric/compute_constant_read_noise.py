import numpy as np
from astropy import units as u
from astropy.table import QTable

from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class ComputeConstantReadNoise(Task):
    r"""
    Computes the read noise for each aperture in the provided table.

    Returns
    -------
    astropy.table.QTable
        The input table with two additional columns:
        - 'read_noise_variance': computed read noise variance for each aperture
        - 'read_noise': computed total read noise for each aperture, normalized by the signal
    astropy.units.Quantity
        Array of read noise values for each aperture (normalized by the signal).

    Raises
    ------
    ValueError
        If required keys are missing in the description dictionary.
    TypeError
        If the signal is not a Quantity.
    """

    def __init__(self):
        """
        Parameters
        ----------
        signal : astropy.units.Quantity
            Signal array for normalization.
        aperture_table : astropy.table.QTable or array-like
            Table containing the aperture information. Must include the columns:
            - 'aperture_size': area of the aperture
            - 'frame_time': frame time for each aperture
        description : dict, optional
            Dictionary containing the channel description. Must include:
            - 'read_noise_sigma': read noise per pixel (astropy.units.Quantity, in ct)
        multiaccum_gain : float or Quantity
            Multiaccum gain factor.
        """

        self.add_task_param("signal", "signal array")
        self.add_task_param("aperture_table", "table containing aperture information")
        self.add_task_param("description", "channel description", None)
        self.add_task_param("multiaccum_gain", "multiaccum gain factor", None)

    def execute(self):
        self.debug("compute read noise")

        signal = self.get_task_param("signal")
        aperture_table = self.get_task_param("aperture_table")
        description = self.get_task_param("description")
        multiaccum_gain = self.get_task_param("multiaccum_gain")

        table, model = self.model(signal, aperture_table, description, multiaccum_gain)
        self.set_output([table, model])

    def model(self, signal, aperture_table, description, multiaccum_gain):
        r"""
        Compute the read noise for each aperture using the provided parameters.

        This method estimates the total read noise based on the per-pixel read noise value, the aperture area, the frame time, and the multiaccum gain.
        The area of each aperture is taken from the ``aperture_size`` column of the input table.

        The total read noise variance is computed as:

        .. math::

            \mathrm{read\_noise\_variance} = G \cdot \sigma_\mathrm{RN}^2 \cdot \frac{A}{t_\mathrm{frame}}

        where

        - :math:`G` is the multiaccum gain,
        - :math:`\sigma_\mathrm{RN}` is the read noise per pixel (in ct),
        - :math:`A` is the aperture area (in pixels),
        - :math:`t_\mathrm{frame}` is the frame time (in seconds).

        The read noise is then:

        .. math::

            \mathrm{read\_noise} = \sqrt{\mathrm{read\_noise\_variance}}

        The result is normalized by the input signal:

        .. math::

            \mathrm{read\_noise\_norm} = \frac{\mathrm{read\_noise}}{S}

        where :math:`S` is the signal.

        Parameters
        ----------
        signal : astropy.units.Quantity
            Signal array for normalization.
        aperture_table : astropy.table.QTable or array-like
            Table containing the aperture information with 'aperture_size' and 'frame_time' columns.
        description : dict
            Dictionary containing the channel description with detector read noise parameters.
        multiaccum_gain : float or Quantity
            Multiaccum gain factor.

        Returns
        -------
        astropy.table.QTable
            The input table with additional 'read_noise_variance' and 'read_noise' columns.
        astropy.units.Quantity
            Array of read noise values for each aperture (normalized by the signal).
        """

        aperture_table = QTable(aperture_table)
        noise_table = QTable()

        if "read_noise_sigma" not in description["detector"]:
            raise ValueError("Read noise sigma is missing in the input.")

        read_noise_sigma = description["detector"]["read_noise_sigma"]
        read_noise_sigma = check_units(read_noise_sigma, u.ct)

        if "frame_time" not in aperture_table.colnames:
            raise ValueError("Frame time is missing in the input.")
        frame_time = aperture_table["frame_time"]
        frame_time = check_units(frame_time, u.s)

        # Calculate total read noise for each aperture
        read_noise_variance = (
            multiaccum_gain
            * read_noise_sigma**2
            * aperture_table["aperture_size"]
            / frame_time
            / u.s
        )
        read_noise = np.sqrt(read_noise_variance / u.hr.to(u.s) * u.hr)

        # Convert read noise to per signal unit
        if not isinstance(signal, u.Quantity):
            raise TypeError("Signal must be a Quantity")
        read_noise = read_noise / signal

        noise_table["read_noise_variance"] = read_noise_variance
        noise_table["read_noise"] = read_noise

        return noise_table, noise_table["read_noise"]
