import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

import exosim.utils as utils
from exosim.tasks.instrument.computeSourcesPointingOffset import angle_of_view
from exosim.tasks.task import Task
from exosim.utils.checks import check_units


class EstimateChJitter(Task):
    """
    It scales the pointing jitter expressed as :math:`deg` into the pixel unit according to the channel plate scale
    and interpolates it to a time grid which is aligned to the detector readout time grid.

    Returns
    --------
    (:class:`~numpy.ndarray`, :class:`~numpy.ndarray`)
        Tuple containing the pointing jitter in the spatial and spectral direction expressed in units of pixels.

    """

    def __init__(self):
        """
        Parameters
        __________
        pointing_jitter: (:class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`,  :class:`~astropy.units.Quantity`)
            Tuple containing the pointing jitter in the spatial and spectral direction expressed in units of deg.
        parameters: dict
            dictionary containing the channel parameters.
            This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """
        self.add_task_param("pointing_jitter", " ")
        self.add_task_param("parameters", "channel parameters dict")

    def execute(self):
        parameters = self.get_task_param("parameters")
        jitter_spa, jitter_spe, jitter_time = self.get_task_param(
            "pointing_jitter"
        )

        jitter_time = check_units(jitter_time, "s")
        jitter_time_step = jitter_time[1] - jitter_time[0]
        jitter_freq = check_units(jitter_time_step, "Hz")
        readout_freq = check_units(
            parameters["readout"]["readout_frequency"], "Hz"
        )

        new_freq = (
            np.ceil(jitter_freq.value / readout_freq.value) * readout_freq
        )

        new_step = check_units(new_freq, "s")

        new_jitter_time = utils.grids.time_grid(
            jitter_time[0], jitter_time[-1], new_step
        )
        new_jitter_time = new_jitter_time.to(u.s)

        jitter_spe_inter = interp1d(jitter_time, jitter_spe)
        jitter_spe = jitter_spe_inter(new_jitter_time) * jitter_spe.unit

        jitter_spa_inter = interp1d(jitter_time, jitter_spa)
        jitter_spa = jitter_spa_inter(new_jitter_time) * jitter_spa.unit

        aov_spatial, aov_spectral = angle_of_view(
            parameters["detector"]["plate_scale"],
            parameters["detector"]["delta_pix"],
            parameters["detector"]["oversampling"],
        )

        offset_spatial = jitter_spa.to(u.deg) / aov_spatial
        offset_spectral = jitter_spe.to(u.deg) / aov_spectral

        self.set_output(
            [
                jitter_spe,
                jitter_spa,
                offset_spatial,
                offset_spectral,
                new_jitter_time,
            ]
        )
