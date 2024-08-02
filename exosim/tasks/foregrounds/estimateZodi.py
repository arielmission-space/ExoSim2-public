import astropy.units as u
from astropy.modeling.physical_models import BlackBody

import exosim.models.signal as signal
import exosim.utils.checks as checks
from exosim.tasks.task import Task


class EstimateZodi(Task):
    """
    It estimate the zodiacal radiance in the target direction for a specific wavelength range

    Returns
    -------
    :class:`~exosim.models.signal.Radiance`
        zodiacal radiance

    Examples
    --------
    >>> estimateZodi = EstimateZodi()
    >>> wavelength = np.logspace(np.log10(0.45), np.log10(2.2), 6000) * u.um
    >>> zodi = estimateZodi(wavelength=wavelength, zodiacal_factor=1)

    or, given the pointing direction

    >>> zodi = self.estimateZodi(wavelength=wavelength,  coordinates=(90 * u.deg, -66 * u.deg))

    """

    def __init__(self):
        """
        Parameters
        -----------
        wavelength: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
                wavelength grid. If no units are attached is considered as expressed in `um`
        zodiacal_factor: float
            zodiacal model multiplicative factor
        coordinates: (float, float)
            pointing coordinates as (ra, dec)
        zodi_map: str
            map file containing the zodiacal coefficients per sky positions. A default map is inclluded in ExoSim,
            that contains the coefficients fitted over Kelsall et al. 1998 model.
        """
        self.add_task_param("wavelength", "wavelength range to investigate")
        self.add_task_param(
            "zodiacal_factor", "zodiacal multiplicative factor", None
        )
        self.add_task_param("coordinates", "pointing direction", None)
        self.add_task_param(
            "zodi_map",
            " map file containing the zodiacal coefficients per sky positions",
            None,
        )

    def execute(self):
        self.info("estimating zodiacal foreground")
        wl = self.get_task_param("wavelength")
        zodiacal_factor = self.get_task_param("zodiacal_factor")
        coordinates = self.get_task_param("coordinates")
        zodi_map = self.get_task_param("zodi_map")

        wl = checks.check_units(wl, u.um, self)

        if coordinates:
            zodiacal_factor = self.zodiacal_fit_direction(
                coordinates, zodi_map
            )

        rad = self.model(zodiacal_factor, wl)
        self.debug("zodical radiance: {}".format(rad.data))

        self.set_output(rad)

    def model(self, a, wl):
        r"""
        The used zodiacal model is based on the zodiacal light model presented in Glasse et al. 2010

        .. math::

            I_{zodi}(\lambda) = a \left( 3.5 \cdot 10^{-14} BB(\lambda, 5500 \, K) + 3.52 \cdot 10^{-8} BB(\lambda, 270 \, K) \right)

        where :math:`BB(\lambda, T)` is the Planck black body law and :math:`a` is the fitted coefficient.

        Parameters
        ----------
        a: float
            zodiacal multiplicative factor. Default is 0.
        wl: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
            wavelength grid. If no units are attached is considered as expressed in `um`

        Returns
        -------
        :class:`~exosim.models.signal.Radiance`
            zodiacal radiance

        """
        if not a:
            a = 0
        units = u.W / (u.m**2 * u.um * u.sr)
        bb_1 = BlackBody(5500.0 * u.K)
        bb_1 = bb_1(wl).to(u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl))
        bb_2 = BlackBody(270.0 * u.K)
        bb_2 = bb_2(wl).to(u.W / u.m**2 / u.sr / u.um, u.spectral_density(wl))

        zodi_emission = a * (3.5e-14 * bb_1 + 3.58e-8 * bb_2).to(units)
        return signal.Radiance(wl, zodi_emission)

    def zodiacal_fit_direction(self, coordinates, zodi_map=None):
        r"""
        In this case the :math:`A` coefficient is selected by a precompiled grid.
        The grid has been estimated by fitting our model with Kelsall et al. (1998) data.

        A custom map can be provided, to replace the default one, as long as it matches the format.

        Parameters
        ----------
        coordinates: (float, float)
            pointing coordinates as (ra, dec)
        zodi_map: str (optional)
            map file containing the zodiacal coefficients per sky positions. A default map is inclluded in ExoSim,
            that contains the coefficients fitted over Kelsall et al. 1998 model.

        Returns
        -------
        float
            zodiacal factor for Glasse et al. 2010 zodiacal model

        """
        import os
        from pathlib import Path

        import numpy as np
        from astropy.io.misc.hdf5 import read_table_hdf5

        ra_input = coordinates[0]
        dec_input = coordinates[1]

        if zodi_map:
            zodi_map_file = zodi_map
        else:
            dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
            i = 0
            while (
                "data"
                not in [d.stem for d in Path(dir_path).iterdir() if d.is_dir()]
                or i > 10
            ):
                dir_path = dir_path.parent
                i += 1
            if i > 10:
                self.error("default zodi map file not found")
                raise OSError("default zodi map file not found")
            data_path = os.path.join(dir_path.absolute().as_posix(), "data")
            zodi_map_file = os.path.join(data_path, "Zodi_map.hdf5")
            self.debug("map data:{}".format(zodi_map_file))

        try:
            zodi_table = read_table_hdf5(zodi_map_file)
            self.debug(zodi_table)
            # here we find the minimun distance from the desired point
            distance = (zodi_table["ra_icrs"] * u.deg - ra_input) ** 2 + (
                zodi_table["dec_icrs"] * u.deg - dec_input
            ) ** 2
            idx = np.argmin(distance)
            # TODO implement an interpolator
            self.debug("selected line {}".format(idx))
            self.debug(zodi_table[idx])
            return zodi_table["zodi_coeff"][idx]

        except OSError:
            self.error("Zodi map file not found")
            raise OSError("Zodi map file not found")
