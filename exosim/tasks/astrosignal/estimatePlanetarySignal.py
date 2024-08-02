import json

import astropy.units as u
import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from exosim.tasks.astrosignal.estimateAstronomicalSignal import (
    EstimateAstronomicalSignal,
)
from exosim.utils.binning import rebin
from exosim.utils.checks import check_units
from exosim.utils.types import ArrayType


class EstimatePlanetarySignal(EstimateAstronomicalSignal):
    """
    This task estimates the planetary signal.
    The signal parameters are extracted from the sky configuration file.
    This Task is a wrapper for the batman package: it parses the planet parameters and computes the transit model.
    The transit model is computed at the input wavelength grid.
    By default, the transit type is set to 'primary' (i.e. the planet transits in front of the star).
    The planetary sarius can be parsed as a string (path to a file) or as a float (in units of stellar radii).
    If a file is indicated, the planetary radius is binned or interpolated to the input wavelength grid.

    Based on Batman package by L. Kreidberg (https://ui.adsabs.harvard.edu/abs/2015PASP..127.1161K/abstract).
    """

    def model(
        self,
        timeline: ArrayType,
        wl_grid: ArrayType,
        ch_parameters: dict = {},
        source_parameters: dict = {},
    ) -> ArrayType:
        """
        Parameters
        ----------
        timeline : :class:`astropy.units.Quantity`
            timeline to compute the signal
        wl_grid : :class:`astropy.units.Quantity`
            wavelength grid
        ch_parameters : dict, optional
            channel parameters, by default {}
        source_parameters : dict
            source parameters, by default {}

        Returns
        -------
        ArrayType
            returns the planetary signal in a 2D array (wavelength, time)
        """

        self.info("Estimating planetary signal")
        # set timeline units to sec
        input_timeline = check_units(timeline, u.s, self, True)

        import batman

        # load planetary parameters from sky parameters dictionary
        planet = source_parameters["planet"]
        params = batman.TransitParams()
        params.t0 = check_units(
            planet["t0"], input_timeline.unit, self, True
        ).value
        params.per = check_units(
            planet["period"], input_timeline.unit, self, True
        ).value
        params.a = planet["sma"]  # (in units of stellar radii)
        params.inc = check_units(planet["inc"], u.deg, self, True).value
        params.ecc = planet["ecc"]
        params.w = check_units(planet["w"], u.deg, self, True).value
        params.limb_dark = planet["limb_darkening"]
        raw_u = planet["limb_darkening_coefficients"]
        params.u = json.loads(raw_u)

        transittype = (
            planet["transittype"] if "transittype" in planet else "primary"
        )

        # load planet rp
        rp_ = planet["rp"]
        if isinstance(rp_, str):
            rp_data = ascii.read(rp_)
            rp_func = interp1d(
                rp_data["Wavelength"],
                rp_data["rp/rs"],
                assume_sorted=False,
                bounds_error=False,
                fill_value="extrapolate",
                kind="linear",
            )
            rp = rp_func(wl_grid)
        else:
            try:
                rp_ = float(rp_)
                rp = rp_ * np.ones(wl_grid.size)
            except ValueError:
                self.error("rp must be a string or a float")

        # reduce the timeline to the transit duration
        transit_durations = self.get_t14(
            params.inc * u.deg,
            params.a,
            params.per * input_timeline.unit,
            rp,  # use max rp to maximise the transit duration
        )
        margins = 1.05  # apply a 5% margin to the transit duration to avoid edge effects
        transit_durations *= margins
        max_transit_duration = np.nanmax(transit_durations)
        start_t = (
            np.abs(
                input_timeline
                - (params.t0 * input_timeline.unit - max_transit_duration / 2)
            )
        ).argmin()
        end_t = (
            np.abs(
                input_timeline
                - (params.t0 * input_timeline.unit + max_transit_duration / 2)
            )
        ).argmin()
        timeline = input_timeline[start_t:end_t]

        # iterate over the wavelength grid
        out_model = np.zeros((wl_grid.size, timeline.size))
        for i in tqdm(range(wl_grid.size)):
            params.rp = rp[i]  # planet radius (in units of stellar radii)

            # initialise batman model
            m = batman.TransitModel(params, timeline, transittype=transittype)

            out_model[i] = m.light_curve(params)

        return timeline, out_model

    def get_t14(
        self, inc: u.Quantity, sma: float, period: u.Quantity, rp: np.ndarray
    ) -> u.Quantity:
        """t14
        Calculates the transit time based on the work of Seager, S., & Mallen-Ornelas, G. 2003, ApJ, 585, 1038

        Parameters
        __________
        inc: :class:`astropy.units.Quantity`
            Planet oprbital inclination
        sma: float
            Semimajor axis in stellar units
        period: :class:`astropy.units.Quantity`
            Orbital period
        rp: np.ndarray
            Planet radius in stellar units

        Returns
        __________
        transit duration : :class:`astropy.units.Quantity`
            Returns the transit duration


        """
        period = period.to(u.s)
        impact_parameter = np.cos(inc.to(u.rad)) * sma
        t14 = (
            period / np.pi / sma * np.sqrt((1 + rp) ** 2 - impact_parameter**2)
        )
        return t14
