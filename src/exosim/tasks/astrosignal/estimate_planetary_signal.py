import importlib.util
import json

import astropy.units as u
import numpy as np
from astropy.io import ascii
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from exosim.tasks.astrosignal.estimate_astronomical_signal import (
    EstimateAstronomicalSignal,
)
from exosim.utils import RunConfig
from exosim.utils.binning import rebin
from exosim.utils.checks import check_units, find_key
from exosim.utils.types import ArrayType

batman_spec = importlib.util.find_spec("batman")
if batman_spec is not None:
    import batman


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
        ch_parameters: dict | None = None,
        source_parameters: dict | None = None,
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

        # load planetary parameters from sky parameters dictionary
        planet = source_parameters["planet"]
        params = batman.TransitParams()
        params.t0 = check_units(planet["t0"], input_timeline.unit, self, True).value
        params.per = check_units(
            planet["period"], input_timeline.unit, self, True
        ).value
        params.a = planet["sma"]  # (in units of stellar radii)
        params.inc = check_units(planet["inc"], u.deg, self, True).value
        params.ecc = planet["ecc"]
        params.w = check_units(planet["w"], u.deg, self, True).value
        params.limb_dark = planet["limb_darkening"]
        transittype = planet.get("transittype", "primary")

        # load planet rp
        rp_ = planet["rp"]
        if isinstance(rp_, str):
            self.debug("loading rp from file")
            rp = self.load_rp(rp_, wl_grid)
        else:
            try:
                self.debug("rp is a float")
                rp_ = float(rp_)
                rp = rp_ * np.ones(wl_grid.size)
            except (ValueError, TypeError) as exc:
                self.error("rp must be a string or a float")
                raise ValueError(
                    "planet 'rp' must be a string (path to a file) or a float"
                ) from exc

        # load limb darkening coefficients
        ldc_ = planet["limb_darkening_coefficients"]
        try:
            self.debug("loading ldc from file")
            ldc = self.load_ldc(ldc_, wl_grid)
        except FileNotFoundError:
            try:
                self.debug("ldc is a list")
                ldc = json.loads(ldc_)
                ldc = np.array(ldc)
            except ValueError:
                self.error("ldc must be a string or a list")
                raise ValueError(
                    "ldc must be a string (path to a file) or a list of coefficients"
                ) from None

        ldc = np.array(ldc)

        self.debug(f"transit center: {params.t0}")
        # reduce the timeline to the transit duration
        transit_durations = self.get_t14(
            params.inc * u.deg,
            params.a,
            params.per * input_timeline.unit,
            rp,  # use max rp to maximise the transit duration
        )

        margins = (
            1.05  # apply a 5% margin to the transit duration to avoid edge effects
        )
        self.debug(f"margin to transit duration: {margins}")
        transit_durations *= margins
        max_transit_duration = np.nanmax(transit_durations)
        self.debug(f"max transit durations: {transit_durations}")
        start_t = (
            np.abs(
                input_timeline
                - (params.t0 * input_timeline.unit - max_transit_duration / 2)
            )
        ).argmin()
        self.debug(
            f"starting model at: {params.t0 * input_timeline.unit - max_transit_duration / 2}"
        )
        end_t = (
            np.abs(
                input_timeline
                - (params.t0 * input_timeline.unit + max_transit_duration / 2)
            )
        ).argmin()
        self.debug(
            f"ending model at: {params.t0 * input_timeline.unit + max_transit_duration / 2}"
        )

        self.debug(f"Reducing timeline to the transit duration: {start_t} - {end_t}")
        timeline = input_timeline[start_t:end_t]
        self.debug(f"edited timeline size: {timeline.size}")

        # iterate over the wavelength grid
        self.debug("computing the transit model")
        out_model_list = Parallel(n_jobs=RunConfig.n_job, require="sharedmem")(
            delayed(self._compute_transit_model)(
                i, rp, ldc, timeline, params, transittype
            )
            for i in tqdm(
                range(wl_grid.size),
                total=wl_grid.size,
                desc="building transit model over wavelengths",
            )
        )
        out_model = np.array(out_model_list)

        return timeline, out_model

    def load_rp(self, path: str, wl: ArrayType) -> np.ndarray:
        """
        Load the planet radius from a file and rebin it to the input wavelength grid.

        Parameters
        ----------
        path : str
            Path to the file containing the planet radius data.
            The file must include columns named "Wavelength" and "rp/rs" representing the wavelengths
            and the corresponding planet radius (in units of stellar radii), respectively.
        wl : astropy.units.Quantity
            Wavelength grid onto which the planet radius data will be interpolated.

        Returns
        -------
        np.ndarray
            Array of planet radius values (rp/rs) interpolated onto the specified wavelength grid.

        Notes
        -----
        The function reads the data using `ascii.read` and then applies linear interpolation via
        `interp1d` to bin the planet radius values to the provided wavelength grid. Extrapolation is
        enabled for wavelengths that fall outside the range of the data.
        """
        rp_data = ascii.read(path)
        self.debug("binning rp to the input wavelength grid")
        wl_key = find_key(rp_data.keys(), ["wavelength", "wl"], self)
        rp_data[wl_key] = check_units(rp_data[wl_key], wl.unit, self, True).value

        return self._rebin_table(path, wl, rp_data[wl_key], rp_data["rp/rs"])

    def _rebin_table(self, path, wl, table_wl, table_values):
        """
        Rebin a table onto the wavelength grid ``wl``, without leaving NaN.

        :func:`~exosim.utils.binning.rebin` leaves empty bins as NaN when it
        bins (see its documentation). Here the table is the truth: values
        requested outside the table range are linearly extrapolated and
        empty bins inside the range are linearly interpolated. Bins with data
        are returned unchanged. A warning is logged whenever this happens.
        """
        wl_v = np.asarray(getattr(wl, "value", wl), dtype=float)
        tw = np.asarray(getattr(table_wl, "value", table_wl), dtype=float)
        tv = np.asarray(table_values, dtype=float)
        out = np.array(rebin(wl_v, tw, tv, fill_value="extrapolate"), dtype=float)
        # rebin drops NaN and duplicate samples from its inputs: do the same
        ok = ~(np.isnan(tw) | np.isnan(tv))
        tw, tv = tw[ok], tv[ok]
        bad = np.isnan(out)
        if bad.any():
            lo, hi = tw.min(), tw.max()
            n_out = int(np.count_nonzero(bad & ((wl_v < lo) | (wl_v > hi))))
            n_in = int(np.count_nonzero(bad)) - n_out
            func = interp1d(tw, tv, fill_value="extrapolate", kind="linear")
            out[bad] = func(wl_v[bad])
            self.warning(
                f"{path}: table covers {lo:g}-{hi:g} but "
                f"{wl_v.min():g}-{wl_v.max():g} was requested: "
                f"{n_out} values extrapolated linearly outside the table range"
                f", {n_in} empty bins inside it interpolated linearly"
            )
        return out

    def load_ldc(self, path: str, wl: ArrayType) -> np.ndarray:
        """
        Load the limb darkening coefficients from a file and rebin them to the input wavelength grid.

        Parameters
        ----------
        path : str
            Path to the file containing the limb darkening coefficients data.
            The file must include a column representing the wavelength (e.g. "wavelength" or "wl") and one or more
            columns with the corresponding limb darkening coefficients.
        wl : astropy.units.Quantity
            The wavelength grid to which the limb darkening coefficients will be rebinned.

        Returns
        -------
        np.ndarray
            An array of rebinned limb darkening coefficients. If the file contains multiple coefficient columns,
            the resulting array will be 2-dimensional with shape (n_coeff, len(wl)), where n_coeff is the number
            of coefficient columns.

        Notes
        -----
        The function utilises `ascii.read` to load the data from the file.
        The wavelength column is identified using the `find_key` function by searching for keys such as "wavelength" or "wl".
        Each remaining column is rebinned to the provided wavelength grid using the `rebin` function.
        """
        ldc_data = ascii.read(path)

        wl_key = find_key(ldc_data.keys(), ["wavelength", "wl"], self)
        ldc_wl = ldc_data[wl_key]
        ldc_wl = check_units(ldc_wl, wl.unit, self, True)

        ldc_out = []
        self.debug("binning ldc to the input wavelength grid")
        for k in ldc_data.colnames:
            if "ldc" in k:
                ldc = self._rebin_table(path, wl, ldc_wl, ldc_data[k])
                ldc_out.append(ldc)
        return ldc_out

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
        t14 = period / np.pi / sma * np.sqrt((1 + rp) ** 2 - impact_parameter**2)
        self.debug(f"transit duration: {t14}")
        return t14

    def _compute_transit_model(self, i, rp, ldc, timeline, params, transittype):
        """
        Compute the transit model for a single wavelength index.

        Parameters
        ----------
        i : int
            Wavelength index.
        rp : np.ndarray
            Array of planet radius values for each wavelength.
        ldc : np.ndarray
            Array of limb darkening coefficients.
        timeline : astropy.units.Quantity
            Timeline reduced to the transit duration.
        params : batman.TransitParams
            Transit parameters.
        transittype : str
            Type of transit (e.g. 'primary').

        Returns
        -------
        np.ndarray
            The computed light curve for the given wavelength.
        """
        params.rp = rp[i]  # Set planet radius for the current wavelength
        if ldc.ndim == 2:
            params.u = list(ldc[:, i])
        else:
            params.u = ldc
        # Initialise batman model and compute light curve
        m = batman.TransitModel(params, timeline, transittype=transittype)
        return m.light_curve(params)
