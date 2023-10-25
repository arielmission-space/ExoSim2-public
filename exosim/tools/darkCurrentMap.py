import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

import exosim.models.signal as signal
import exosim.output as output
from exosim.tasks.task import Task
from exosim.utils import check_units
from exosim.utils import RunConfig


class DarkCurrentMap(Task):
    """
    Produces the channel dark current map

    Returns
    --------
    :class:`~exosim.models.signal.Signal`
        channel dark current map

    Raises
    ------
    TypeError:
        if the output is not a :class:`~exosim.models.signal.Signal` class


    Notes
    -----
    This is a default class with standardized inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid
        output: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("parameters", "parameters dict")
        self.add_task_param("time", "time grid")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("creating dark current map")
        parameters = self.get_task_param("parameters")
        tt = self.get_task_param("time")

        dc_map = self.model(parameters, tt)

        # checking output
        if not isinstance(dc_map, signal.Signal):
            self.error("wrong output format")
            raise TypeError("wrong output format")

        self.debug("dark current map: {}".format(dc_map.data))

        output_file = self.get_task_param("output")
        if output_file:
            if issubclass(output_file.__class__, output.Output):
                dc_map.write(output_file, "dc_map")

        self.set_output(dc_map)

    def model(self, parameters, time):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            dark current efficiency map

        """

        # variance map
        if "dc_sigma" not in parameters["detector"].keys():
            self.error("dc_sigma not provided in detector configuration")
            raise KeyError("dc_sigma not provided in detector configuration")

        if "dc_mean" in parameters["detector"].keys():
            pass

        elif "dc_median" in parameters["detector"].keys():
            check_units(parameters["detector"]["dc_median"], "ct/s")
            self.compute_dc_mean(parameters["detector"])

        check_units(parameters["detector"]["dc_mean"], "ct/s")
        check_units(parameters["detector"]["dc_sigma"], "ct/s")

        variance_map = RunConfig.random_generator.normal(
            parameters["detector"]["dc_mean"].value,
            parameters["detector"]["dc_sigma"].value,
            (
                1,
                parameters["detector"]["spatial_pix"],
                parameters["detector"]["spectral_pix"],
            ),
        )
        idx = np.where(variance_map <= 0)
        variance_map[idx] = 1e-16
        t = [0] * u.hr
        # temporal variation
        if "dc_aging_factor" in parameters["detector"].keys():
            t0 = parameters["detector"]["dc_aging_time_scale"]
            t0 = check_units(t0, u.hr, force=True)

            # creates the aging effect
            dc_aging = RunConfig.random_generator.normal(
                1,
                parameters["detector"]["dc_aging_factor"],
                (
                    1,
                    parameters["detector"]["spatial_pix"],
                    parameters["detector"]["spectral_pix"],
                ),
            )

            # remove values greater than 1, because QE should become smaller
            dc_aging[dc_aging > 1] = 2 - dc_aging[dc_aging > 1]

            # create the map evolution over time
            variance_map = np.vstack((variance_map, variance_map * dc_aging))
            t = np.hstack((t, t0))
            variance_map = variance_map.reshape(
                (t.size, variance_map.shape[1] * variance_map.shape[2])
            )
            f = interp1d(t, variance_map, fill_value="extrapolate", axis=0)
            variance_map = f(time)
            variance_map = variance_map.reshape(
                (
                    time.size,
                    parameters["detector"]["spatial_pix"],
                    parameters["detector"]["spectral_pix"],
                )
            )
            t = time

        # resized map considering an oversampling factor
        resized_map = variance_map.repeat(
            parameters["detector"]["oversampling"], axis=1
        ).repeat(parameters["detector"]["oversampling"], axis=2)
        dc_map = signal.Dimensionless(
            data=resized_map,
            spectral=np.arange(0, parameters["detector"]["spectral_pix"])
            * u.pix,
            time=t,
        )

        dc_map.temporal_rebin(time)

        return dc_map

    def compute_dc_mean(self, detector):
        """
        Computes the mean of the dark current (dc_mean) from the log-normal distributon.

        Notes
        -----
        The probability density function for the log-normal distributon is:

        .. math::
            pdf(x) = \frac{1}{\\sigma x \\sqrt{2\\pi}}
                      \\exp\\left(-\frac{(\\log(x) - \\mu)^2}{2 \\sigma^2}\right)

        The mean of the pdf can be computed as:

        .. math::
            mean = \\exp(\\mu + \frac{s^2}{2})

        where s is the pdf standard deviation, computed by taking the sqrt of the variance, defined as:

        .. math::

            var = \\left(\\exp(\\sigma^2) - 1\right) \\exp(2 \\mu + \\sigma^2)


        Parameters
        ----------
        detector: dict
            Dictionary for the detector. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        -------

        None
            Updates the detector dictionary with the value of dc_mean
        """

        mu = np.log(detector["dc_median"].value)

        root = lambda s: detector["dc_sigma"].value ** 2 - (
            np.exp(s**2) - 1
        ) * np.exp(2 * mu + s**2)

        from scipy.optimize import newton

        sigma = newton(func=root, x0=0.5)

        atol = 1e-6
        if not np.isclose(root(sigma), 0.0, atol=atol):
            self.error(
                "dc_sigma root not close to 0 (tol={:.1e})".format(atol)
            )
            raise ValueError(
                "dc_sigma root not close to 0 (tol={:.1e})".format(atol)
            )

        dc_mean = np.exp(mu + sigma**2 / 2)
        detector.update({"dc_mean": dc_mean * u.ct / u.s})
