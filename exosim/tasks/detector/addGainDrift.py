from copy import deepcopy
from typing import Optional

import astropy.units as u
import numpy as np

from exosim.models.signal import Signal
from exosim.output import Output
from exosim.tasks.task import Task
from exosim.utils import RunConfig
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.types import ArrayType


class AddGainDrift(Task):
    r"""
    It adds a gain noise map to the array.

    The gain Drift is modeled as a polynomial model for a spectral and time dependent modulation.


    Notes
    -----
    This is a default class with standardised inputs and outputs.
    The user can load this class and overwrite the "model" method
    to implement a custom Task to replace this.
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        parameters: dict
            channel parameters dictionary
        time: :class:`~astropy.units.Quantity`
            sub-exposures times
        integration_times: :class:`~astropy.units.Quantity`
            sub-exposures integration times
        outputs: :class:`~exosim.output.output.Output` (optional)
            output file
        """

        self.add_task_param("subexposures", "sub-exposures cached signal")
        self.add_task_param("parameters", "channel parameters dictionary")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("adding gain noise")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        output = self.get_task_param("output")

        self.model(subexposures, parameters, output)

    def model(
        self,
        subexposures: Signal,
        parameters: dict,
        output: Optional[Output] = None,
    ) -> None:
        """
        Apply a gain drift model to the subexposures.

        This method models the gain drift as a polynomial
        trend based on time and wavelength.

        Parameters
        ----------
        subexposures : Signal
            Sub-exposures cached signal.
        parameters : dict
            A dictionary containing parameters for the noise model. Expected keys and sub-keys include:
            - 'readout': dict with 'readout_frequency'
            - 'detector': dict with keys 'gain_w0', 'gain_f0', 'gain_coeff_order_t',
                'gain_coeff_t_min', 'gain_coeff_t_max', 'gain_coeff_order_w',
                'gain_coeff_w_min', and 'gain_coeff_w_max'.
        integration_times : np.ndarray
            Sub-exposures integration times.
        output : Optional[Output]
            Output file to write information, defaults to None.

        Returns
        -------
        None
            This method does not return a value but updates the subexposures dataset
            with applied gain noise.

        Notes
        -----
        The method computes Brownian noise based on the frequency parameters in 'parameters' and
        applies a polynomial trend to it. The trend's coefficients are randomly generated within
        specified ranges. The resulting gain noise is then rebinned and applied to the subexposures.
        """

        if output and issubclass(output.__class__, Output):
            out_grp = output.create_group("gain noise")
        else:
            out_grp = None

        time = subexposures.time * u.Unit(subexposures.time_units)
        integration_times = subexposures.metadata["integration_times"]
        if isinstance(integration_times, dict):
            integration_times = integration_times["value"] * u.Unit(
                integration_times["unit"]
            )
            integration_times.to(u.s)

        # Calculating polynomial trend
        order_t = parameters["detector"]["gain_coeff_order_t"]
        coeff_t_min = parameters["detector"]["gain_coeff_t_min"]
        coeff_t_max = parameters["detector"]["gain_coeff_t_max"]
        coeff_t = RunConfig.random_generator.uniform(
            low=coeff_t_min, high=coeff_t_max, size=order_t + 1
        )
        if out_grp is not None:
            out_grp.write_list("gain coeff_t", coeff_t)
        y_t = self._pol_t(time, coeff_t)

        wl_ax = np.arange(0, subexposures.dataset.shape[2], 1)
        order_w = parameters["detector"]["gain_coeff_order_w"]
        coeff_w_min = parameters["detector"]["gain_coeff_w_min"]
        coeff_w_max = parameters["detector"]["gain_coeff_w_max"]
        coeff_w = RunConfig.random_generator.uniform(
            low=coeff_w_min, high=coeff_w_max, size=order_w + 1
        )
        if out_grp is not None:
            out_grp.write_list("gain coeff_w", coeff_w)
        y_w = self._pol_w(wl_ax, coeff_w)

        z = np.zeros([y_t.size, y_w.size])
        for i, y in enumerate(y_t):
            z[i] = y * y_w

        if "gain_drift_amplitude" in parameters["detector"].keys():
            amplitude = parameters["detector"]["gain_drift_amplitude"]
        elif "gain_drift_amplitude_range_min" in parameters["detector"].keys():
            amplitude_min = parameters["detector"][
                "gain_drift_amplitude_range_min"
            ]
            amplitude_max = parameters["detector"][
                "gain_drift_amplitude_range_max"
            ]
            amplitude = RunConfig.random_generator.uniform(
                amplitude_min, amplitude_max
            )
            if out_grp is not None:
                out_grp.write_list(
                    "gain amplitude random_seed", RunConfig.random_seed
                )
        else:
            self.error("missing amplitude definition for gain")

        dinamic_range = z.max() - z.min()
        z *= amplitude / dinamic_range
        z += 1

        if out_grp is not None:
            out_grp.write_array("gain drift", z)
            out_grp.write_array("coeff_t", coeff_t)
            out_grp.write_scalar("amplitude", amplitude)
            out_grp.write_array("coeff_w", coeff_w)

        # Applying the multiplicative noise
        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying gain noise"
        ):
            chunk_data = deepcopy(subexposures.dataset[chunk])
            subexposures.dataset[chunk] = (
                chunk_data * z[chunk[0].start : chunk[0].stop, np.newaxis, :]
            )
            subexposures.output.flush()

    @staticmethod
    def _pol_t(tt: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """
        Polynomial trend calculation for time.

        Parameters
        ----------
        tt : np.ndarray
            Time array.
        coef : np.ndarray
            Coefficients for the polynomial trend.

        Returns
        -------
        np.ndarray
            Calculated polynomial trend for time.
        """
        x = ((tt - tt.min()) / (tt.max() - tt.min())).si.value
        y = np.zeros(tt.size)
        for i, c in enumerate(coef):
            y += c * x**i
        return y

    @staticmethod
    def _pol_w(wl: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """
        Polynomial trend calculation for wavelength.

        Parameters
        ----------
        wl : np.ndarray
            spectral axis array [pixel unit].
        coef : np.ndarray
            Coefficients for the polynomial trend.

        Returns
        -------
        np.ndarray
            Calculated polynomial trend for wavelength.
        """
        x = (wl - wl.min()) / (wl.max() - wl.min())
        y = np.zeros(wl.size)
        for i, c in enumerate(coef):
            y += c * x**i
        return y

    @staticmethod
    def _psd(f: np.ndarray, w0: float, f0: float) -> np.ndarray:
        """
        Calculate the power spectral density.

        Parameters
        ----------
        f : np.ndarray
            Frequency array.
        w0 : float
            Parameter for PSD calculation.
        f0 : float
            Parameter for PSD calculation.

        Returns
        -------
        np.ndarray
            Calculated power spectral density.
        """
        psd = np.zeros_like(f)
        psd[1:] = w0 * np.sqrt((f0 / f[1:]) ** 2 + 1)
        return psd

    @staticmethod
    def _noise_generator(
        f: np.ndarray, psd: np.ndarray, out=None
    ) -> np.ndarray:
        """
        Generate noise based on provided frequency and power spectral density.

        Parameters
        ----------
        f : np.ndarray
            Array of frequencies.
        psd : np.ndarray
            Power spectral density corresponding to the frequencies.
        out : object, optional
            Output object to write information, defaults to None.

        Returns
        -------
        np.ndarray
            Generated noise array.
        """
        # Randomize amplitude
        noise = RunConfig.random_generator.normal(
            0, psd * np.sqrt(f[-1]), psd.size
        )
        if out is not None:
            out.write_list("gain amplitude random_seed", RunConfig.random_seed)

        # Randomize phase
        phi = RunConfig.random_generator.normal(-0.5, 1, psd.size) * np.pi
        if out is not None:
            out.write_list("gain phase random_seed", RunConfig.random_seed)

        noisef = noise * np.exp(1j * phi)
        noise = np.fft.irfft(noisef) * np.sqrt(2 * noisef.size - 1)
        return noise
