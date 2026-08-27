import gc

import astropy.units as u
import numpy as np

import exosim.tasks.instrument as instrument
from exosim.models.signal import Signal
from exosim.tasks.task import Task
from exosim.utils.klass_factory import find_task


class CreateFocalPlaneArray(Task):
    """
    It produces the focal plane array

    Returns
    -------
    :class:`~exosim.models.signal.Signal`
        focal plane array (no time evolution)
    """

    def __init__(self):
        """
        Parameters
        __________
        parameters: dict
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
        efficiency: :class:`~exosim.models.signal.Dimensionless`
            channel efficiency
        """

        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("efficiency", "channel efficiency")

    def execute(self):
        parameters = self.get_task_param("parameters")

        focal_plane_dimension = (
            parameters["detector"]["spatial_pix"],
            parameters["detector"]["spectral_pix"],
        )
        self.debug(f"focal plane dimensions: {focal_plane_dimension}")

        if "oversampling" in parameters["detector"]:
            if "irf_task" in parameters["detector"]:
                oversampling = parameters["detector"]["oversampling"]
            else:
                self.warning(
                    "The oversampling factor is set but irf_task is not provided in detector configuration. Oversampling factor is set to 1."
                )
                oversampling = 1
        else:
            oversampling = 1
        self.debug(f"pixel over sampling factor: {oversampling}")

        focal_plane_dimension = tuple([d * oversampling for d in focal_plane_dimension])
        focal_plane_array = np.zeros(focal_plane_dimension)
        focal_plane_delta = parameters["detector"]["delta_pix"] / oversampling

        if parameters["type"].lower() == "spectrometer":
            # get wavelength solution
            wl_solution_task = find_task(
                parameters["wl_solution"]["wl_solution_task"],
                instrument.LoadWavelengthSolution,
            )
            wl_instance = wl_solution_task()
            wl_solution = wl_instance(parameters=parameters)

            # estimate spectral dispersion law
            spectral_pix_osr = (
                np.arange(focal_plane_array.shape[1]) * focal_plane_delta
            ).to(u.um)
            spectral_wav_osr = self._wav_osr(
                wl_solution, "spectral", parameters, spectral_pix_osr
            )

            # estimate spatial dispersion law
            spatial_pix_osr = (
                np.arange(focal_plane_array.shape[0]) * focal_plane_delta
            ).to(u.um)
            spatial_wav_osr = self._wav_osr(
                wl_solution, "spatial", parameters, spatial_pix_osr
            )

        elif parameters["type"].lower() == "photometer":
            efficiency = self.get_task_param("efficiency")

            # if we select efficiency > eff_max/e we underestimate the
            # total flux excluding to many efficiency data
            idx = np.where(
                efficiency.data[0, 0] > 0
            )  # efficiency.data[0, 0].max() / np.e)
            x_wav_osr = (
                np.linspace(
                    efficiency.spectral[idx].min().item(),
                    efficiency.spectral[idx].max().item(),
                    32 * oversampling,
                )
                * efficiency.spectral_units
            )
            # 32 is the minimum number of data we want to compute the
            # derivative. For less data points we have a wavelength
            # dependent effect if photometers efficiency

            #            x_wav_center = (efficiency.spectral[idx] *
            #                            efficiency.data[0, 0, idx]).sum() / \
            #                           efficiency.data[0, 0, idx].sum()

            spectral_wav_osr = x_wav_osr
            spatial_wav_osr = x_wav_osr

        focal_plane_array = Signal(
            spectral=spectral_wav_osr,
            spatial=spatial_wav_osr,
            data=focal_plane_array,
            metadata={
                "focal_plane_delta": focal_plane_delta,
                "oversampling": oversampling,
            },
        )
        try:
            focal_plane_array.metadata["wl_min"] = parameters["wl_min"]
            focal_plane_array.metadata["wl_max"] = parameters["wl_max"]
        except KeyError:
            pass

        self.set_output(focal_plane_array)

        del focal_plane_array
        gc.collect()

    def _wav_osr(self, wl_solution, key, parameters, pix_osr):
        if wl_solution[key].data == np.zeros_like(wl_solution["wavelength"]):
            # wavelength on each x pixel
            wav_osr = np.zeros(pix_osr.size) * u.um
        else:
            # estimate dispersion law
            par = np.polyfit(
                wl_solution[key].to(u.um).value,
                wl_solution["wavelength"].to(u.um).value,
                2,
            )
            # Compute dispersion law (pixel -> wavelength) by fitting
            # `wl_solution[key]` (pixel positions) -> `wl_solution["wavelength"]`.
            # To obtain the inverse mapping (wavelength -> pixel) we solve
            # the polynomial equation p(pixel) = wavelength when needed.
            par = np.polyfit(
                wl_solution[key].to(u.um).value,
                wl_solution["wavelength"].to(u.um).value,
                2,
            )

            # Compute the spectral dispersion law
            spectral_dispersion_law = np.poly1d(par)

            # estimate center
            if "center" in parameters["wl_solution"]:
                pix_osr = self._centering(parameters, wl_solution, pix_osr, key)

            # walength on each x pixel
            wav_osr = spectral_dispersion_law(pix_osr.to(u.um).value) * u.um
        self.debug(f"{key} wavelength solution: {wav_osr}")
        return wav_osr

    def _centering(self, parameters, wl_solution, spectral_pix_osr, key="spectral"):
        """
        Shift the pixel array. If "auto" it sets the central wavelength of
        the channel in the center of the pixel array. If a wavelength is indicated,
        it centers the wl solution on that wavelength.
        Else, it shifts the pixel array by the indicated number of pixels.

        Parameters
        ----------
        parameters : dict
            Channel parameters dictionary
        wl_solution : dict
            Wavelength solution data
        spectral_pix_osr : astropy.units.Quantity
            Pixel array to be centered
        key : str, optional
            Spectral or spatial key (default: "spectral")

        Returns
        -------
        astropy.units.Quantity
            Centered pixel array
        """

        # Get polynomial degree from config or use default
        poly_degree = parameters["wl_solution"].get("poly_degree", 2)

        if "center" in parameters["wl_solution"]:
            # Compute dispersion law (pixel -> wavelength) by fitting
            # `wl_solution[key]` (pixel positions) -> `wl_solution["wavelength"]`.
            # To obtain the inverse mapping (wavelength -> pixel) we solve
            # the polynomial equation p(pixel) = wavelength when needed.
            par = np.polyfit(
                wl_solution[key].to(u.um).value,
                wl_solution["wavelength"].to(u.um).value,
                poly_degree,
            )
            # Pixel array center position
            array_center = (spectral_pix_osr[0] + spectral_pix_osr[-1]) / 2

            if parameters["wl_solution"]["center"] == "auto":
                # Auto mode: center on the midpoint of wl_min and wl_max
                # Solve p(pixel) = wl_min and p(pixel) = wl_max for pixel positions.
                wl_min_val = parameters["wl_min"].to(u.um).value
                wl_max_val = parameters["wl_max"].to(u.um).value

                coeffs_min = par.copy()
                coeffs_min[-1] -= wl_min_val
                roots_min = np.roots(coeffs_min)
                real_min = roots_min[np.isreal(roots_min)].real
                pix_min_val = real_min if real_min.size else roots_min.real

                coeffs_max = par.copy()
                coeffs_max[-1] -= wl_max_val
                roots_max = np.roots(coeffs_max)
                real_max = roots_max[np.isreal(roots_max)].real
                pix_max_val = real_max if real_max.size else roots_max.real

                # Choose the root closest to the array center in each case
                center_val = array_center.to(u.um).value
                first_pixel_val = pix_min_val[
                    np.argmin(np.abs(pix_min_val - center_val))
                ]
                last_pixel_val = pix_max_val[
                    np.argmin(np.abs(pix_max_val - center_val))
                ]

                first_pixel = first_pixel_val * u.um
                last_pixel = last_pixel_val * u.um
                target_pixel = (first_pixel + last_pixel) / 2
                offset = array_center - target_pixel

                self.debug(
                    "wl solution auto-centering mode: centering on wavelength range midpoint. "
                    f"offset = {offset}"
                )

            elif isinstance(parameters["wl_solution"]["center"], u.Quantity):
                # Wavelength mode: center on specific wavelength
                center_wl = parameters["wl_solution"]["center"].to(u.um).value

                # Validate wavelength is in valid range
                wl_min = wl_solution["wavelength"].min().to(u.um).value
                wl_max = wl_solution["wavelength"].max().to(u.um).value

                if not (wl_min <= center_wl <= wl_max):
                    self.warning(
                        f"Centering wavelength {center_wl} um is outside the "
                        f"wavelength solution range [{wl_min}, {wl_max}] um. "
                        "Extrapolation may be inaccurate."
                    )

                # Solve p(pixel) = center_wl for pixel position
                coeffs = par.copy()
                coeffs[-1] -= center_wl
                roots = np.roots(coeffs)
                real_roots = roots[np.isreal(roots)].real
                candidates = real_roots if real_roots.size else roots.real
                center_val = array_center.to(u.um).value
                chosen_val = candidates[np.argmin(np.abs(candidates - center_val))]
                target_pixel = chosen_val * u.um
                offset = array_center - target_pixel

                self.debug(
                    f"wl solution wavelength-centering mode: centering on {center_wl} um. "
                    f"offset = {offset}"
                )
            else:
                # Invalid value for center parameter
                self.error(
                    f"Invalid value for 'center' parameter: {parameters['wl_solution']['center']}. "
                    "Expected 'auto' or astropy Quantity."
                )
                return spectral_pix_osr

            spectral_pix_osr -= offset

        else:
            # Manual mode: apply numeric offset directly
            center = parameters["wl_solution"][f"{key}_center"]
            self.debug(f"wl solution manual-centering mode: applying offset {center}")
            spectral_pix_osr -= center

        return spectral_pix_osr
