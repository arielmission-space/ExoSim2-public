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
            channel parameter dictionary. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
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
        self.debug("focal plane dimensions: {}".format(focal_plane_dimension))

        if "oversampling" in parameters["detector"]:
            oversampling = parameters["detector"]["oversampling"]
        else:
            oversampling = 1
        self.debug("pixel over sampling factor: {}".format(oversampling))

        focal_plane_dimension = tuple(
            [d * oversampling for d in focal_plane_dimension]
        )
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
            spatial_dispersion_law = np.poly1d(par)

            # estimate center
            if any("center" in k for k in parameters["wl_solution"].keys()):
                pix_osr = self._centering(
                    parameters, wl_solution, pix_osr, key
                )

            # walength on each x pixel
            wav_osr = spatial_dispersion_law(pix_osr.to(u.um).value) * u.um
        self.debug("{} wavelength solution: {}".format(key, wav_osr))
        return wav_osr

    def _centering(
        self, parameters, wl_solution, spectral_pix_osr, key="spectral"
    ):
        """
        Shift the pixel array. If "auto" it sets the central wavelength of
        the channel in the center of the pixel array. If a wavelength  is indicated,
        it centers the wl solution on that wavelength.
        Else, it shifts the pixel array by the indicated number of pixels.

        """

        if "center" in parameters["wl_solution"].keys():
            if parameters["wl_solution"]["center"] == "auto":
                par = np.polyfit(
                    wl_solution["wavelength"].to(u.um).value,
                    wl_solution[key].to(u.um).value,
                    2,
                )
                spectral_dispersion_law_inv = np.poly1d(par)

                first_pixel = (
                    spectral_dispersion_law_inv(
                        parameters["wl_min"].to(u.um).value
                    )
                    * u.um
                )
                last_pixel = (
                    spectral_dispersion_law_inv(
                        parameters["wl_max"].to(u.um).value
                    )
                    * u.um
                )
                spectral_pix_osr_center = (first_pixel + last_pixel) / 2
                # spectral_pix_osr_center = spectral_dispersion_law_inv(
                #     central_wl.to(u.um).value) * u.um
                offset = (
                    spectral_pix_osr[0] + spectral_pix_osr[-1]
                ) / 2 - spectral_pix_osr_center
                self.debug(
                    "wl solution auto-centering mode. offset {}".format(offset)
                )
                spectral_pix_osr -= offset
            elif isinstance(parameters["wl_solution"]["center"], u.Quantity):
                par = np.polyfit(
                    wl_solution["wavelength"].to(u.um).value,
                    wl_solution[key].to(u.um).value,
                    2,
                )
                spectral_dispersion_law_inv = np.poly1d(par)
                central_pixel = (
                    spectral_dispersion_law_inv(
                        parameters["wl_solution"]["center"].to(u.um).value
                    )
                    * u.um
                )
                offset = (
                    spectral_pix_osr[0] + spectral_pix_osr[-1]
                ) / 2 - central_pixel
                self.debug(
                    "wl solution auto-centering mode. offset {}".format(offset)
                )
                spectral_pix_osr -= offset
        else:
            center = parameters["wl_solution"]["{}_center".format(key)]
            self.debug(
                "wl solution manual-centering mode. offset {}".format(center)
            )
            spectral_pix_osr -= center
        return spectral_pix_osr
