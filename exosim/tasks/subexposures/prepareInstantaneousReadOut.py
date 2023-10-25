import copy
from copy import deepcopy

import astropy.units as u
import numpy as np
from numba import jit
from numba import prange
from scipy.interpolate import RectBivariateSpline

import exosim.output as output
from exosim.models.signal import Counts
from exosim.tasks.subexposures.computeReadingScheme import ComputeReadingScheme
from exosim.tasks.subexposures.estimateChJitter import EstimateChJitter
from exosim.tasks.task import Task
from exosim.utils.checks import check_units
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.iterators import searchsorted
from exosim.utils.operations import operate_over_axis


class PrepareInstantaneousReadOut(Task):
    """
    This task prepares the instantaneous read out.
    It calls  :class:`~exosim.tasks.subexposures.computeReadingScheme.ComputeReadingScheme` to compute the ramp sampling scheme,
    and :class:`~exosim.tasks.subexposures.estimateChJitter.EstimateChJitter` to scale the input pointing jitter to the focal planet pixel units.
    The jittering is based on the focal plane oversampling factor.

    Returns
    --------
    dict
        readout_parameters dict
    :class:`~astropy.units.Quantity`
        sub-exposures integration times

    """

    def __init__(self):
        """
        Parameters
        ----------
        main_parameters: dict
            main parameters dict
        focal_plane: :class:`~exosim.models.signal.CountsPerSecond`
            channel focal plane
        pointing_jitter: (:class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`,  :class:`~astropy.units.Quantity`)
            Tuple containing the pointing jitter in the spatial and spectral direction expressed in units of deg, adn jitter time expressed as sec.
        parameters: dict
            dictionary containing the channel parameters.
            This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: str or :class:`~exosim.output.hdf5.hdf5.HDF5Output` or :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`
            output file
        """
        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("main_parameters", "channel parameters dict")
        self.add_task_param("focal_plane", "loaded focal plane")
        self.add_task_param("pointing_jitter", "")
        self.add_task_param("output_file", "output file")
        self.store_dict = {}

    def execute(self):
        parameters = self.get_task_param("parameters")
        main_parameters = self.get_task_param("main_parameters")
        pointing_jitter = self.get_task_param("pointing_jitter")
        focal_plane = self.get_task_param("focal_plane")
        output_file = self.get_task_param("output_file")

        osf = focal_plane.metadata["oversampling"]

        if pointing_jitter != (None, None, None):
            estimateChJitter = EstimateChJitter()
            jitter_spe, jitter_spa, y_jit, x_jit, jit_time = estimateChJitter(
                pointing_jitter=pointing_jitter, parameters=parameters
            )

            new_freq = check_units(jit_time[1] - jit_time[0], "Hz")
            readout_freq = check_units(
                parameters["readout"]["readout_frequency"], "Hz"
            )
            readout_oversampling = new_freq / readout_freq
        else:
            readout_oversampling = 1

        computeReadingScheme = ComputeReadingScheme()
        (
            clock,
            base_mask,
            frame_sequence,
            number_of_exposures,
        ) = computeReadingScheme(
            parameters=parameters,
            main_parameters=main_parameters,
            readout_oversampling=readout_oversampling,
            output_file=output_file,
        )

        # Number of frames contributing to each NDR
        ndr_end_cumulative_sequence = np.ma.array(
            frame_sequence.cumsum(),
            mask=np.tile(np.logical_not(base_mask), number_of_exposures),
        ).compressed()

        base_mask_start = base_mask
        base_mask_start[0] = 1
        base_mask_start[-2] = 0

        ndr_start_cumulative_sequence = np.ma.array(
            frame_sequence.cumsum(),
            mask=np.tile(np.logical_not(base_mask_start), number_of_exposures),
        ).compressed()

        ndr_integration_times = (
            ndr_end_cumulative_sequence - ndr_start_cumulative_sequence
        ) * clock

        focal_time_sampled = (focal_plane.time * focal_plane.time_units).to(
            u.s
        )
        fp_time = searchsorted(
            focal_time_sampled, ndr_start_cumulative_sequence * clock
        )

        self.store_dict = {
            "number_of_exposures": number_of_exposures,
            "frame_sequence": frame_sequence,
            "ndr_end_cumulative_sequence": ndr_end_cumulative_sequence,
            "ndr_start_cumulative_sequence": ndr_start_cumulative_sequence,
            "ndr_integration_times": ndr_integration_times,
            "simulation_clock": clock,
            "fp_time": fp_time,
        }

        if pointing_jitter != (None, None, None):
            self.debug("Pointing jitter found")

            # check if the jitter is well sampled

            # take min rms if min is not zero
            jitter_rms_array = np.array(
                [np.sqrt(np.mean(y_jit**2)), np.sqrt(np.mean(x_jit**2))]
            )

            try:
                jitter_rms = min(
                    jitter_rms_array[np.nonzero(jitter_rms_array)]
                )
            except ValueError:
                self.debug("jitter rms is zero in both directions")
                jitter_rms = 0
            self.debug("jitter rms: {:.2f}".format(jitter_rms))

            # check if specific resolution is required else set it to 3 by default
            try:
                jitter_res = parameters["detector"][
                    "jitter_rms_min_resolution"
                ]
                self.debug("found jitter rms minimum resolution")
            except KeyError:
                jitter_res = 3.0

            # check if magnification is wanted or needed
            mag = 1
            # check if magnification is needed
            if jitter_rms != 0 and jitter_rms < jitter_res:
                self.debug("jitter rms < 3: focal plane resampling needed")
                mag = np.ceil(jitter_res / jitter_rms).astype(int)

            # check if magnification is suggested
            try:
                forced_mag = parameters["detector"]["jitter_resampler_mag"]
                self.debug("found jitter resampler magnification")
            except KeyError:
                forced_mag = None
            # if magnification is suggested choose between the suggested and the computed
            if forced_mag:
                if forced_mag < mag:
                    self.warning(
                        "suggested jitter magnification ({}) is too small: {} used instead".format(
                            forced_mag, mag
                        )
                    )
                else:
                    mag = forced_mag

            # producing diagnostic info
            spe_jit_ave = (
                self._average_pointing(
                    jitter_spe.value,
                    ndr_start_cumulative_sequence,
                    ndr_end_cumulative_sequence,
                )
                * jitter_spe.unit
            )
            spa_jit_ave = (
                self._average_pointing(
                    jitter_spa.value,
                    ndr_start_cumulative_sequence,
                    ndr_end_cumulative_sequence,
                )
                * jitter_spa.unit
            )
            x_jit_ave = (
                self._average_pointing(
                    x_jit,
                    ndr_start_cumulative_sequence,
                    ndr_end_cumulative_sequence,
                )
                / osf
                * u.pix
            )
            y_jit_ave = (
                self._average_pointing(
                    y_jit,
                    ndr_start_cumulative_sequence,
                    ndr_end_cumulative_sequence,
                )
                / osf
                * u.pix
            )
            jit_indexes = []
            for start, stop in zip(
                ndr_start_cumulative_sequence, ndr_end_cumulative_sequence
            ):
                jit_indexes.append(np.arange(start, stop).astype(int))
            self.store_dict.update(
                {
                    "y_jit": y_jit / osf * u.pix,
                    "x_jit": x_jit / osf * u.pix,
                    "y_jit_averaged": y_jit_ave,
                    "x_jit_averaged": x_jit_ave,
                    "spe_jit_averaged": spe_jit_ave,
                    "spa_jit_averaged": spa_jit_ave,
                    "jit_indexes": {"se": jit_indexes},
                    "mag": mag,
                    "effective_osf": osf * mag,
                }
            )

        for k, v in self.store_dict.items():
            self.debug("{}: {}".format(k, v))

        if issubclass(output_file.__class__, output.Output):
            output_file.store_dictionary(
                self.store_dict, "instantaneous_readout_params"
            )

        self.set_output([self.store_dict, ndr_integration_times])

    def force_power_conservation(
        self, out, parameters, focal_plane, fp_time, osf
    ):
        # to compute the total power on the focal plane I use the undersampled focal plane

        total_power = np.empty(out.dataset.shape[0])
        desired_power = np.empty(out.dataset.shape[0])

        for chunk in iterate_over_chunks(
            out.dataset,
            desc="computing median incoming power {}".format(
                parameters["value"]
            ),
        ):
            # computing the total power in the jittered focal plane
            dset = out.dataset[chunk]
            total_power[chunk[0]] = dset.sum(axis=-1).sum(axis=-1)

            # computing the desired power from the original focal planes
            fp_time_ = fp_time[chunk[0]]
            fp_times = list(set(fp_time_))
            for time_id in fp_times:
                mask = np.where(fp_time_ == time_id)[0]
                # I estimated the expected power from the oversampled focal plane
                desired_power[chunk[0]][mask] = (
                    np.sum(focal_plane.data[time_id]) / osf**2
                )

        # applying integration time to the jittered focal planes
        for chunk in iterate_over_chunks(
            out.dataset,
            desc="forcing conservation of power {}".format(
                parameters["value"]
            ),
        ):
            dset = out.dataset[chunk]
            dset = operate_over_axis(
                dset, desired_power[chunk[0]] / total_power[chunk[0]], 0, "*"
            )

            out.dataset[chunk] = dset

            out.output.flush()

        self.store_dict.update(
            {"median_power": desired_power, "total_power": total_power}
        )

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _average_pointing(jitter, start, stop):
        out = np.zeros(start.shape)
        for i in range(start.size):
            # print(start[i], stop[i])
            for j in range(start[i], stop[i], 1):
                # print(i, out[i], jitter[j])
                out[i] += jitter[j]
            out[i] /= stop[i] - start[i]
        return out
