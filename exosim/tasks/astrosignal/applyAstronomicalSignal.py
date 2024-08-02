from copy import deepcopy

import astropy.units as u
import numpy as np
from numba import jit
from numba import prange
from scipy.signal import convolve

import exosim.tasks.instrument as instrument
from exosim.tasks.instrument import CreateIntrapixelResponseFunction
from exosim.tasks.task import Task
from exosim.utils.focal_plane_locations import locate_wavelength_windows
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.types import ArrayType


class ApplyAstronomicalSignal(Task):
    """
    This task applies the astronomical signal to the sub-exposures.
    To do so, it first convolve the astronomical signal with the instrument line shape (ILS)
    on the focal plane. This is done by populating the focal plane with the ILS and then weighting the contributions to each pixel.
    The resulting model is then convolved with the intrapixel response function (IPRF) and downsampled to the sub-exposure time resolution.
    Then is finally multiplied to the sub-exposure signal.

    Returns
    --------
    :class:`~exosim.models.signal.Counts`
        sub-exposure cached signal class

    """

    def __init__(self) -> None:
        """
        Parameters
        ------------
        source: dict
            source parameters
        model: :class:`np.ndarray`
            model of the astronomical signal in a 2D array (wavelength, time)
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal class
        focal_plane: :class:`~exosim.models.signal.CountsPerSecond`
            channel focal plane
        ils: :class:`np.ndarray`
            instrument line shape. The shape is (time, wavelength, spectral direction)
        timeline : :class:`astropy.units.Quantity`
            timeline to compute the signal
        ch_parameters : dict
            channel parameters
        pointing: (:class:`astropy.units.Quantity`, :class:`astropy.units.Quantity`) (optional)
            telescope pointing direction, expressed ad a tuple of RA and DEC in degrees. Default is ``None``
        """

        self.add_task_param("source", "source")
        self.add_task_param("model", "astronomical signal model")
        self.add_task_param("subexposures", "subexposures")
        self.add_task_param("focal_plane", "focal_plane")
        self.add_task_param("ils", "instrument line shape")
        self.add_task_param("timeline", "timeline")
        self.add_task_param("ch_parameters", "channel parameters")
        self.add_task_param("pointing", "telescope pointing", None)

    def execute(self):
        source = self.get_task_param("source")
        model = self.get_task_param("model")
        subexposures = self.get_task_param("subexposures")
        focal_plane = self.get_task_param("focal_plane")
        ils = self.get_task_param("ils")
        timeline = self.get_task_param("timeline")
        ch_parameters = self.get_task_param("ch_parameters")
        pointing = self.get_task_param("pointing")

        integration_times = subexposures.metadata["integration_times"]
        # estimate Subexposure midtimes
        se_time = (
            subexposures.time * subexposures.time_units + integration_times / 2
        )
        self.debug("se mid-times: {}".format(se_time))

        # check if astro timeline is empty
        if timeline.size == 0:
            self.set_output(subexposures)
            self.debug("timeline is empty")
            return
        # extract the timeline starting point from the model
        timeline = timeline.to(se_time.unit)
        start_t = (np.abs(se_time - timeline[0])).argmin()
        end_t = (np.abs(se_time - timeline[-1])).argmin()

        self.debug("model start: {}. model end: {}".format(start_t, end_t))

        # preparing the intrapixel response function
        iprf = self._bild_intrapixel(ch_parameters)
        iprf /= iprf.sum()

        # locate the instrument line shape on the focal plane relative to the source
        _, j0 = locate_wavelength_windows(ils, focal_plane, ch_parameters)
        compute_offset = instrument.ComputeSourcesPointingOffset()
        offset_spectral, _ = compute_offset(
            source=source,
            pointing=pointing,
            parameters=ch_parameters,
        )
        j0 += offset_spectral

        # temporal index of the time dependent ILS
        index = subexposures.metadata["focal_plane_time_indexes"]
        id_start = 0  # index for chunking the model
        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="applying astronomical signal"
        ):
            chunk = list(chunk)  # workaround to replace the tuple first value
            chunk[0] = self.select_chunk_range(chunk[0], start_t, end_t)
            if chunk[0] is not None:
                chunk = tuple(chunk)
                id_stop = id_start + chunk[0].stop - chunk[0].start

                multiplicative_signal, norm = populate(
                    deepcopy(j0),
                    ils,
                    index[chunk[0].start : chunk[0].stop],
                    focal_plane.data.shape[2],
                    model[:, id_start:id_stop],
                )

                # normalise ILS and remove zeroes
                idx = np.where(norm != 0)
                multiplicative_signal[idx] /= norm[idx]
                idx = np.where(norm == 0)
                multiplicative_signal[idx] += 1

                # convolve the model with the intrapixel response function
                for t in range(multiplicative_signal.shape[0]):
                    multiplicative_signal[t] = convolve(
                        multiplicative_signal[t], iprf, mode="same"
                    )
                # downsample the model to the subexposure time resolution
                multiplicative_signal = multiplicative_signal[
                    :, iprf.shape[0] // 2 :: iprf.shape[0]
                ]
                # multiply the model with the subexposure
                subexposures.dataset[chunk] *= multiplicative_signal[
                    :, np.newaxis, :
                ]

                subexposures.output.flush()
                id_start = id_stop
                # asd

        self.set_output(subexposures)

    def select_chunk_range(
        self, chunk: slice, start_t: int, end_t: int
    ) -> slice:
        """
        Selects and adjusts the range of the chunk to be processed based on
        the given start and end times.

        Parameters
        ----------
        chunk : slice
            The original slice object representing the chunk to be processed.
        start_t : int
            The start time to consider for processing.
        end_t : int
            The end time to consider for processing.

        Returns
        -------
        slice
            The adjusted slice object representing the new chunk to be processed.
            Returns None if the chunk is entirely outside the start_t and end_t range.

        Examples
        --------
        >>> select_chunk_range(slice(5, 15, 1), 7, 13)
        slice(7, 13, 1)
        >>> select_chunk_range(slice(5, 15, 1), 16, 20)
        None
        """

        # Check if the chunk intersects with the start time and adjust accordingly
        if start_t > chunk.start and start_t < chunk.stop:
            chunk = slice(start_t, chunk.stop, chunk.step)

        # Check if the chunk intersects with the end time and adjust accordingly
        if end_t < chunk.stop and end_t > chunk.start:
            chunk = slice(chunk.start, end_t, chunk.step)

        # If the chunk is entirely outside the range [start_t, end_t], return None
        if chunk.stop < start_t or chunk.start > end_t:
            return None

        return chunk

    def _bild_intrapixel(self, ch_parameters: dict) -> ArrayType:
        createIntrapixelResponseFunction = CreateIntrapixelResponseFunction()
        iprf, _ = createIntrapixelResponseFunction(parameters=ch_parameters)
        iprf = np.array(iprf)
        # central slice
        # TODO what if the iprf is not odd?
        # TODO what if the wl solution is curved?
        iprf = iprf[iprf.shape[0] // 2, :]
        return iprf


@jit(nopython=True, parallel=True)
def populate(
    j0: np.array,
    psf: np.array,
    psf_index: np.array,
    shape_spectral: int,
    model: np.array,
) -> tuple([np.array, np.array]):
    """creates the variation of the source signal on the focal plane"""
    out = np.zeros((model.shape[1], shape_spectral))
    norm = np.zeros((model.shape[1], shape_spectral))

    for k in range(len(j0)):
        # set the expected end of the psf
        j1 = j0[k] + psf.shape[2]

        # psf starting index
        j_start = 0
        # check that teh expected start is not out of the focal plane
        if j0[k] < 0:
            j_start = -j0[k]
            j0[k] = 0

        # set the actual end of the psf
        j_stop = j_start + psf.shape[2]

        # check that the expected end is not out of the focal plane
        if j1 > out.shape[1]:
            j_stop = j_start + out.shape[1] - j0[k]
            j1 = out.shape[1]

        for t in prange(out.shape[0]):
            psf_index_t = psf_index[t]
            if psf_index.shape[0] == 1:
                psf_index_t = 0
            out[t, j0[k] : j1] += (
                psf[psf_index_t, k, j_start:j_stop] * model[k, t]
            )
            norm[t, j0[k] : j1] += psf[psf_index[t], k, j_start:j_stop]

    return out, norm
