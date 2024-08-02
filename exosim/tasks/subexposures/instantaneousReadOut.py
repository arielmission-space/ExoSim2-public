import copy
from copy import deepcopy

import astropy.units as u
import numpy as np
from numba import jit
from numba import prange
from scipy.interpolate import RectBivariateSpline

from exosim.models.signal import Counts
from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.operations import operate_over_axis


class InstantaneousReadOut(Task):
    """
    This task implements the instantaneous read out.
    It loads the readout configuration from a dictionary that is produced by :class:`~exosim.tasks.subexposures.prepareInstantaneousReadOut.PrepareInstantaneousReadOut`.
    Then it creates the sub-exposure datacube, with sub-exposure for each NDRs in the ramp sampling scheme.
    Each of these sub-exposures collects more simulation steps, which have their own jitter offset.
    This Task iterates over the time steps to jitter the focal plane and pile it to the appropriate sub-exposure.
    The jittering is based on the focal plane oversampling factor.
    The contributions to each sub-exposures are averaged and the final product is multiplied by its integration time.

    Returns
    --------
    :class:`~exosim.models.signal.Counts`
        sub-exposure cached signal class
    """

    def __init__(self):
        """
        Parameters
        ----------
        readout_parameters: dict
            readout_parameters dict
        focal_plane: :class:`~exosim.models.signal.CountsPerSecond`
            channel focal plane
        pointing_jitter: (:class:`~astropy.units.Quantity`, :class:`~astropy.units.Quantity`,  :class:`~astropy.units.Quantity`)
            Tuple containing the pointing jitter in the spatial and spectral direction expressed in units of deg, adn jitter time expressed as sec.
        parameters: dict
            dictionary containing the channel parameters.
            This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        output: str or :class:`~exosim.output.hdf5.hdf5.HDF5Output` or :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`
            output file
        dataset_name: str (optional)
            dataset name. Default is "SubExposures".
        """
        self.add_task_param("parameters", "channel parameters dict")
        self.add_task_param("readout_parameters", "channel parameters dict")
        self.add_task_param("focal_plane", "loaded focal plane")
        self.add_task_param("pointing_jitter", "")
        self.add_task_param("output_file", "output file")
        self.add_task_param("dataset_name", "dataset name", "SubExposures")
        self.add_task_param(
            "slicing",
            "jittering by slice, avoid to create a large cube in memory",
            False,
        )
        self.store_dict = {}

    def execute(self):
        parameters = self.get_task_param("parameters")
        readout_parameters = self.get_task_param("readout_parameters")
        pointing_jitter = self.get_task_param("pointing_jitter")
        focal_plane = self.get_task_param("focal_plane")
        output_file = self.get_task_param("output_file")
        dataset_name = self.get_task_param("dataset_name")

        focal = copy.deepcopy(focal_plane.data.astype(np.float64))
        base_osf = focal_plane.metadata["oversampling"]

        ndr_integration_times = readout_parameters["ndr_integration_times"]
        clock = readout_parameters["simulation_clock"]
        fp_time = readout_parameters["fp_time"]
        start_index = readout_parameters[
            "ndr_start_cumulative_sequence"
        ].astype(int)
        end_index = readout_parameters["ndr_end_cumulative_sequence"].astype(
            int
        )
        saveMemory = self.get_task_param("slicing")
        # saveMemory=True
        out = Counts(
            spectral=focal_plane.spectral[int(base_osf // 2) :: base_osf]
            * focal_plane.spectral_units,
            time=(start_index * clock).to(u.hr),
            data=None,
            spatial=focal_plane.spatial[int(base_osf // 2) :: base_osf]
            * focal_plane.spatial_units,
            shape=(
                end_index.shape[0],
                focal.shape[1] // base_osf,
                focal.shape[2] // base_osf,
            ),
            cached=True,
            output=output_file,
            dataset_name=dataset_name,
            output_path=None,
            metadata={"integration_times": ndr_integration_times},
            dtype=np.float64,
        )
        out.metadata["focal_plane_time_indexes"] = fp_time

        if pointing_jitter != (None, None, None):
            self.debug("Pointing jitter found")

            # if jitter is enabled, the following key are available
            mag = readout_parameters["mag"]
            osf = base_osf * mag
            y_jit = (readout_parameters["y_jit"] * osf / u.pix).value
            x_jit = (readout_parameters["x_jit"] * osf / u.pix).value

            y_jit = y_jit.astype(int)
            x_jit = x_jit.astype(int)

            if saveMemory:
                """
                debug=False
                if debug:
                    new_focal = [self.oversample(fp, mag) for fp in focal]
                """

                if mag != 1:
                    xin, yin, xout, yout = self.getOversampleFactors(
                        focal[0, ...], mag
                    )

                    yshape = int(yout.shape[0] // osf)
                    xshape = int(xout.shape[0] // osf)

                    # time_line = np.zeros((start_index.shape[0], int(yout.shape[0]  // osf),
                    # int( xout.shape[0]// osf)),dtype=np.float64,)

                else:
                    yshape = int(focal.shape[1] // osf)
                    xshape = int(focal.shape[2] // osf)

                self.info(
                    "jittering {} for {}".format(
                        dataset_name, parameters["value"]
                    )
                )

                for chunk in iterate_over_chunks(
                    out.dataset,
                    desc="jittering {}".format(parameters["value"]),
                ):
                    time_line = np.zeros(
                        (start_index[chunk[0]].shape[0], yshape, xshape),
                        dtype=np.float64,
                    )
                    # focal:       (3420, 192, 1068)
                    # chunk          274,  64,  356
                    # time_line     (7856, 64,  356 )
                    # iterate over the timeline sub-exposureA
                    t_cache = -1
                    fp_cache = None
                    time_line_slice = None
                    """
                    ndrs=prange(start_index[chunk[0]].shape[0])
                    message="chunk iteration: ndr %i-%i , t= %i -%i "%(ndrs.start,ndrs.stop-1,fp_time[chunk[0]][ndrs.start], fp_time[chunk[0]][ndrs.stop-1] )      
                    self.info(message)
                    """

                    for ndr in prange(start_index[chunk[0]].shape[0]):
                        # select the focal plane at the right time
                        t = fp_time[chunk[0]][ndr]

                        # print (" ndr=",ndr, "t=",t, end='\r')
                        fp_slice = focal[t, ...]

                        if t == t_cache:
                            fp = fp_cache
                        else:
                            if mag != 1:
                                fp = self.oversample(fp_slice, mag)
                            else:
                                fp = fp_slice
                            t_cache = t
                            fp_cache = fp
                        """    
                        if debug:
                           fp_slice_test= new_focal[t]
                          
                           message="Total difference sliced focal plane ndr=%i t=%i difference =%g"%(ndr,t,np.nansum(fp-fp_slice_test))
                           self.info(message)    
                           print (" fp.shape=", fp.shape)
                        """
                        time_line_slice = self.jittering_the_focalplane_by_slice(
                            fp,
                            osf,
                            start_index[chunk[0]],
                            end_index[chunk[0]],
                            x_jit,  # TODO: check if this is correct: is it correct to use chunks?
                            y_jit,
                            fp_time[chunk[0]],
                            time_line[ndr, ...],
                            ndr,
                        )
                        time_line[ndr, ...] = time_line_slice
                    # dset=   time_line[chunk[0]]

                    out.dataset[chunk] = time_line

                    out.output.flush()
            else:
                # apply jitter magnification
                if mag != 1:
                    self.info(
                        "resampling the focal plane: magnification factor {}".format(
                            mag
                        )
                    )

                    # resampling the focal plane and replace it with a new array
                    new_focal = [self.oversample(fp, mag) for fp in focal]
                    # focal = np.array(new_focal)
                    focal = copy.deepcopy(np.array(new_focal))

                    self.debug(
                        "focal plane resampled: new shape {}".format(
                            focal.shape
                        )
                    )
                self.info(
                    "jittering {} for {}".format(
                        dataset_name, parameters["value"]
                    )
                )

                for chunk in iterate_over_chunks(
                    out.dataset,
                    desc="jittering {}".format(parameters["value"]),
                ):
                    dset = self.jittering_the_focalplane(
                        focal,
                        osf,
                        start_index[chunk[0]],
                        end_index[chunk[0]],
                        x_jit,  # TODO: check if this is correct: is it correct to use chunks?
                        y_jit,
                        fp_time[chunk[0]],
                    )
                    out.dataset[chunk] = dset

                    out.output.flush()

            # Here we force the power conservation, if the user enabled the option
            if "force_power_conservation" in parameters.keys():
                if parameters["force_power_conservation"]:
                    self.warning("forcing power conservation")
                    self.force_power_conservation(
                        out, parameters, focal_plane, fp_time, osf
                    )

            # applying integtation time to the jittered focal planes
            for chunk in iterate_over_chunks(
                out.dataset,
                desc="applying integration time {}".format(
                    parameters["value"]
                ),
            ):
                dset = out.dataset[chunk]
                dset = operate_over_axis(
                    dset, ndr_integration_times[chunk[0]].value, 0, "*"
                )

                out.dataset[chunk] = dset

                out.output.flush()

        else:
            self.info(
                "no jitter in {} for {}".format(
                    dataset_name, parameters["value"]
                )
            )
            focal = deepcopy(focal_plane.data[:, 0::base_osf, 0::base_osf])
            for chunk in out.dataset.iter_chunks():
                dset = self.replicating_the_focalplane(
                    focal, start_index[chunk[0]], fp_time[chunk[0]]
                )
                self.debug("focal plane replicated")

                dset = operate_over_axis(
                    dset, ndr_integration_times[chunk[0]].value, 0, "*"
                )

                out.dataset[chunk] = dset

                out.output.flush()

        self.set_output(out)

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
    def getOversampleFactors(fp, ad_osf):
        """
        Used in oversample method to determine the shape of the arrays .
        Parameters
        ----------
        fp: :class:`~numpy.ndarray`
            2D focal plane
        ad_osf: int
            magnification factor

        Returns
        -------
        :class:`~numpy.ndarray`
            the x and y grids used in the input and output of the oversampling

        """
        xin = np.linspace(0, fp.shape[1] - 1, fp.shape[1])
        yin = np.linspace(0, fp.shape[0] - 1, fp.shape[0])
        x_step = abs(xin[1]) - abs(xin[0])
        y_step = abs(yin[1]) - abs(yin[0])

        # calculates the new step sizes for new grid
        x_step_new = np.float64(x_step / ad_osf)
        y_step_new = np.float64(y_step / ad_osf)

        # new grid must start with an exact offset to produce correct number of new points
        x_start = -x_step_new * np.float64((ad_osf - 1) / 2)
        y_start = -y_step_new * np.float64((ad_osf - 1) / 2)

        # new grid points- with correct start, end and spacing
        xout = np.arange(
            x_start, x_start + x_step_new * fp.shape[1] * ad_osf, x_step_new
        )
        yout = np.arange(
            y_start, y_start + y_step_new * fp.shape[0] * ad_osf, y_step_new
        )
        return xin, yin, xout, yout

    @staticmethod
    def oversample(fp: np.array, ad_osf: int) -> np.array:
        """
        It increases the oversampling factor of the focal plane.

        Parameters
        ----------
        fp: :class:`~numpy.ndarray`
            2D focal plane
        ad_osf: int
            magnification factor

        Returns
        -------
        :class:`~numpy.ndarray`
            2D focal plane sampled with the new oversampling factor
        """
        xin, yin, xout, yout = InstantaneousReadOut.getOversampleFactors(
            fp, ad_osf
        )

        # interpolate fp onto new grid
        fn = RectBivariateSpline(yin, xin, fp)
        new_fp = fn(yout, xout)

        return new_fp

    @staticmethod
    @jit(nopython=True, parallel=True)
    def jittering_the_focalplane_by_slice(
        fp: np.array,
        osf: int,
        start_index: np.array,
        end_index: np.array,
        x_jit: np.array,
        y_jit: np.array,
        fp_time: np.array,
        time_line_slice: np.array,
        ndr: int,
    ) -> np.array:
        """
        Same as jittering_the_focalplane but to operate only on a single slice fp
        """
        # iterate over the timeline sub-exposures
        j = int(osf // 2)  # starting index for spatial direction
        # iterate over the spatial direction
        for y in range(time_line_slice.shape[0]):
            i = int(osf // 2)  # starting index for spectral direction
            # iterate over the spectral direction
            for x in range(time_line_slice.shape[1]):
                # iterate over the jitter indices
                for idx in range(start_index[ndr], end_index[ndr]):
                    # selecting the jitter offset indices
                    j_jit = y_jit[idx] + j
                    i_jit = x_jit[idx] + i

                    # if negative index, then roll the array
                    if i_jit < 0:
                        i_jit = i_jit + fp.shape[1]
                    elif i_jit >= fp.shape[1]:
                        i_jit = i_jit - fp.shape[1]

                    if j_jit < 0:
                        j_jit = j_jit + fp.shape[0]
                    elif j_jit >= fp.shape[0]:
                        j_jit = j_jit - fp.shape[0]

                    # if negative index, then roll the array
                    time_line_slice[y, x] = (
                        time_line_slice[y, x] + fp[j_jit, i_jit]
                    )
                # move to the next pixel in the spectral direction
                i = i + osf
            # move to the next pixel in the spatial direction
            j = j + osf
        # divide by the number of jitter positions added
        time_line_slice = time_line_slice / (end_index[ndr] - start_index[ndr])

        return time_line_slice

    @staticmethod
    @jit(nopython=True, parallel=True)
    def jittering_the_focalplane(
        fp: np.array,
        osf: int,
        start_index: np.array,
        end_index: np.array,
        x_jit: np.array,
        y_jit: np.array,
        fp_time: np.array,
    ) -> np.array:
        # create an empty array to store the jittered focal plane
        time_line = np.zeros(
            (
                start_index.shape[0],
                int(fp.shape[1] // osf),
                int(fp.shape[2] // osf),
            ),
            dtype=np.float64,
        )
        # iterate over the timeline sub-exposures
        for ndr in prange(start_index.shape[0]):
            # select the focal plane at the right time
            t = fp_time[ndr]

            j = int(osf // 2)  # starting index for spatial direction
            # iterate over the spatial direction
            for y in range(time_line.shape[1]):
                i = int(osf // 2)  # starting index for spectral direction
                # iterate over the spectral direction
                for x in range(time_line.shape[2]):
                    # iterate over the jitter indices
                    for idx in range(start_index[ndr], end_index[ndr]):
                        # selecting the jitter offset indices
                        j_jit = y_jit[idx] + j
                        i_jit = x_jit[idx] + i

                        # if negative index, then roll the array
                        if i_jit < 0:
                            i_jit = i_jit + fp.shape[2]
                        elif i_jit >= fp.shape[2]:
                            i_jit = i_jit - fp.shape[2]

                        if j_jit < 0:
                            j_jit = j_jit + fp.shape[1]
                        elif j_jit >= fp.shape[1]:
                            j_jit = j_jit - fp.shape[1]

                        # if negative index, then roll the array
                        time_line[ndr, y, x] = (
                            time_line[ndr, y, x] + fp[t, j_jit, i_jit]
                        )
                    # move to the next pixel in the spectral direction
                    i = i + osf
                # move to the next pixel in the spatial direction
                j = j + osf
            # divide by the number of jitter positions added
            time_line[ndr] = time_line[ndr] / (
                end_index[ndr] - start_index[ndr]
            )
        return time_line

    @staticmethod
    @jit(nopython=True, parallel=True)
    def replicating_the_focalplane(fp, index, fp_time):
        time_line = np.zeros(
            (index.shape[0], int(fp.shape[1]), int(fp.shape[2])),
            dtype=np.float64,
        )
        for ndr in prange(index.shape[0]):
            t = fp_time[ndr]
            time_line[ndr, :, :] = fp[t, :, :]

        return time_line
