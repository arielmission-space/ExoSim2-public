import os

import astropy.units as u
import h5py
import numpy as np

from exosim.tasks.task import Task
from exosim.utils.checks import check_units
from exosim.utils.types import ArrayType


class LoadILS(Task):
    """
    This Task loads the instrument line shapes (ILS) from the input file.
    The loaded ILS are then used to convolve the astronomical signal.
    The ILS here are intended as the PSF of the instrument.
    The ILS shapes are normalized to the maximum value.

    Returns
    -------
    :class:`numpy.ndarray`
        instrument line shape.
        The shape is (time, wavelength, spectral direction)

    Note
    -----
    The instrument line shapes produced by this task are not the same as the instrument line shapes
    as defined in the literature.
    The ILS produced by this task are the PSF of the instrument.
    To be used as the instrument line shapes as defined in the literature they need to be convolved with the intra-pixel response.
    This convolution is not part of this Task as it affects the way the ILS are sampled.
    The convolution with the intra-pixel response is done in the :class:`exosim.tasks.astrosignal.applyAstronomicalSignal.ApplyAstronomicalSignal` Task,
    where the ILS are used to convolve the astronomical signal.
    """

    def __init__(self):
        """
        Parameters
        -----------
        input_file: str
            focal plane input file
        ch_param: dict
            channel parameter
        wl_grid: :class:`astropy.units.Quantity`
            output wavelength grid
        """

        self.add_task_param("input_file", "focal plane input file")
        self.add_task_param("ch_param", "channel parameter")
        self.add_task_param("wl_grid", "wavelength grid")

    def execute(self):
        self.info("loading the instrument line shapes")
        parameters = self.get_task_param("ch_param")
        input_file = self.get_task_param("input_file")
        wl_grid = self.get_task_param("wl_grid")
        psf = self.model(input_file, parameters, wl_grid)
        self.set_output(psf)

    def model(
        self,
        input_file: str,
        parameters: dict,
        wl_grid: ArrayType,
    ) -> ArrayType:
        """
        It loads the channel instrument line shapes from the input file.

        Parameters
        ----------
        input_file: str
            focal plane input file
        ch_param: dict
            channel parameter
        wl_grid: :class:`astropy.units.Quantity`
            output wavelength grid

        Returns
        -------
        :class:`numpy.ndarray`
            instrument line shape.
            The shape is (time, wavelength, spectral direction)
        """
        ch_name = parameters["value"]

        with h5py.File(input_file, "r") as f:
            file_path = os.path.join("channels", ch_name)
            psf_path = os.path.join(file_path, "psf")
            psf_data = os.path.join(psf_path, "psf_cube")
            psf_wl = os.path.join(psf_path, "wavelength")

            psf = f[psf_data][()]
            sampled_wl = f[psf_wl]["value"][()] * u.Unit(f[psf_wl]["unit"][()])

        # select the wavelength range
        sampled_wl = check_units(sampled_wl, wl_grid.unit, self)
        sorter = np.argsort(sampled_wl)
        idx = np.searchsorted(sampled_wl.value, wl_grid.value, sorter=sorter)
        idx = idx[sorter]
        psf = psf[:, idx, :, :]
        # get the central slice
        central_spat = psf.shape[2] // 2
        # extract the central slice
        psf = psf[:, :, central_spat, :]
        # TODO what if the wl solution is curved? We need to interpolate
        # normalize the psf to the maximum
        psf /= psf.max(axis=-1)[:, :, np.newaxis]

        return psf
