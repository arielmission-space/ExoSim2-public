import gc
import os
import re
from copy import deepcopy

import astropy.units as u
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

import exosim.tasks.instrument.loadPsf as loadPsf
from exosim.utils.aperture import find_rectangular_aperture


class LoadPsfPaos(loadPsf.LoadPsf):
    """
    It loads the PSFs from a PAOS file.
    Each PAOS file contains a PSF vs wavelength.
    The task loads the PSF cube provided by the `filename` parameter.
    The PSF are then interpolated over a grid matching the one used to produce the focal planes,
    to convert them into the physical units.
    Then the total volume of the interpolated PSF is rescaled to the total volume of the original one.
    This allow to take into account for loss in the transmission due to the optical path.
    The PSF are then interpolated over a wavelength grid matching the one used to for the focal plane, producing the cube.
    Then, the PSF cube is repeated on the temporal axis, because no temporal variation is considered in this Task.

    Returns
    -------
    :class:`~numpy.ndarray`
        cube of psfs.  axis=0 is time, axis=1 is wavelength, axis=2 is spatial direction, axis=3 is spectral direction.
    """

    def model(self, filename, parameters, wavelength, time):
        self.debug("loading PAOS psf")

        if (
            "psf" in parameters
            and "time_dependence" in parameters["psf"].keys()
        ):
            timeDependence = parameters["psf"]["time_dependence"]
        else:
            timeDependence = True
        if "oversampling" in parameters["detector"].keys():
            oversampling = parameters["detector"]["oversampling"]
        else:
            oversampling = 1
        delta_pix = parameters["detector"]["delta_pix"] / oversampling
        with h5py.File(os.path.expanduser(filename), "r") as data:
            # find the sampled wl
            temp = re.compile(r"\d+(?:\.\d*)")
            wl_sampled_h5 = [ele for ele in data.keys() if temp.match(ele)]

            # load psf interpolated into the window
            psf_cube = [
                self.load_imac(
                    wl,
                    data,
                    parameters,
                    oversampling,
                    delta_pix,
                )
                for wl in wl_sampled_h5
            ]
            wl_sampled = [
                self.load_wl_sampled(wl, data) for wl in wl_sampled_h5
            ]

        wl_sampled = np.array(wl_sampled) * u.m
        psf_cube = np.array(psf_cube)

        # wl_interpolate to the output wl grid
        psf_out = self.wl_interpolate(
            psf_cube, wavelength, wl_sampled.to(u.um)
        )

        del psf_cube
        gc.collect()

        # repeat over time
        psf_cube_out = psf_out[np.newaxis, ...]
        del psf_out
        gc.collect()

        if timeDependence:
            psf_cube_out = np.repeat(psf_cube_out, time.size, axis=0)
        return psf_cube_out

    @staticmethod
    def load_wl_sampled(wl, data):
        """
        extract the wavelength from the data stored in the surfaces

        Parameters
        ----------
        wl: str
            wavelength
        data: :class:`h5py.File`
            opened HDF5 file

        Returns
        -------
        float:
            wavelength

        """
        data_wl_surfaces = list(data[wl].keys())
        data_wl_surfaces.sort()
        data_cube = data[wl][data_wl_surfaces[-1]]
        return float(data_cube["wl"][()])

    def load_imac(self, wl, data, parameters, oversampling, delta_pix):
        """
        Returns the PSF interpolated to the detector pixel size.
        The PSF is normalized to the initial volume after the interpolation.

        Parameters
        ----------
        wl: str
            wavelength
        data: :class:`h5py.File`
            opened HDF5 file
        parameters: dict
            detector description
        oversampling: int
            oversampling factor
        delta_pix: float or :class:`astropy.units.Quantity`
            sub-pixel size

        Returns
        -------
        :class:`numpy.ndarray`
            PSF scaled to physical size
        """

        data_wl_surfaces = list(data[wl].keys())
        data_wl_surfaces.sort()
        data_cube = data[wl][data_wl_surfaces[-1]]

        # load data
        dx = data_cube["dy"][()] * u.m
        dy = data_cube["dx"][()] * u.m
        ima = data_cube["amplitude"][()] * data_cube["amplitude"][()]
        ima = ima.T
        vol = ima.sum()

        # scale the psf image
        paos_x, paos_y = np.arange(0, ima.shape[0]) * dx.to(u.um), np.arange(
            0, ima.shape[1]
        ) * dy.to(u.um)

        size = min(
            parameters["detector"]["spatial_pix"],
            parameters["detector"]["spectral_pix"],
        )
        spatial_size = size * oversampling
        spectral_size = size * oversampling

        self.debug("windows sizes: {}, {}".format(spatial_size, spectral_size))

        exosim_x = np.arange(0, spatial_size - 1) * delta_pix.to(u.um)
        exosim_y = np.arange(0, spectral_size - 1) * delta_pix.to(u.um)

        exosim_x -= (
            exosim_x[int(spatial_size // 2)] - paos_x[ima.shape[0] // 2]
        )
        exosim_y -= (
            exosim_y[int(spectral_size // 2)] - paos_y[ima.shape[1] // 2]
        )

        f = RectBivariateSpline(paos_x, paos_y, ima, kx=1, ky=1)
        imac = f(exosim_x.value, exosim_y.value)

        imac -= imac.min()

        # normalise
        norm = imac.sum()
        imac /= norm
        imac *= vol

        return np.array(imac)

    def crop_image_stack(self, psf_out, ene=0.99, time_iteration=False):
        """
        It crops the image stack.
        It takes the PSF for the longest wavelength, as it is expected to be the largest PSF,
        and then selects the smallest aperture which collect at least the desired ene,
        using :func:`~exosim.utils.psf.find_rectangular_aperture`.
        Then it remove these area from all the image in the psf data cube.

        Parameters
        ----------
        psf_out: :class:`numpy.ndarray`
            output PSF cube
        ene: float
            encircled energy desired. Default is 99%.
        time_iteration: bool
            if time iteration is True then it perform

        Returns
        -------
        :class:`numpy.ndarray`
            PSF cube cropped
        """

        h, w = 0, 0
        if time_iteration or psf_out.ndims > 3:
            self.debug("looking for best aperture: iterating over time")
            for t in range(psf_out.shape[0]):
                sizes, surf, ene = find_rectangular_aperture(
                    psf_out[t, -1, :, :], ene
                )
                h = sizes[0] if sizes[0] > h else h
                w = sizes[1] if sizes[1] > w else w

            crop_h = (psf_out.shape[2] - h) // 2
            crop_w = (psf_out.shape[3] - w) // 2

            return psf_out[
                :,
                :,
                int(crop_h) : int(crop_h + h),
                int(crop_w) : int(crop_w + w),
            ]

        else:
            (h, w), surf, ene = find_rectangular_aperture(
                psf_out[-1, :, :], ene
            )
            crop_h = (psf_out.shape[1] - h) // 2
            crop_w = (psf_out.shape[2] - w) // 2

            return psf_out[
                :, int(crop_h) : int(crop_h + h), int(crop_w) : int(crop_w + w)
            ]

    @staticmethod
    # @jit(nopython=True, parallel=True)
    def wl_interpolate(psf_cube, wl, wl_sampled):
        """
        This function interpolates the PSF to the desired wavelength grid

        Parameters
        ----------
        psf_out: :class:`numpy.ndarray`
            output PSF cube
        psf_cube: :class:`numpy.ndarray`
            input PSF cube
        wl: :class:`numpy.ndarray`
            desired wavelength grid
        wl_sampled: :class:`numpy.ndarray`
            input wavelength grid

        Returns
        -------
        :class:`numpy.ndarray`
            output PSF cube
        """
        # take into account different ordering
        if wl_sampled.size > 1:
            wl_unsorted = deepcopy(wl)
            wl.sort()

            psf_cube_resh = psf_cube.reshape(
                psf_cube.shape[0], psf_cube.shape[1] * psf_cube.shape[2]
            )
            f = interp1d(
                wl_sampled,
                psf_cube_resh,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            psf_out = f(wl)
            psf_out = psf_out.reshape(
                wl.size, psf_cube.shape[1], psf_cube.shape[2]
            )
            idx = [int(np.where(wl == wl_)[0]) for wl_ in wl_unsorted]
            psf_out = psf_out[idx, ...]

        else:
            psf_out = np.repeat(psf_cube, wl.size, axis=0)

        # psf_out[k, ...] /= psf_out[k, ...].sum()
        return psf_out
