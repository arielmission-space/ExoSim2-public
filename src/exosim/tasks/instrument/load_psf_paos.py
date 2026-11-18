import gc
import logging
import os
import re

import astropy.units as u
import h5py
import numpy as np
from scipy.interpolate import interp1d

import exosim.tasks.instrument.load_psf as load_psf
from exosim.utils.aperture import find_rectangular_aperture
from exosim.utils.psf import scale_psf_image

logger = logging.getLogger(__name__)


class LoadPsfPaos(load_psf.LoadPsf):
    """
    It loads the PSFs from a PAOS file.
    Each PAOS file contains a PSF vs wavelength.
    The task loads the PSF cube provided by the `filename` parameter.
    The PSF are then interpolated over a grid matching the one used to produce the focal planes,
    to convert them into the physical units.
    The resampling conserves the flux exactly: each PAOS sample is the integral over its own cell and each sub-pixel
    holds the sum of the overlaps of the PAOS cells with it, so that the window contains its true share of the PAOS volume
    (the sum of the squared PAOS amplitude, which already accounts for the transmission loss of the optical path).
    The energy of the PAOS PSF that falls outside the window is therefore *not* put back into the core.
    The PSF are then interpolated over a wavelength grid matching the one used to for the focal plane, producing the cube.
    Between the PAOS wavelengths the images are linearly interpolated. Outside them the image of the nearest PAOS
    wavelength is scaled in size proportionally to the wavelength, conserving its energy
    (:func:`~exosim.utils.psf.scale_psf_image`), because a linear extrapolation of the images gives negative,
    unphysical PSFs. This is exact for the diffraction pattern and keeps the aberrations of the edge wavelength.
    Then, the PSF cube is repeated on the temporal axis, because no temporal variation is considered in this Task.

    Returns
    -------
    :class:`~numpy.ndarray`
        cube of psfs.  axis=0 is time, axis=1 is wavelength, axis=2 is spatial direction, axis=3 is spectral direction.
    """

    def model(self, filename, parameters, wavelength, time):
        self.debug("loading PAOS psf")

        if "psf" in parameters and "time_dependence" in parameters["psf"]:
            timeDependence = parameters["psf"]["time_dependence"]
        else:
            timeDependence = True
        oversampling = parameters["detector"].get("oversampling", 1)
        delta_pix = parameters["detector"].get("delta_pix", 1) / oversampling
        with h5py.File(os.path.expanduser(filename), "r") as data:
            # find the sampled wl
            temp = re.compile(r"\d+(?:\.\d*)")
            wl_sampled_h5 = [ele for ele in data if temp.match(ele)]

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
            wl_sampled = [self.load_wl_sampled(wl, data) for wl in wl_sampled_h5]

        wl_sampled = np.array(wl_sampled) * u.m
        psf_cube = np.array(psf_cube)

        # wl_interpolate to the output wl grid
        psf_out = self.wl_interpolate(psf_cube, wavelength, wl_sampled.to(u.um))

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
        Returns the PSF resampled to the detector sub-pixel size.
        The resampling is exact and flux conserving: each PAOS sample (squared amplitude) is treated as the
        integral of the intensity over its own rectangular cell
        (``dx * dy``, centred on the sample) and each sub-pixel holds the sum
        of the overlaps of the PAOS cells with it (separable overlap
        matrices). Each sub-pixel therefore holds the *flux in its cell*
        (integrated), not a point value; the IRF box sum in the focal-plane
        population then turns it into the exact pixel integral. The PAOS
        energy inside the window is conserved exactly at any oversampling,
        and the PSF is zero outside the PAOS array.

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
        # ima = ima.T

        # scale the psf image
        paos_x, paos_y = (
            np.arange(0, ima.shape[0]) * dx.to(u.um),
            np.arange(0, ima.shape[1]) * dy.to(u.um),
        )

        size = min(
            parameters["detector"]["spatial_pix"],
            parameters["detector"]["spectral_pix"],
        )
        spatial_size = size * oversampling
        spectral_size = size * oversampling

        self.debug(f"windows sizes: {spatial_size}, {spectral_size}")

        exosim_x = np.arange(0, spatial_size - 1) * delta_pix.to(u.um)
        exosim_y = np.arange(0, spectral_size - 1) * delta_pix.to(u.um)

        # align the PSF centre (PAOS array centre) with the sample that
        # downstream tasks take as the centre of the returned image, i.e.
        # ``shape // 2``. The grids have size ``spatial_size - 1``, so the
        # index ``spatial_size // 2`` would be one sample off.
        exosim_x -= exosim_x[exosim_x.size // 2] - paos_x[ima.shape[0] // 2]
        exosim_y -= exosim_y[exosim_y.size // 2] - paos_y[ima.shape[1] // 2]

        # exact flux-conserving rebinning (separable): every PAOS sample
        # is the integral over its own cell, every sub-pixel receives the
        # sum of the overlaps of the PAOS cells with it. Outside the PAOS
        # array there is no information, hence no energy.
        wx = self._overlap_matrix(
            exosim_x.to(u.um).value,
            delta_pix.to(u.um).value,
            paos_x.to(u.um).value,
            dx.to(u.um).value,
        )
        wy = self._overlap_matrix(
            exosim_y.to(u.um).value,
            delta_pix.to(u.um).value,
            paos_y.to(u.um).value,
            dy.to(u.um).value,
        )
        imac = wx @ ima @ wy.T

        return np.array(imac)

    @staticmethod
    def _overlap_matrix(sub_centres, sub_size, paos_centres, paos_size):
        """
        Overlap matrix between two sets of equally spaced 1D cells.

        Element ``[i, j]`` is the length of the overlap between the sub-pixel
        interval ``i`` (``sub_centres[i] +- sub_size / 2``) and the PAOS cell
        ``j`` (``paos_centres[j] +- paos_size / 2``) divided by the PAOS cell
        length, i.e. the fraction of the PAOS cell that falls in the
        sub-pixel. Each row of ``W @ image`` is thus the energy collected by
        the sub-pixel, and the energy of the PAOS cells is conserved exactly.
        """
        s_lo = (sub_centres - 0.5 * sub_size)[:, None]
        s_hi = (sub_centres + 0.5 * sub_size)[:, None]
        p_lo = (paos_centres - 0.5 * paos_size)[None, :]
        p_hi = (paos_centres + 0.5 * paos_size)[None, :]
        overlap = np.clip(np.minimum(s_hi, p_hi) - np.maximum(s_lo, p_lo), 0.0, None)
        return overlap / paos_size

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
                sizes, _surf, ene = find_rectangular_aperture(psf_out[t, -1, :, :], ene)
                h = max(h, sizes[0])
                w = max(w, sizes[1])

            crop_h = (psf_out.shape[2] - h) // 2
            crop_w = (psf_out.shape[3] - w) // 2

            return psf_out[
                :,
                :,
                int(crop_h) : int(crop_h + h),
                int(crop_w) : int(crop_w + w),
            ]

        (h, w), _surf, ene = find_rectangular_aperture(psf_out[-1, :, :], ene)
        crop_h = (psf_out.shape[1] - h) // 2
        crop_w = (psf_out.shape[2] - w) // 2

        return psf_out[:, int(crop_h) : int(crop_h + h), int(crop_w) : int(crop_w + w)]

    @staticmethod
    def wl_interpolate(psf_cube, wl, wl_sampled):
        """
        This function interpolates the PSF to the desired wavelength grid.
        Inside the range of the sampled wavelengths the images are linearly
        interpolated. Outside it, the image of the nearest sampled wavelength
        :math:`\\lambda_e` is used: scaled in size by :math:`\\lambda / \\lambda_e`
        conserving its energy (:func:`~exosim.utils.psf.scale_psf_image`). A linear
        extrapolation of the images is not used, as it produces negative,
        unphysical PSFs. A single sampled wavelength is the edge PSF at every other
        wavelength.

        Parameters
        ----------
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
            if hasattr(wl_sampled, "unit") and hasattr(wl, "unit"):
                wl = wl.to(wl_sampled.unit)
            wl_v = np.asarray(getattr(wl, "value", wl), dtype=float)
            ws_v = np.asarray(getattr(wl_sampled, "value", wl_sampled), dtype=float)
            ws_order = np.argsort(ws_v)
            ws_v = ws_v[ws_order]
            psf_cube = psf_cube[ws_order]
            lam_min, lam_max = ws_v[0], ws_v[-1]

            # avoid O(n^2) searches by using argsort + inverse index mapping
            order = np.argsort(wl_v)
            wl_sorted = wl_v[order]

            psf_cube_resh = psf_cube.reshape(
                psf_cube.shape[0], psf_cube.shape[1] * psf_cube.shape[2]
            )
            f = interp1d(
                ws_v,
                psf_cube_resh,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )

            # interpolate on the sorted wavelength grid, then reorder back
            psf_out_sorted = f(wl_sorted)
            psf_out_sorted = psf_out_sorted.reshape(
                wl_sorted.size, psf_cube.shape[1], psf_cube.shape[2]
            )

            # outside the range: edge PSF, scaled (or not) in size
            for mask, edge, lam_e in (
                (wl_sorted < lam_min, 0, lam_min),
                (wl_sorted > lam_max, -1, lam_max),
            ):
                if not mask.any():
                    continue
                logger.info(
                    f"{mask.sum()} wavelengths outside the PSF range: PSFs at "
                    f"{lam_min:.4g}-{lam_max:.4g}, requested "
                    f"{wl_sorted.min():.4g}-{wl_sorted.max():.4g}; "
                    f"using the PSF at {lam_e:.4g}, scaled in size with the wavelength"
                )
                for k in np.where(mask)[0]:
                    psf_out_sorted[k] = scale_psf_image(
                        psf_cube[edge], wl_sorted[k] / lam_e
                    )
            inv_order = np.argsort(order)
            psf_out = psf_out_sorted[inv_order, ...]

        else:
            # a single PSF: it is the edge PSF at every other wavelength
            lam_e = float(
                np.asarray(getattr(wl_sampled, "value", wl_sampled)).ravel()[0]
            )
            if hasattr(wl_sampled, "unit") and hasattr(wl, "unit"):
                wl = wl.to(wl_sampled.unit)
            wl_v = np.asarray(getattr(wl, "value", wl), dtype=float)
            psf_out = np.repeat(psf_cube, wl.size, axis=0)
            if np.any(wl_v != lam_e):
                logger.info(
                    f"a single PSF at {lam_e:.4g}, scaled in size with the wavelength "
                    f"for {wl_v.min():.4g}-{wl_v.max():.4g}"
                )
            for k in np.where(wl_v != lam_e)[0]:
                psf_out[k] = scale_psf_image(psf_cube[0], wl_v[k] / lam_e)

        # psf_out[k, ...] /= psf_out[k, ...].sum()
        return psf_out
