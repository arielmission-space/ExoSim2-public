import logging

import numpy as np
from scipy.interpolate import interp1d

from exosim.models.signal import Signal

logger = logging.getLogger(__name__)


def locate_wavelength_windows(
    psf: np.array, focal_plane: Signal, parameters: dict
) -> tuple[np.array, np.array]:
    focal_plane_shape = (
        focal_plane.data.shape if not focal_plane.cached else focal_plane.shape
    )
    ch_type = parameters["type"].lower()
    if ch_type not in ("spectrometer", "photometer"):
        raise ValueError(
            f"unsupported channel type '{parameters['type']}'; "
            "expected 'spectrometer' or 'photometer'"
        )

    if psf.ndim == 4:
        logger.debug("PSF is 4D, will use spectral and spatial dimensions")
        return _locate_spectral_and_spatial(
            psf, focal_plane, parameters, focal_plane_shape
        )

    if psf.ndim == 3:
        logger.debug("PSF is 3D, will use spectral dimension only")
        if ch_type == "spectrometer":
            j0_ = np.round(np.arange(focal_plane_shape[2]) - psf.shape[2] // 2).astype(
                int
            )
        else:  # photometer
            j0_ = np.repeat(
                focal_plane_shape[2] // 2 - psf.shape[2] // 2 - 1,
                focal_plane.spectral.size,
            )
        return None, j0_

    raise ValueError(
        f"the PSF must be 3D or 4D, got a {psf.ndim}D array of shape {psf.shape}"
    )


def _locate_spectral_and_spatial(
    psf: np.array,
    focal_plane: Signal,
    parameters: dict,
    focal_plane_shape: tuple,
) -> tuple[np.array, np.array]:
    if parameters["type"].lower() == "spectrometer":
        j0_ = np.round(np.arange(focal_plane_shape[2]) - psf.shape[3] // 2).astype(int)

        if np.all(np.asarray(focal_plane.spatial) == 0):
            # no spatial wavelength solution: crop the PSF if it is too big in
            # the spatial direction, then centre it on the array
            if psf.shape[2] > focal_plane_shape[1]:
                center = psf.shape[2] // 2
                size = focal_plane_shape[1] // 2
                psf = psf[:, :, center - size : center + size, :]

            i0_ = np.repeat(
                focal_plane_shape[1] // 2 - psf.shape[2] // 2,
                focal_plane.spectral.size,
            )
        else:
            spatial_wl_sol = interp1d(
                focal_plane.spatial,
                np.arange(0, focal_plane_shape[1]),
                fill_value="extrapolate",
            )
            i0_ = np.round(
                spatial_wl_sol(focal_plane.spectral) - psf.shape[2] // 2
            ).astype(int)

    else:  # photometer
        j0_ = np.repeat(
            focal_plane_shape[2] // 2 - psf.shape[3] // 2 - 1,
            focal_plane.spectral.size,
        )
        i0_ = np.repeat(
            focal_plane_shape[1] // 2 - psf.shape[2] // 2 - 1,
            focal_plane.spectral.size,
        )
    return i0_, j0_
