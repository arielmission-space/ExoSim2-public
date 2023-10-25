import logging

import numpy as np
import photutils
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def find_rectangular_aperture(
    ima, desired_ene, center=None, start_h=None, start_w=None
):
    """
    Estimates the smallest rectangular aperture to collect the desired Encircled Energy from a PSF.
    It uses :func:`photutils.aperture.aperture_photometry`.

    Parameters
    ----------
    ima: :class:`~numpy.ndarray`
        two-dimensional array.
    desired_ene: float
        desired Encircled Energy
    center: (float, float)
        spectral and spatial coordinates of the aperture center in pixels.
        If None the center of the array is used. Default is `None`
    start_h: int
        starting aperture height
    start_w: int
        starting aperture width

    Returns
    -------
    (float, float)
        sizes of the rectangular aperture (h,w)
    float
        number of pixels in the aperture
    float
        encircled energy collected by the aperture

    Example
    --------
    In this example we find the optimal aperture containing the 85% of the energy of a PSF.

    >>> import matplotlib.pyplot as plt
    >>> import astropy.units as u
    >>> import photutils
    >>> from exosim.utils.aperture import find_rectangular_aperture
    >>> from exosim.utils.psf import create_psf
    >>>
    >>> img = create_psf(1*u.um, (60,40), 6*u.um)
    >>> size, area, ene = find_rectangular_aperture(img, 0.85)
    >>> positions = [(img.shape[1]//2,img.shape[0]//2)]
    >>> aperture = photutils.aperture.RectangularAperture(positions, size[0], size[1])
    >>>
    >>> plt.imshow(img)
    >>> aperture.plot(color='r', lw=2,)
    >>> plt.show()

    .. plot:: mpl_examples/find_rectangular_aperture.py

    """

    center = center if center else (ima.shape[1] / 2, ima.shape[0] / 2)

    def ene_func(center, h, w, ima):
        aper = photutils.aperture.RectangularAperture(center, h=h, w=w)
        phot_ = photutils.aperture.aperture_photometry(ima, aper)
        phot = phot_["aperture_sum"].data[0]
        return phot / ima.sum()

    def ene_fit(pos, center, ima, desired_ene):
        ene = ene_func(center, pos[0], pos[1], ima)
        enec = np.abs(ene - desired_ene)
        return enec

    start_h_ = start_h if start_h else 2.0
    start_w_ = start_w if start_w else 2.0

    res = minimize(
        ene_fit,
        x0=np.array([float(start_h_), float(start_w_)]),
        args=(center, ima, desired_ene),
        method="Nelder-Mead",
        tol=1.0e-6,
    )

    h = res.x[0]
    w = res.x[1]
    return (w, h), h * w, ene_func(center, h, w, ima)


def find_elliptical_aperture(ima, desired_ene, center=None):
    """
    Estimates the smallest elliptical aperture to collect the desired Encircled Energy from a PSF.
    It uses :func:`photutils.aperture.aperture_photometry`.

    Parameters
    ----------
    ima: :class:`~numpy.ndarray`
        two-dimensional array.
    desired_ene: float
        desired Encircled Energy
    center: (float, float)
        spectral and spatial coordinates of the aperture center in pixels.
        If None the center of the array is used. Default is `None`

    Returns
    -------
    (float, float)
        sizes of the elliptical aperture (h,w)
    float
        number of pixels in the aperture
    float
        encircled energy collected by the aperture

    Example
    --------

    In this example we find the optimal aperture containing the 85% of the energy of a PSF.

    >>> import matplotlib.pyplot as plt
    >>> import astropy.units as u
    >>> import photutils
    >>> from exosim.utils.psf import find_elliptical_aperture
    >>> from exosim.utils.psf import create_psf
    >>>
    >>> img = create_psf(1*u.um, (60,40), 6*u.um)
    >>> size, area, ene = find_elliptical_aperture(img, 0.85)
    >>> positions = [(img.shape[1]//2,img.shape[0]//2)]
    >>> aperture = photutils.aperture.EllipticalAperture(positions, size[0], size[1])
    >>>
    >>> plt.imshow(img)
    >>> aperture.plot(color='r', lw=2,)
    >>> plt.show()

    .. plot:: mpl_examples/find_elliptical_aperture.py


    """
    center = center if center else (ima.shape[1] / 2, ima.shape[0] / 2)

    def ene_func(center, a, b, ima):
        aper = photutils.aperture.EllipticalAperture(center, a=a, b=b)
        phot_ = photutils.aperture.aperture_photometry(ima, aper)
        phot = phot_["aperture_sum"].data[0]
        return phot / ima.sum()

    def ene_fit(pos, center, ima, desired_ene):
        ene = ene_func(center, pos[0], pos[1], ima)
        enec = np.abs(ene - desired_ene)
        return enec

    res = minimize(
        ene_fit,
        x0=np.array([1, 1]),
        args=(center, ima, desired_ene),
        method="Nelder-Mead",
        tol=1.0e-6,
    )

    a = res.x[0]
    b = res.x[1]
    return (a, b), a * b, ene_func(center, a, b, ima)


def find_bin_aperture(ima, desired_ene, spatial_with, center=None):
    """
    Estimates the smallest rectangular aperture for spectral bin of already fixed spatial size
    to collect the desired Encircled Energy from a PSF.
    It uses :func:`photutils.aperture.aperture_photometry`.

    Parameters
    ----------
    ima: :class:`~numpy.ndarray`
        two-dimensional array.
    spatial_with: float
        fixed bin spatial size
    desired_ene: float
        desired Encircled Energy
    center: (float, float)
        spectral and spatial coordinates of the aperture center in pixels.
        If None the center of the array is used. Default is `None`

    Returns
    -------
    float
        spectral size of the rectangular aperture
    float
        number of pixels in the aperture
    float
        encircled energy collected by the aperture
    """
    center = center if center else (ima.shape[1] / 2, ima.shape[0] / 2)

    def ene_func(center, h, w, psf):
        aper = photutils.aperture.RectangularAperture(center, h=h, w=w)
        phot_ = photutils.aperture.aperture_photometry(psf, aper)
        phot = phot_["aperture_sum"].data[0]

        aper_full = photutils.aperture.RectangularAperture(
            center, h=psf.shape[1], w=w
        )
        phot_ = photutils.aperture.aperture_photometry(psf, aper_full)
        phot_full = phot_["aperture_sum"].data[0]

        return phot / phot_full

    def ene_fit(pos, spatial_with, center, ima, desired_ene):
        h = pos[0]
        ene = ene_func(center, h, spatial_with, ima)
        enec = np.abs(ene - desired_ene)
        return enec

    res = minimize(
        ene_fit,
        x0=np.array([2.0]),
        args=(spatial_with, center, ima, desired_ene),
        method="Nelder-Mead",
        tol=1.0e-6,
    )

    h = res.x[0]
    return h, h * spatial_with, ene_func(center, h, spatial_with, ima)
