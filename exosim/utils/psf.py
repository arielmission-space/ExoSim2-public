import logging

import numpy as np
from scipy.special import j1
from scipy.special import jn_zeros

logger = logging.getLogger(__name__)


def create_psf(
    wl,
    fnum,
    delta,
    nzero=4,
    shape="airy",
    max_array_size=None,
    array_size=None,
):
    """
    Calculates an Airy Point Spread Function arranged as a data-cube.
    The spatial axes are 0 and 1. The wavelength axis is 2.
    Each PSF volume is normalised to unity.

    Parameters
    ----------
    wl: :class:`~astropy.units.Quantity`
        array of wavelengths at which to calculate the PSF
    fnum: float or (float, float)
        Instrument f/number. It can be a tuple of two values, in which case the first value is the f/number for the x axis and the second value is the f/number for the y axis.
    delta: :class:`~astropy.units.Quantity`
        the increment to use [physical unit of length]
    nzero: float
        number of Airy zeros. The PSF kernel will be this big. Calculated at wl.max()
    shape: str (optional)
        Set to 'airy' for a Airy function,to 'gauss' for a Gaussian
    max_array_size: (int,int) (optional)
        Maximum size of the PSF array. If None, the size is calculated from the f/number and the wavelength range.
    array_size: (int or str,int or str) (optional)
        Size of the PSF array. If 'full' then the `max_array_size` are used. If None, the size is calculated from the f/number and the wavelength range.

    Returns
    ------
    :class:`~numpy.ndarray`
        three-dimensional array. Each PSF normalised to unity

    Examples
    ---------
    >>> import astropy.units as u
    >>> from exosim.utils.psf import create_psf

    We produce and plot an Airy PSF:

    >>> img = create_psf(1*u.um, 40, 6*u.um, nzero=8, shape='airy')

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from matplotlib.gridspec import GridSpec
    >>> fig = plt.figure(figsize=(6, 6))
    >>> gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
    >>>                       left=0.1, right=0.9, bottom=0.1, top=0.9,
    >>>                       wspace=0.05, hspace=0.05)
    >>> ax = fig.add_subplot(gs[1, 0])
    >>> ax.imshow(img)
    >>> ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    >>> ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    >>> axis_x = np.arange(0, img.shape[1])
    >>> ax_x.plot(axis_x, img.sum(axis=0))
    >>> ax_x.set_xticks([], [])
    >>> axis_y = np.arange(0, img.shape[0])
    >>> ax_y.plot(img.sum(axis=1), axis_y)
    >>> ax_y.set_yticks([], [])
    >>> plt.show()

    .. plot:: mpl_examples/create_psf_airy.py

    Similarly, we can produce and plot a Gaussian PSF:

    >>> img = create_psf(1*u.um, (40,40), 6*u.um, shape='gauss')

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from matplotlib.gridspec import GridSpec
    >>> fig = plt.figure(figsize=(6, 6))
    >>> gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
    >>>                       left=0.1, right=0.9, bottom=0.1, top=0.9,
    >>>                       wspace=0.05, hspace=0.05)
    >>> ax = fig.add_subplot(gs[1, 0])
    >>> ax.imshow(img)
    >>> ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    >>> ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    >>> axis_x = np.arange(0, img.shape[1])
    >>> ax_x.plot(axis_x, img.sum(axis=0))
    >>> ax_x.set_xticks([], [])
    >>> axis_y = np.arange(0, img.shape[0])
    >>> ax_y.plot(img.sum(axis=1), axis_y)
    >>> ax_y.set_yticks([], [])
    >>> plt.show()

    .. plot:: mpl_examples/create_psf_gauss.py

    We can also create a PSF with different F-numbers:

    >>> img = create_psf(1*u.um, (60,40), 6*u.um, shape='gauss')

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(img, aspect='equal',)
    >>> plt.show()

    .. plot:: mpl_examples/create_psf_gauss_fnum.py


    """

    delta = delta.to(wl.unit)
    if isinstance(fnum, tuple):
        ratio = fnum[1] / fnum[0]
        fnum = fnum[0]
    else:
        ratio = 1

    Nx = int(
        np.round(
            jn_zeros(1, nzero)[-1] / (2.0 * np.pi) * fnum * wl.max() / delta
        ).astype(int)
    )
    Ny = int(
        np.round(
            jn_zeros(1, nzero)[-1]
            / (2.0 * np.pi)
            * fnum
            * ratio
            * wl.max()
            / delta
        ).astype(int)
    )

    if max_array_size is not None:
        max_array_size = list(max_array_size)
        max_array_size[0] = (
            max_array_size[0] // 2
            if max_array_size[0] % 2 == 1
            else (max_array_size[0] + 1) // 2
        )
        max_array_size[1] = (
            max_array_size[1] // 2
            if max_array_size[1] % 2 == 1
            else (max_array_size[1] + 1) // 2
        )

        Nx = max_array_size[0] if Nx > max_array_size[0] else Nx
        Ny = max_array_size[1] if Ny > max_array_size[1] else Ny

    if array_size is not None:
        if array_size[0] == "full":
            if max_array_size is not None:
                Nx = max_array_size[0]
            else:
                logger.error(
                    "max_array_size must be set if array_size is set to full"
                )
                raise ValueError(
                    "max_array_size must be set if array_size is set to full"
                )
        else:
            Nx = (
                array_size[0] // 2
                if array_size[0] % 2 == 1
                else (array_size[0] + 1) // 2
            )

        if array_size[1] == "full":
            if max_array_size is not None:
                Ny = max_array_size[1]
            else:
                logger.error(
                    "max_array_size must be set if array_size is set to full"
                )
                raise ValueError(
                    "max_array_size must be set if array_size is set to full"
                )
        else:
            Ny = (
                array_size[1] // 2
                if array_size[1] % 2 == 1
                else (array_size[1] + 1) // 2
            )

    if shape == "airy":
        d = 1.0 / (fnum * (1.0e-30 * delta.unit + wl))
    elif shape == "gauss":
        sigma = (
            1.029
            * fnum
            * (1.0e-30 * delta.unit + wl)
            / np.sqrt(8.0 * np.log(2.0))
        )
        d = 0.5 / (sigma * sigma)

    x = (
        np.linspace(-Nx * delta.item(), Nx * delta.item(), 2 * Nx + 1)
        * delta.unit
    )
    y = (
        np.linspace(-Ny * delta.item(), Ny * delta.item(), 2 * Ny + 1)
        * delta.unit
    )

    yy, xx = np.meshgrid(y, x)
    yy /= ratio

    if shape == "airy":
        arg = 1.0e-20 * delta.unit + np.pi * np.multiply.outer(
            np.sqrt(yy * yy + xx * xx), d
        )
        arg = arg.value
        img = (j1(arg) / arg) * (j1(arg) / arg)
    elif shape == "gauss":
        arg = np.multiply.outer(yy * yy + xx * xx, d)
        arg = arg.value
        img = np.exp(-arg)

    norm = img.sum(axis=(0, 1))
    img /= norm

    img[..., wl <= 0.0] *= 0.0
    img = np.moveaxis(img, -1, 0)

    return img
