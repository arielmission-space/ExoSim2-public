import logging

import numpy as np
from scipy.special import erf, j1, jn_zeros

logger = logging.getLogger(__name__)


def _scaling_matrix(n, s):
    """
    Matrix of the exact rebinning of a 1D array of ``n`` cells onto the same
    cells after a stretch by ``s`` about the centre cell ``n // 2``.
    Element ``[i, j]`` is the fraction of the stretched cell ``j`` (centred on
    ``s * (j - n // 2)``, of size ``s``) that falls in the cell ``i``.
    """
    idx = np.arange(n) - n // 2
    out_lo, out_hi = (idx - 0.5)[:, None], (idx + 0.5)[:, None]
    src_lo, src_hi = (s * (idx - 0.5))[None, :], (s * (idx + 0.5))[None, :]
    overlap = np.clip(
        np.minimum(out_hi, src_hi) - np.maximum(out_lo, src_lo), 0.0, None
    )
    return overlap / s


def scale_psf_image(image, s):
    """
    Scales a PSF image in size by the factor ``s`` about its centre, conserving
    the energy.

    The image holds the energy of each cell (sum = energy inside the window)
    and its centre is the cell ``shape // 2`` on both axes. The result is

    .. math:: I_s(x, y) = I(x / s, y / s) / s^2

    computed as the exact rebinning (overlap of the cells) of the stretched cells
    onto the original grid, so that it is non-negative and, for ``s >= 1``, the
    energy that stays inside the window is conserved to rounding. The energy
    that the stretching moves outside the window is lost (it is *not* put back),
    consistently with the normalisation of the PSFs to their total
    energy. For ``s < 1`` the PSF
    shrinks, and each stretched cell, smaller than a cell of the grid, deposits
    all its energy in the cell(s) it overlaps: the energy is conserved, but the
    shrunk PSF is block-averaged, so the resolution of the grid is the limit.

    The scaling is exact for the diffraction pattern (its size is proportional
    to the wavelength); it keeps the aberration pattern of the original image.

    Parameters
    ----------
    image: :class:`numpy.ndarray`
        2D array of cell energies.
    s: float
        size factor, ``wavelength / reference wavelength``.

    Returns
    -------
    :class:`numpy.ndarray`
        scaled image, same shape.
    """
    image = np.asarray(image, dtype=float)
    if s <= 0:
        raise ValueError("the scale factor must be positive")
    if abs(s - 1.0) < 1e-12:
        return image.copy()
    wx = _scaling_matrix(image.shape[0], s)
    wy = _scaling_matrix(image.shape[1], s)
    return wx @ image @ wy.T


def create_psf(
    wl,
    fnum,
    delta,
    nzero=4,
    shape="airy",
    max_array_size=None,
    array_size=None,
    name=None,
):
    """
    Calculates an Airy Point Spread Function arranged as a data-cube.
    The spatial axes are 0 and 1. The wavelength axis is 2.
    Each PSF is normalised to the *total* energy of the analytic
    profile: the array sums to the fraction of the
    energy that falls inside the (finite) window, a number smaller than one.
    That fraction is computed analytically (exact for the Gaussian, from a
    precomputed table for the Airy pattern), so it does not depend on the
    sampling, also when the PSF is undersampled.

    Parameters
    ----------
    wl: :class:`~astropy.units.Quantity`
        array of wavelengths at which to calculate the PSF
    fnum: float or (float, float)
        Instrument f/number. It can be a tuple of two values, in which case the first value is the f/number for the x axis and the second value is the f/number for the y axis.
    delta: :class:`~astropy.units.Quantity`
        the increment to use [physical unit of length]
    nzero: float
        number of dark rings of the Airy pattern that the window reaches: the
        half width of the PSF window is the radius of the ``nzero``-th dark
        ring, :math:`j_{1,n} F \\lambda / \\pi`, calculated at ``wl.max()``. The
        Gaussian PSF uses the same window.
    shape: str (optional)
        Set to 'airy' for a Airy function,to 'gauss' for a Gaussian
    max_array_size: (int,int) (optional)
        Maximum size of the PSF array. If None, the size is calculated from the f/number and the wavelength range.
    array_size: (int or str,int or str) (optional)
        Size of the PSF array. If 'full' then the `max_array_size` are used. If None, the size is calculated from the f/number and the wavelength range.
    name: str (optional)
        label used in the log messages (e.g. the channel name).

    Returns
    ------
    :class:`~numpy.ndarray`
        three-dimensional array. The sum of each PSF is the fraction of its
        energy inside the window.

    Examples
    ---------
    >>> import astropy.units as u
    >>> from exosim.utils.psf import create_psf

    We produce and plot an Airy PSF:

    >>> img = create_psf(
    ...     1 * u.um, 40, 6 * u.um, nzero=8, shape="airy"
    ... )

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

    >>> img = create_psf(
    ...     1 * u.um, (40, 40), 6 * u.um, shape="gauss"
    ... )

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

    >>> img = create_psf(
    ...     1 * u.um, (60, 40), 6 * u.um, shape="gauss"
    ... )

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(
    ...     img,
    ...     aspect="equal",
    ... )
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
        np.round(jn_zeros(1, nzero)[-1] / np.pi * fnum * wl.max() / delta).astype(int)
    )
    Ny = int(
        np.round(
            jn_zeros(1, nzero)[-1] / np.pi * fnum * ratio * wl.max() / delta
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

        Nx = min(Nx, max_array_size[0])
        Ny = min(Ny, max_array_size[1])

    if array_size is not None:
        if array_size[0] == "full":
            if max_array_size is not None:
                Nx = max_array_size[0]
            else:
                logger.error("max_array_size must be set if array_size is set to full")
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
                logger.error("max_array_size must be set if array_size is set to full")
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
        sigma = 1.029 * fnum * (1.0e-30 * delta.unit + wl) / np.sqrt(8.0 * np.log(2.0))
        d = 0.5 / (sigma * sigma)

    x = np.linspace(-Nx * delta.item(), Nx * delta.item(), 2 * Nx + 1) * delta.unit
    y = np.linspace(-Ny * delta.item(), Ny * delta.item(), 2 * Ny + 1) * delta.unit

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

    # Fraction of the TOTAL energy of the analytic profile that falls inside
    # the window, computed without using the samples, so that it does not
    # depend on the sampling (the sum of the point samples of an undersampled
    # PSF differs from the integral because of aliasing). The window is the
    # rectangle holding the 2N+1 sample cells: its half extent is
    # (n_samples / 2) * delta = (N + 1/2) * delta on each axis, i.e. from the
    # centre of the central sample to the outer edge of the last cell.
    valid = np.asarray(wl.value > 0.0)
    f_in = _window_energy_fraction(
        shape, fnum, ratio, wl, delta, img.shape[0], img.shape[1], valid
    )

    window_sum = img.sum(axis=(0, 1))
    # avoid dividing by zero at wl <= 0 (those PSFs are set to zero below)
    window_sum = np.where(valid, window_sum, 1.0)
    inside = np.where(valid, f_in, np.nan)
    _log_window_fraction(inside, valid, name)

    # keep the shape of the samples, set the sum to the true window fraction
    img /= window_sum
    img *= np.where(valid, f_in, 1.0)

    img[..., wl <= 0.0] *= 0.0
    return np.moveaxis(img, -1, 0)


_AIRY_EE_TABLE = None
_AIRY_EE_RANGE = (0.04, 300.0)


def _build_airy_ee_table():
    """Energy of the standard Airy pattern inside rectangles centred on it.

    The pattern is ``I(x, y) = (2 J1(pi r) / (pi r))**2`` with ``r**2 = x**2 +
    y**2`` in units of ``F * wl`` and total energy ``4 / pi``. The table holds
    ``EE(u, v)``, the fraction of the total energy inside ``|x| < u, |y| < v``,
    on a log-spaced grid of ``u`` and ``v``. It is computed once by a separable
    2-point Gauss-Legendre quadrature in each cell of a Cartesian grid (cell
    0.01 up to 2, 0.05 up to 30 and 0.1 beyond), using the four-fold symmetry. The grid
    abscissae are snapped to cell edges, so the integration is exact in the
    sense that the rectangles are exactly the integrated areas.
    """
    lo, hi = _AIRY_EE_RANGE
    edges = np.concatenate(
        [
            np.arange(0.0, 2.0, 0.01),
            np.arange(2.0, 30.0, 0.05),
            np.arange(30.0, hi + 0.05, 0.1),
        ]
    )
    edges = np.round(edges, 10)
    # 2-point Gauss-Legendre nodes in each cell
    g = 1.0 / np.sqrt(3.0)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    nodes = np.empty(2 * mid.size)
    weights = np.empty(2 * mid.size)
    nodes[0::2], nodes[1::2] = mid - g * half, mid + g * half
    weights[0::2] = weights[1::2] = half
    # table abscissae: log spaced, snapped to cell edges
    target = np.geomspace(lo, hi, 700)
    idx = np.unique(np.searchsorted(edges, target))
    idx = idx[(idx > 0) & (idx < edges.size)]
    u = edges[idx]
    node_idx = 2 * idx - 1  # last node below each edge

    ee = np.zeros((u.size, u.size))
    run = np.zeros(u.size)
    chunk = 400
    ptr = 0
    for start in range(0, nodes.size, chunk):
        stop = min(start + chunk, nodes.size)
        xx = nodes[start:stop, None]
        arg = np.pi * np.sqrt(xx**2 + nodes[None, :] ** 2)
        inten = (2.0 * j1(arg) / arg) ** 2
        cy = np.cumsum(inten * weights[None, :], axis=1)[:, node_idx]
        cx = np.cumsum(cy * weights[start:stop, None], axis=0) + run[None, :]
        run = cx[-1]
        while ptr < u.size and node_idx[ptr] < stop:
            ee[ptr] = cx[node_idx[ptr] - start]
            ptr += 1
    # quadrant integral -> rectangle |x|<u, |y|<v, fraction of 4 / pi
    ee *= 4.0 / (4.0 / np.pi)
    # store EE / (u v), a smooth function equal to pi at the origin: the
    # interpolation is then accurate also for small windows
    return np.log(u), ee / (u[:, None] * u[None, :])


def airy_rectangle_energy(u, v):
    """Fraction of the total energy of an Airy pattern inside a rectangle.

    Parameters
    ----------
    u, v: float or :class:`~numpy.ndarray`
        half widths of the rectangle along the two axes, in units of
        ``F_x * wl`` and ``F_y * wl`` respectively (``F`` is the f-number
        along the axis).

    Returns
    -------
    :class:`~numpy.ndarray`
        energy inside ``|x| < u, |y| < v`` divided by the total energy.

    Notes
    -----
    The values come from a table computed once (lazily) and interpolated
    bilinearly in ``log(u)``, ``log(v)`` (the interpolated quantity is
    ``EE / (u v)``, which is smooth). Half widths beyond the table (300)
    are clamped to it (the neglected energy is below ``3e-4``); below
    0.04 the energy is proportional to the area of the rectangle.
    """
    global _AIRY_EE_TABLE
    if _AIRY_EE_TABLE is None:
        _AIRY_EE_TABLE = _build_airy_ee_table()
    logu, ee = _AIRY_EE_TABLE
    lo, hi = np.exp(logu[0]), np.exp(logu[-1])
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    uc, vc = np.clip(u, lo, hi), np.clip(v, lo, hi)
    fu = np.interp(np.log(uc), logu, np.arange(logu.size))
    fv = np.interp(np.log(vc), logu, np.arange(logu.size))
    i = np.minimum(fu.astype(int), logu.size - 2)
    j = np.minimum(fv.astype(int), logu.size - 2)
    a, b = fu - i, fv - j
    g = (
        ee[i, j] * (1 - a) * (1 - b)
        + ee[i + 1, j] * a * (1 - b)
        + ee[i, j + 1] * (1 - a) * b
        + ee[i + 1, j + 1] * a * b
    )
    # EE = g * u * v; below the table g is constant (energy ~ area)
    return g * np.minimum(u, hi) * np.minimum(v, hi)


def _window_energy_fraction(shape, fnum, ratio, wl, delta, nx, ny, valid):
    """Fraction of the total analytic energy inside the ``nx * ny`` window."""
    wx = 0.5 * nx * delta.to(wl.unit).value
    wy = 0.5 * ny * delta.to(wl.unit).value
    wl_v = np.where(valid, wl.value, 1.0)
    if shape == "airy":
        u = wx / (fnum * wl_v)
        v = wy / (fnum * ratio * wl_v)
        return airy_rectangle_energy(u, v)
    sigma = 1.029 * fnum * wl_v / np.sqrt(8.0 * np.log(2.0))
    return erf(wx / (np.sqrt(2.0) * sigma)) * erf(wy / (np.sqrt(2.0) * sigma * ratio))


def _log_window_fraction(inside, valid, name):
    """Logs the fraction of PSF energy that falls outside the PSF window."""
    if not np.any(valid):
        return
    outside = 1.0 - inside[valid]
    label = f" of {name}" if name else ""
    logger.info(
        "PSF%s: fraction of energy outside the window %.2f%% - %.2f%% "
        "(min - max over wavelength)",
        label,
        100 * np.min(outside),
        100 * np.max(outside),
    )
    if np.max(outside) > 0.01:
        logger.warning(
            "PSF%s: up to %.1f%% of the energy falls outside the PSF window. "
            "It is NOT counted in the signal. Consider a larger `nzero` or `size`.",
            label,
            100 * np.max(outside),
        )
