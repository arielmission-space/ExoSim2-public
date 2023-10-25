import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from exosim.log import generate_logger_name


def rebin(
    x,
    xp,
    fp,
    axis=0,
    mode="mean",
    fill_value=0.0,
):
    """
    This function can resample multidimensional array along a given axis.
    Resample a function fp(xp) over the new grid x, rebinning if necessary,
    otherwise interpolates. Interpolation is done using 'linear' method.
    This function doesn't perform extrapolation: unsample coordinates will be filled with filled value.


    Parameters
    ----------
    x: 	:class:`~numpy.ndarray`
        New coordinates
    fp:	:class:`~numpy.ndarray`
        y-coordinates to be resampled
    xp:	:class:`~numpy.ndarray`
        x-coordinates at which fp are sampled
    axis: int (optional)
        fp axis to resample. Default is 0.
    mode: str (optional)
        the mode indicates the statistc to use for binning by :func:`scipy.stats.binned_statistic`.
        Default is 'mean'.
    fill_value: float (optional)
        fill value for unsampled coordinates. Default is 0.0.

    Returns
    -------
    :class:`~numpy.ndarray`
        re-sampled fp

    Raises
    -------
    NotImplementedError
        If the mode is not implemented.

    Examples
    ---------
    >>> import numpy as np
    >>> from exosim.utils.binning import rebin

    We define the original function, sampled in data 50 points:

    >>> xp = np.linspace(1,10, 50)
    >>> fp = np.sin(xp)

    We bin it down, sampling it at 10 points

    >>> x_bin = np.linspace(1,10, 10)
    >>> f_bin = rebin(x_bin, xp, fp)

    We wl_interpolate the original function, sampling it at 100 points

    >>> x_inter = np.linspace(1,10, 100)
    >>> f_inter = rebin(x_inter, xp, fp)


    To compare the outcome of our interpolation, we produce a plot.

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax0, ax1) = plt.subplots(2,1)

    In the top panel we want to see the original function compared to the binned one and the interpolated one.

    >>> ax0.plot(xp, fp, label='original array', alpha=0.5, c='r')
    >>> ax0.scatter(x_inter, f_inter, marker='X', label='interpolated array', alpha=0.5)
    >>> ax0.scatter(x_bin, f_bin, marker='V', label='binned array', alpha=0.5)
    >>> ax0.legend()

    In the bottom panel we compare each function with the true value and we divide this quantity by the true value.

    >>> ax1.axhline(0, c='r', alpha=0.5)
    >>> ax1.scatter(x_inter, (f_inter- np.sin(x_inter))/np.sin(x_inter),
    ...             marker='X', label='interpolated array', alpha=0.5)
    >>> ax1.scatter(x_bin, (f_bin - np.sin(x_bin))/np.sin(x_bin),
    ...             marker='v', label='binned array', alpha=0.5)
    >>> ax1.legend()
    >>> plt.show()

    .. plot:: mpl_examples/rebin.py

    It's important to notice here that when we perform the binning, we are not simply resampling the input function,
    but we are using the mean value inside each of the bins.
    """

    x, xp, fp = _clean_up_inputs(x, xp, fp, axis)

    # binning
    idx = np.where(
        np.logical_and(x > np.min(xp), x < np.max(xp))
    )  # I'm only investigating in the interested area of the grid
    try:
        if (
            np.diff(xp).max() < np.diff(x[idx]).min()
        ):  # using max vs min I'm sure that if there is an empty bin it switches to interpolate
            return _bin_with_statistics(x, xp, fp, axis, statistic=mode)
        else:
            return _interpolating(x, xp, fp, axis, fill_value=fill_value)
    except ValueError:
        # if there are empty bins, switch to interpolation
        logger.warning("empty bins, switching to interpolation")
        return _interpolating(x, xp, fp, axis, fill_value=fill_value)


def _bin_with_statistics(x, xp, fp, axis, statistic="mean"):
    logger.debug("binning")

    bin_x = 0.5 * (x[1:] + x[:-1])
    x0 = x[0] - (bin_x[0] - x[0]) / 2.0
    x1 = x[-1] + (x[-1] - bin_x[-1]) / 2.0
    bin_x = np.insert(bin_x, [0], x0)
    bin_x = np.append(bin_x, x1)

    new_f = np.apply_along_axis(
        lambda a: binned_statistic(xp, a, bins=bin_x, statistic=statistic)[0],
        axis=axis,
        arr=fp,
    )
    return new_f


def _interpolating(x, xp, fp, axis, fill_value=0.0):
    logger.debug("interpolating")

    func = interp1d(
        xp,
        fp,
        axis=axis,
        fill_value=fill_value,
        assume_sorted=False,
        bounds_error=False,
        kind="linear",
    )
    new_f = func(x)
    return new_f


def _clean_up_inputs(x, xp, fp, axis):
    # cast into numpy array if they are not
    xp = np.array(xp)
    x = np.array(x)
    fp = np.array(fp)

    # remove NaNs
    idx = np.where(np.isnan(xp))[0]
    if idx.size > 0:
        logger.debug("Nans found in input x array: removing it")
        xp = np.delete(xp, idx)
        fp = np.delete(fp, idx, axis=axis)

    idx = np.where(np.isnan(fp))[0]
    if idx.size > 0:
        logger.debug("Nans found in input f array: removing it")
        xp = np.delete(xp, idx)
        fp = np.delete(fp, idx, axis=axis)

    idx = np.where(np.isnan(x))[0]
    if idx.size > 0:
        logger.debug("Nans found in new x array: removing it")
        x = np.delete(x, idx)

    # remove duplicates
    while np.diff(xp).min() == 0:
        logger.debug("duplicate found in input x array: removing it")
        idx = np.argmin(np.diff(xp))
        xp = np.delete(xp, idx)
        fp = np.delete(fp, idx, axis=axis)

    while np.diff(x).min() == 0:
        logger.debug("duplicate found in new x array: removing it")
        idx = np.argmin(np.diff(x))
        x = np.delete(x, idx)

    return x, xp, fp


logger = logging.getLogger(generate_logger_name(rebin))
