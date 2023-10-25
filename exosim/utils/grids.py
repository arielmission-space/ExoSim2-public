import astropy.units as u
import numpy as np

from .checks import check_units


def wl_grid(wl_min, wl_max, R, return_bin_width=False):
    r"""
    It returns the wavelength log-grid in microns

    The wavelength at the center of the spectral bins is defined as

    .. math::

        \lambda_c = \frac{1}{2} (\lambda_j + \lambda_{j+1} )

    where :math:`\lambda_j` is the wavelength at the bin edge defined by the recursive relation,
    and :math:`R` is the `logbin_resolution` defined by the user.

    .. math::

        \lambda_{j+1} = \lambda_{j} \left( 1 + \frac{1}{R} \right)

    And, given the maximum and minimum wavelengths, provided by the user, the number of bins is

    .. math::

        n_{bins} = \frac{\log \left( \frac{\lambda_{max}}{\lambda_{min}} \right) } {\log \left( 1 + \frac{1}{R}\right)} + 1


    Parameters
    ----------
    wl_min: :class:`~astropy.units.Quantity` or float.
        minimum wavelength sampled. If no units are attached is considered as expressed in `um`
    wl_max: :class:`~astropy.units.Quantity` or float.
        maximum wavelength sampled. If no units are attached is considered as expressed in `um`
    R: int
        spectral resolving power
    return_bin_width: bool
        if True returns also the bin width. Default is False.

    Returns
    -------
    :class:`~astropy.units.Quantity`
        wavelength grid
    :class:`~astropy.units.Quantity`
        bin wavelength width
    """
    wl_min, wl_max = check_units(wl_min, u.um), check_units(wl_max, u.um)

    number_of_spectral_bins = (
        np.ceil(np.log(wl_max / wl_min) / np.log(1.0 + 1.0 / R)) + 1
    )
    wl_bin = wl_min * (1.0 + 1.0 / R) ** np.arange(number_of_spectral_bins)
    wl_bin_c = 0.5 * (wl_bin[0:-1] + wl_bin[1:])
    wl_bin_width = wl_bin[1:] - wl_bin[0:-1]

    if return_bin_width:
        return wl_bin_c, wl_bin_width
    else:
        return wl_bin_c


def time_grid(time_min, time_max, low_frequencies_resolution):
    """
    It returns the time grid in hours

    Parameters
    ----------
    time_min: :class:`~astropy.units.Quantity` or float.
        minimum time sampled. If no units are attached is considered as expressed in hours.
    time_max: :class:`~astropy.units.Quantity` or float.
        maximum time sampled. If no units are attached is considered as expressed in hours.
    low_frequencies_resolution: :class:`~astropy.units.Quantity` or float.
        time sampling interval. If no units are attached is considered as expressed in hours.


    Returns
    -------
    :class:`~astropy.units.Quantity`
        time grid

    """
    time_min, time_max, low_frequencies_resolution = (
        check_units(time_min, u.hr),
        check_units(time_max, u.hr),
        check_units(low_frequencies_resolution, u.hr),
    )

    return (
        np.arange(
            time_min.value, time_max.value, low_frequencies_resolution.value
        )
        * u.hr
    )
