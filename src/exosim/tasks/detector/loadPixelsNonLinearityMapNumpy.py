import numpy as np

from exosim.tasks.detector import LoadPixelsNonLinearityMap


class LoadPixelsNonLinearityMapNumpy(LoadPixelsNonLinearityMap):
    """
    Loads the pixels non-linearity map given a numpy map.
    The input must be a NPY format file (see `numpy documentation <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_) containing the coefficients for each pixel.
    The map should have dimensions (n, y, x) where n is the number of coefficients for the polynomial approximation and y and x are the number of pixels in spatial and spectral directions.
    The map should be indicated under ``pnl_filename`` keyword.

    Returns
    --------
    dict
        channel non linearity map

    Raises
    ------
    KeyError:
        if the output do not have the `map` key

    """

    def model(self, parameters: dict) -> dict:
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the channel parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        dict
            channel pixel non-linearity map

        """
        file_name = parameters["detector"]["pnl_filename"]
        linear_coeff = np.load(file_name)

        pnl_dict = {"map": linear_coeff}

        return pnl_dict
