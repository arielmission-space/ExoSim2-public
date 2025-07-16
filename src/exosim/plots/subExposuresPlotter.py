import os
from typing import Tuple

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np

import exosim.log as log
from exosim.utils.ascii_arts import observatory


class SubExposuresPlotter(log.Logger):
    """
    Sub-Exposures plotter.
    This class handles the methods to plot all the sub-exposures produced by `ExoSim`.


    Examples
    ----------
    The following example, given the `test_file.h5` preoduced by Exosim,
    plots the sub-exposures stores the figures in the indicated folder.

    >>> from exosim.plots import SubExposuresPlotter
    >>> subExposuresPlotter = SubExposuresPlotter(input='./test_se.h5')
    >>> subExposuresPlotter.plot('plots/')

    """

    def __init__(self, input):
        """

        Parameters
        ----------
        input: str
            input data
        """
        self.set_log_name()
        self.graphics(observatory)
        self.input = input
        self.announce("started")

    def plot(self, out_dir: str):
        """It iterates over the channels and plot the sub-exposures.

        Parameters
        ----------
        out_dir : str
            output directory
        """ """ """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with h5py.File(self.input, "r") as f:
            ch_list = list(f["channels"].keys())
            ch_list.sort()

            for ch in ch_list:
                self.info("plotting {}".format(ch))
                (
                    exposures,
                    time_line,
                    integration_times,
                    n_exposure_per_ramp,
                ) = self.load_ndrs(ch, f)

                for i in range(n_exposure_per_ramp):
                    self.plot_SubExposure(
                        exposures, time_line, integration_times, i, ch, out_dir
                    )

    def plot_SubExposure(
        self,
        exposures: np.ndarray,
        time_line: u.Quantity,
        integration_times: u.Quantity,
        i: int,
        ch: str,
        out_dir: str,
    ) -> None:
        """It plots the sub-exposures for a given channel.


        Parameters
        ----------
        ndrs : np.ndarray
            ndrs array
        time_line : u.Quantity
            temporal array
        integration_times: u.Quantity
            array for the integration times
        i : int
            index of ndr to plot
        ch : str
            channel name
        out_dir : str
            output directory name
        """ """ """
        t = time_line[i]
        it = integration_times[i]

        size_x = exposures[i].shape[1]
        size_y = exposures[i].shape[0]

        size_y_fig = 10
        size_x_fig = (size_y_fig * size_y / size_x) * 1.2

        fig, ax0 = plt.subplots(
            constrained_layout=True, dpi=300, figsize=(size_y_fig, size_x_fig)
        )

        if t < 0.01 * u.hr:
            t = t.to(u.s)
        if it < 0.01 * u.hr:
            it = it.to(u.s)

        t_str = "{:.3f} {}".format(t.value, t.unit)
        if t < 0.01 * u.s:
            t_str = "{:.2e} {}".format(t.value, t.unit)
        it_str = "{:.3f} {}".format(it.value, it.unit)
        if it < 0.01 * u.s:
            it_str = "{:.2e} {}".format(it.value, it.unit)

        ax0.set_title(
            "{} - num {} - time {} - integration time {}".format(
                ch, i, t_str, it_str
            )
        )

        im = ax0.imshow(exposures[i], interpolation="none")
        plt.colorbar(im, ax=ax0)

        fname = os.path.join(out_dir, "{}_se_{}.png".format(ch, i))
        fig.savefig(fname, format="png", dpi=300)

        self.info("plot saved in {}".format(i))

    def load_ndrs(
        self, ch: str, f: h5py.File
    ) -> Tuple[np.ndarray, u.Quantity, u.Quantity, int]:
        """
        It loads the channel sub-exposures from the input file:

        Parameters
        ----------
        ch: str
            channel name
        f: :class:`h5py.File`
            input file

        Returns
        -------
        np.ndarray
            sub-exposures array
        u.Quantity
            temporal array
        u.Quantity
            array for the integration times
        int
            number of exposures per ramp
        """
        file_path = os.path.join("channels/", ch)
        se = f[os.path.join(file_path, "SubExposures")]
        exposures = se["data"]

        time_line = se["time"][()] * u.Unit(
            se["time_units"][()].decode("utf-8")
        )
        integration_times = se["metadata"]["integration_times"]["value"][
            ()
        ] * u.Unit(
            se["metadata"]["integration_times"]["unit"][()].decode("utf-8")
        )
        params = dict(f[os.path.join(file_path, "reading_scheme_params")])
        n_subexposures_per_groups = params["n_NRDs_per_group"][()]
        n_groups_per_ramp = params["n_GRPs"][()]
        n_exposure_per_ramp = n_subexposures_per_groups * n_groups_per_ramp
        self.debug("SubExposures loaded from {}".format(ch))

        return exposures, time_line, integration_times, n_exposure_per_ramp
