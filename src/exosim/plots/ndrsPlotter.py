import os
from typing import Tuple

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np

import exosim.log as log
from exosim.utils.ascii_arts import observatory


class NDRsPlotter(log.Logger):
    """
    Sub-Exposures plotter.
    This class handles the methods to plot all the sub-exposures produced by `ExoSim`.


    Examples
    ----------
    The following example, given the `test_file.h5` preoduced by Exosim,
    plots the sub-exposures stores the figures in the indicated folder.

    >>> from exosim.plots import SubExposuresPlotter
    >>> subExposuresPlotter = SubExposuresPlotter(input='./test_ndr.h5')
    >>> subExposuresPlotter.plot('plots/')

    """

    def __init__(self, input: str) -> None:
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

    def plot(self, out_dir: str) -> None:
        """It iterates over the channels and plot the ndrs.

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
                exposures, time_line, n_groups_per_ramp = self.load_ndrs(ch, f)

                for i in range(n_groups_per_ramp):
                    self.plot_NDRs(exposures, time_line, i, ch, out_dir)

    def plot_NDRs(
        self,
        ndrs: np.ndarray,
        time_line: u.Quantity,
        i: int,
        ch: str,
        out_dir: str,
    ) -> None:
        """It plots the ndrs for a given channel.

        Parameters
        ----------
        ndrs : np.ndarray
            ndrs array
        time_line : u.Quantity
            temporal array
        i : int
            index of ndr to plot
        ch : str
            channel name
        out_dir : str
            output directory name
        """
        t = time_line[i]

        size_x = ndrs[i].shape[1]
        size_y = ndrs[i].shape[0]

        size_y_fig = 10
        size_x_fig = (size_y_fig * size_y / size_x) * 1.2

        fig, ax0 = plt.subplots(
            constrained_layout=True, dpi=300, figsize=(size_y_fig, size_x_fig)
        )

        if t < 0.01 * u.hr:
            t = t.to(u.s)

        t_str = "{:.3f} {}".format(t.value, t.unit)
        if t < 0.01 * u.s:
            t_str = "{:.2e} {}".format(t.value, t.unit)

        ax0.set_title("{} - num {} - time {}".format(ch, i, t_str))

        im = ax0.imshow(ndrs[i], interpolation="none")
        plt.colorbar(im, ax=ax0)

        fname = os.path.join(out_dir, "{}_ndrs_{}.png".format(ch, i))
        fig.savefig(fname, format="png", dpi=300)

        self.info("plot saved in {}".format(i))

    def load_ndrs(
        self, ch: str, f: h5py.File
    ) -> Tuple[np.ndarray, u.Quantity, int]:
        """
        It loads the channel NDRs from the input file:

        Parameters
        ----------
        ch: str
            channel name
        file: :class:`h5py.File`
           input file

        Returns
        -------
        :class:`numpy.ndarray`
            NDR
        :class:`astropy.units.Quantity`
            time line
        int
            number of groups per ramp
        """
        file_path = os.path.join("channels/", ch)
        se = f[os.path.join(file_path, "NDRs")]
        ndrs = se["data"]

        time_line = se["time"][()] * u.Unit(
            se["time_units"][()].decode("utf-8")
        )

        n_groups_per_ramp = se["metadata"]["n_groups_per_ramp"][()]

        self.debug("NDRs loaded from {}".format(ch))

        return ndrs, time_line, n_groups_per_ramp
