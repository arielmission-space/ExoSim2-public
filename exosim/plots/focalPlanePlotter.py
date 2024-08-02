import os
from copy import deepcopy
from typing import List
from typing import Tuple

import astropy.constants as const
import astropy.units as u
import h5py
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import exosim.log as log
from .utils import _create_ordered_cmap
from .utils import prepare_channels_list
from exosim.output.hdf5.utils import load_signal
from exosim.utils.ascii_arts import observatory

plt.rcParams.update({"font.size": 22})

cmap_band = _create_ordered_cmap("Pastel1", roll=-2, delete=-3)
cmap = _create_ordered_cmap("Set1", roll=-3, delete=2)


class FocalPlanePlotter(log.Logger):
    """
    Focal plane plotter.
    This class handles the methods to plot the focal palnes produced by `exosim`.

    Attributes
    -----------
    input: str
        input file name
    fig: :class:`matplotlib.figure.Figure`
        produced figure

    Examples
    ----------

    The following example, given the `test_file.h5` produced by `exosim`,
    plots the focal plane at the first time stamp and stores the figure as `focal_plane.png`.

    >>> from exosim.plots import FocalPlanePlotter
    >>> focalPlanePlotter = FocalPlanePlotter(input='./test_file.h5')
    >>> focalPlanePlotter.plot_focal_plane(time_step=0)
    >>> focalPlanePlotter.save_fig('focal_plane.png')

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
        self.fig = None
        self.announce("started")

    def _prepare_figure(self) -> Tuple[plt.Figure, GridSpec, List[str]]:
        with h5py.File(self.input, "r") as f:
            ch_list = list(f["channels"].keys())
            ch_list.sort()
            widths = []
            for ch in ch_list:
                size_x = f["channels"][ch]["focal_plane"]["spectral"].shape[0]
                size_y = f["channels"][ch]["focal_plane"]["spatial"].shape[0]
                widths += [int(np.ceil(size_x / size_y)), 0.1]

            heights = [1, 1]
            scale = np.ceil(len(widths) / len(heights))
            size_y_fig = 10
            size_x_fig = size_y_fig * scale
            fig = plt.figure(
                constrained_layout=True,
                dpi=300,
                figsize=(size_x_fig, size_y_fig),
            )

            spec = fig.add_gridspec(
                ncols=len(widths),
                nrows=len(heights),
                width_ratios=widths,
                height_ratios=heights,
                wspace=0.1,
                hspace=0.1,
            )
        return fig, spec, ch_list

    def _plot_ch(
        self,
        ch: str,
        fig: plt.Figure,
        spec: GridSpec,
        i: int,
        time_step: int,
        scale="linear",
    ) -> Tuple[plt.Figure, int]:
        """
        It plots a channel focal plane in the indicated figure

        Parameters
        ----------
        ch: str
            channel name\
        fig: :class:`matplotlib.figure.Figure`
            figure to use
        spec: :class:`matplotlib.gridspec.GridSpec`
            figure grid spec
        i: int
            iterator
        time_step: int
            time step identifier
        scale: str (optional)
            scale to use for the plot. If 'dB' the plot is in dB. Default is 'linear'. 

        Returns
        -------
        :class:`matplotlib.figure.Figure`
            populated figure
        i: int
           iterator
        """
        focal_plane, osf = self.load_focal_plane(ch, time_step)
        ax0 = fig.add_subplot(spec[0, i])
        ax1 = fig.add_subplot(spec[1, i])
        i += 1

        ax0.set_title(ch)

        ima = focal_plane
        ima = 10 * np.log10(ima / ima.max()) if scale == "dB" else ima
        im = ax0.imshow(ima, interpolation="none")
        plt.colorbar(im, ax=ax0, cax=fig.add_subplot(spec[0, i]))

        ax1.set_title("extracted focal plane")

        ima = focal_plane[osf // 2 :: osf, osf // 2 :: osf]
        ima = 10 * np.log10(ima / ima.max()) if scale == "dB" else ima
        im = ax1.imshow(ima, interpolation="none")
        plt.colorbar(im, ax=ax1, cax=fig.add_subplot(spec[1, i]))
        return fig, i

    def load_focal_plane(
        self, ch: str, time_step: int
    ) -> Tuple[np.ndarray, int]:
        """
        It loads the channel focal plane from the input file:

        Parameters
        ----------
        ch: str
            channel name
        time_step: int
            time step to plot

        Returns
        -------
        :class:`numpy.ndarray`
            focal plane
        int
            over sampling factor
        """
        with h5py.File(self.input, "r") as f:
            file_path = os.path.join("channels", ch)
            source = load_signal(f[os.path.join(file_path, "focal_plane")])
            frg = load_signal(
                f[os.path.join(file_path, "frg_focal_plane")]
            ).data[time_step]
            try:
                bkg = load_signal(
                    f[os.path.join(file_path, "bkg_focal_plane")]
                ).data[time_step]
            except KeyError:
                bkg = 0
            osf = source.metadata["oversampling"]

            final = source.data[time_step] + frg + bkg

        self.debug("focal planes loaded from {}".format(ch))

        return final, osf

    def plot_focal_plane(
        self, time_step: int = 0, scale="linear"
    ) -> plt.Figure:
        """
        It plots the focal planes at a specific time.
        For each channel it adds a :class:`~matplotlib.axes.Axes` to a figure.
        It returns a :class:`~matplotlib.figure.Figure` with two rows:
        on the first row are reported the oversampled focal planes.
        In the second row are reported the extracted focal plane,
        where the oversampling is removed.
        The focal plane plotted is the combination of the source focal plane
        plus the foreground focal plane.

        Parameters
        ----------
        time_step: int
            time step identifier
        scale: str (optional)
            scale to use for the plot. If 'dB' the plot is in dB. Default is 'linear'.

        Returns
        -------
        :class:`matplotlib.figure.Figure`
            populated figure
        """
        self.info("plot focal plane")

        fig, spec, ch_list = self._prepare_figure()
        i = 0
        for ch in ch_list:
            fig, i = self._plot_ch(ch, fig, spec, i, time_step, scale)
            i += 1

        self.fig = fig

        return fig

    def plot_bands(
        self,
        ax: matplotlib.axes.Axes,
        scale: str = "log",
        channel_edges: bool = True,
        add_legend: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        It plots the channels bands behind the indicated axes.

        Parameters
        -----------
        ax: :class:`matplotlib.axes.Axes`
            axes where to plot the bands
        scale: str
            x axes scale. Default is `log`.
        channel_edges: bool
            if ``True`` the x axes ticks are placed at the channel edges. Default is ``True``.

        Returns
        --------
        :class:`matplotlib.axes.Axes`
            axes with channel bands added

        """
        channels, norm = prepare_channels_list(self.input)

        tick_list, patches = [], []
        for k, channel_name in enumerate(channels):
            with h5py.File(self.input, "r") as f:
                file_path = os.path.join("channels", channel_name)
                file_path = os.path.join(file_path, "focal_plane/metadata")
                wl_min = f[os.path.join(file_path, "wl_min/value")][()]
                wl_max = f[os.path.join(file_path, "wl_max/value")][()]

                ax.axvspan(
                    wl_min,
                    wl_max,
                    alpha=0.3,
                    zorder=0,
                    color=cmap_band(
                        norm(k),
                    ),
                )
                ax.axvspan(
                    wl_min,
                    wl_max,
                    alpha=0.3,
                    zorder=0,
                    color=cmap_band(
                        norm(k),
                    ),
                )
                #            wl_maxs += [wl_max]
                tick_list.append(wl_min)
                tick_list.append(wl_max)
                patches += [
                    mpatches.Patch(
                        color=cmap_band(norm(k)), alpha=0.1, label=channel_name
                    )
                ]
            #       tick_list.append(max(wl_maxs))
        ax.set_xscale(scale)
        if channel_edges:
            ax.set_xticks(tick_list)
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter()
            )

        if add_legend:
            band_legend = ax.legend(
                handles=patches, title="Channels legend", bbox_to_anchor=(1, 1)
            )
            ax.add_artist(band_legend)
        return ax

    def plot_efficiency(
        self,
        scale: str = "log",
        channel_edges: bool = False,
        ch_lengend: bool = False,
        efficiency: str = "all",
    ) -> Tuple[plt.Figure, List[matplotlib.axes.Axes]]:
        """
        It produces a figure with efficiencies for the input table.

        Parameters
        ----------
        scale: str
            x axes scale. Default is `log`.
        channel_edges: bool
            if ``True`` the x axes ticks are placed at the channel edges. Default is ``True``.
        ch_lengend: bool
            if ``True`` add a legend for the channels color. Default is ``True``.
        efficiency: str
            what efficiency to plot. Options are "optical efficiency", "responsivity", "quantum efficiency", "photon conversion efficiency" and "all". Default is "all".
        Returns
        --------
        :class:`matplotlib.figure.Figure`
            plotted figure
        (:class:`matplotlib.axes.Axes`, :class:`matplotlib.axes.Axes`, :class:`matplotlib.axes.Axes`, :class:`matplotlib.axes.Axes`)
            tuple of axis. First axes is for optical efficiency, second is for responsivity, third is for quantum efficiency and fourth is for photon conversion efficiency.

        """
        self.info("plotting efficiency")

        channels, norm = prepare_channels_list(self.input)

        def bands(axs):
            try:
                for ax in axs.flatten():
                    self.plot_bands(
                        ax, scale, channel_edges, add_legend=ch_lengend
                    )
            except:
                try:
                    axs = self.plot_bands(
                        axs, scale, channel_edges, add_legend=ch_lengend
                    )
                except KeyError:
                    pass

        with h5py.File(self.input, "r") as f:
            if efficiency == "optical efficiency":
                fig, axs = plt.subplots(1, 1, figsize=(10, 8))
                bands(axs)
                for k, ch in enumerate(channels):
                    eff_path = "channels/{}/efficiency".format(ch)
                    eff = load_signal(f[eff_path])
                    axs.plot(
                        eff.spectral,
                        eff.data[0, 0],
                        label=ch,
                        color=cmap(norm(k)),
                        zorder=0,
                    )
                axs.set_title("Efficiency")
                axs.set_ylabel("Efficiency")
                axs.set_xlabel(r"Wavelength [$\mu m$]")
                axs.grid()

            elif efficiency == "responsivity":
                fig, axs = plt.subplots(1, 1, figsize=(10, 8))
                bands(axs)
                for k, ch in enumerate(channels):
                    resp_path = "channels/{}/responsivity".format(ch)
                    resp = load_signal(f[resp_path])
                    axs.plot(
                        resp.spectral,
                        resp.data[0, 0],
                        label=ch,
                        color=cmap(norm(k)),
                        zorder=0,
                    )
                axs.set_title("Responsivity")
                axs.set_ylabel(r"${}$".format(resp.data_units))
                axs.set_xlabel(r"Wavelength [$\mu m$]")
                axs.grid()

            elif efficiency == "quantum efficiency":
                fig, axs = plt.subplots(1, 1, figsize=(10, 8))
                bands(axs)
                for k, ch in enumerate(channels):
                    resp_path = "channels/{}/responsivity".format(ch)
                    resp = load_signal(f[resp_path])
                    wl = resp.spectral * u.Unit(resp.spectral_units)
                    qe = (
                        resp.data[0, 0]
                        / wl.to(u.m)
                        * const.c
                        * const.h
                        / u.count
                        * resp.data_units
                    )
                    axs.plot(
                        resp.spectral,
                        qe,
                        label=ch,
                        color=cmap(norm(k)),
                        zorder=0,
                    )
                axs.set_title("Quantum efficiency")
                axs.set_ylabel("Efficiency")
                axs.set_xlabel(r"Wavelength [$\mu m$]")
                axs.grid()

            elif efficiency == "photon conversion efficiency":
                fig, axs = plt.subplots(1, 1, figsize=(10, 8))
                bands(axs)
                for k, ch in enumerate(channels):
                    eff_path = "channels/{}/efficiency".format(ch)
                    eff = load_signal(f[eff_path])
                    resp_path = "channels/{}/responsivity".format(ch)
                    resp = load_signal(f[resp_path])
                    wl = resp.spectral * u.Unit(resp.spectral_units)
                    qe = (
                        resp.data[0, 0]
                        / wl.to(u.m)
                        * const.c
                        * const.h
                        / u.count
                        * resp.data_units
                    )
                    axs.plot(
                        resp.spectral,
                        qe * eff.data[0, 0],
                        label=ch,
                        color=cmap(norm(k)),
                        zorder=0,
                    )
                axs.set_title("photon conversion efficiency")
                axs.set_ylabel("Efficiency")
                axs.set_xlabel(r"Wavelength [$\mu m$]")
                axs.grid()

            elif efficiency == "all":
                fig, axs = plt.subplots(2, 2, figsize=(20, 15))
                bands(axs)

                ax1, ax2, ax3, ax4 = axs.flatten()
                with h5py.File(self.input, "r") as f:
                    for k, ch in enumerate(channels):
                        eff_path = "channels/{}/efficiency".format(ch)
                        eff = load_signal(f[eff_path])
                        ax1.plot(
                            eff.spectral,
                            eff.data[0, 0],
                            label=ch,
                            color=cmap(norm(k)),
                            zorder=0,
                        )

                        resp_path = "channels/{}/responsivity".format(ch)
                        resp = load_signal(f[resp_path])
                        ax2.plot(
                            resp.spectral,
                            resp.data[0, 0],
                            label=ch,
                            color=cmap(norm(k)),
                            zorder=0,
                        )

                        wl = resp.spectral * u.Unit(resp.spectral_units)
                        qe = (
                            resp.data[0, 0]
                            / wl.to(u.m)
                            * const.c
                            * const.h
                            / u.count
                            * resp.data_units
                        )
                        ax3.plot(
                            resp.spectral,
                            qe,
                            label=ch,
                            color=cmap(norm(k)),
                            zorder=0,
                        )

                        ax4.plot(
                            resp.spectral,
                            qe * eff.data[0, 0],
                            label=ch,
                            color=cmap(norm(k)),
                            zorder=0,
                        )

                # locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
                # ax.yaxis.set_major_locator(locmaj)
                # locmin = matplotlib.ticker.LogLocator(base=10.0,
                #                                       subs=(0.2, 0.4, 0.6, 0.8),
                #                                       numticks=12)
                # ax.yaxis.set_minor_locator(locmin)
                # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                # ax.grid(axis='y', which='minor', alpha=0.3)
                # ax.grid(axis='y', which='major', alpha=0.5)
                # #        ax.legend(bbox_to_anchor=(1, 1))
                ax1.set_title("Efficiency")
                ax1.set_xlabel(r"Wavelength [$\mu m$]")
                ax1.set_ylabel("Efficiency")
                ax1.grid()

                ax2.set_title("Responsivity")
                ax2.set_ylabel(r"${}$".format(resp.data_units))
                ax2.set_xlabel(r"Wavelength [$\mu m$]")
                ax2.set_yscale("log")
                ax2.grid()

                ax3.set_title("Quantum efficiency")
                ax3.set_xlabel(r"Wavelength [$\mu m$]")
                ax3.set_ylabel("Efficiency")
                ax3.grid()

                ax4.set_title("Photon conversion efficiency")
                ax4.set_xlabel(r"Wavelength [$\mu m$]")
                ax4.set_ylabel("Efficiency")
                ax4.grid()

            try:
                ax3.legend(
                    prop={"size": 16},
                    loc="upper left",
                    ncol=7,
                    bbox_to_anchor=(0.05, -0.11),
                    labelspacing=1.2,
                    handlelength=1,
                )
                plt.tight_layout()
                plt.subplots_adjust(
                    top=0.92,
                    bottom=0.12,
                    hspace=0.2,
                    wspace=0.2,
                    right=0.85,
                    left=0.1,
                )
            except:
                axs.legend(
                    prop={"size": 12},
                    loc="upper left",
                    ncol=7,
                    bbox_to_anchor=(0.05, -0.11),
                    labelspacing=1.2,
                    handlelength=1,
                )
                plt.tight_layout()
                plt.subplots_adjust(
                    top=0.92,
                    bottom=0.15,
                    # hspace=0.2,
                    # wspace=0.2,
                    # right=0.9,
                    # left=0.1,
                )

        self.fig = fig
        return fig, axs

    def save_fig(self, name: str) -> None:
        """
        It saves the produced figure.

        Parameters
        --------
        name: str
            figure name
        """
        dir_name = os.path.dirname(os.path.abspath(name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        try:
            self.fig.savefig("{}".format(name))

            self.info("plot saved in {}".format(name))
        except AttributeError:
            self.error(
                "the indicated figure is not available. Check if you have produced it."
            )
