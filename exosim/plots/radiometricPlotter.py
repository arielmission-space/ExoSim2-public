# This code is inspired by the code developed for ExoRad 2.
# Therefore, we attach here for ExoRad 2 license:
#
# BSD 3-Clause License
#
# Copyright (c) 2020, Lorenzo V. Mugnai, Enzo Pascale, "La Sapienza" Università di Roma
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the names of the copyright holders nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
from typing import Tuple
from typing import Union

import astropy.units as u
import h5py
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import Table

import exosim.log as log
from .utils import _create_ordered_cmap
from .utils import prepare_channels_list
from exosim.output.hdf5.utils import load_signal
from exosim.utils.ascii_arts import observatory

plt.rcParams.update({"font.size": 16})


class RadiometricPlotter(log.Logger):
    """
    Radiometric plotter.
    This class handles the methods to plot the radiometric table produced by `exosim`.

    Attributes
    -----------
    input: str or :class:`astropy.table.QTable`
            input data
    input_table: :class:`astropy.table.QTable`
        input radiometric table
    fig: :class:`matplotlib.figure.Figure`
        produced figure

    Examples
    ----------
    The following example, given the `test_file.h5` preoduced by Exosim,
    plots the radiometric table and stores the figure as `radiometric.png`.

    >>> from exosim.plots import RadiometricPlotter
    >>> radiometricPlotter = RadiometricPlotter(input='./test_file.h5')
    >>> radiometricPlotter.plot_table()
    >>> radiometricPlotter.save_fig('radiometric.png')

    """

    def __init__(self, input: Union[str, Table]) -> None:
        """

        Parameters
        ----------
        input: str or :class:`astropy.table.QTable`
            input data
        """
        self.set_log_name()
        self.graphics(observatory)
        self.announce("started")

        self.input = input
        if isinstance(input, str):
            self.input_table = self.load_table(input)
        else:
            self.input_table = input
        self.fig = None

    def load_table(self, input_file: str) -> Table:
        """
        It loads the radiometric table from the input file:

        Parameters
        ----------
        input_file: str
            input file name

        Returns
        -------
        :class:`astropy.table.QTable`
            loaded radiometric table
        """
        with h5py.File(input_file, "r") as f:
            file_path = "radiometric"
            tab = read_table_hdf5(f[file_path], path="table")
        self.debug("radiometric table loaded")

        return tab

    def plot_bands(
        self,
        ax: plt.Axes,
        scale: str = "log",
        channel_edges: bool = True,
        add_legend: bool = True,
    ) -> plt.Axes:
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
        cmap = _create_ordered_cmap("Pastel1", roll=-2, delete=-3)

        tick_list, patches = [], []
        for k, channel_name in enumerate(channels):
            wl_min = min(
                self.input_table["left_bin_edge"][
                    np.where(self.input_table["ch_name"] == channel_name)
                ]
            )
            if hasattr(wl_min, "unit"):
                wl_min = wl_min.value
            wl_max = max(
                self.input_table["right_bin_edge"][
                    np.where(self.input_table["ch_name"] == channel_name)
                ]
            )
            if hasattr(wl_max, "unit"):
                wl_max = wl_max.value
            ax.axvspan(
                wl_min,
                wl_max,
                alpha=0.3,
                zorder=0,
                color=cmap(
                    norm(k),
                ),
            )
            ax.axvspan(
                wl_min,
                wl_max,
                alpha=0.3,
                zorder=0,
                color=cmap(
                    norm(k),
                ),
            )
            #            wl_maxs += [wl_max]
            tick_list.append(wl_min)
            tick_list.append(wl_max)
            patches += [
                mpatches.Patch(
                    color=cmap(norm(k)), alpha=0.3, label=channel_name
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

    def plot_noise(
        self,
        ax: plt.Axes,
        scale: str = "log",
        channel_edges: bool = True,
        contribs: bool = False,
        ch_lengend: bool = True,
    ) -> plt.Axes:
        """
        It plots the noise components found in the input table in the indicated axes.

        Parameters
        -----------
        ax: :class:`matplotlib.axes.Axes`
            axes where to plot the noises
        scale: str
            x axes scale. Default is `log`.
        channel_edges: bool
            if ``True`` the x axes ticks are placed at the channel edges. Default is ``True``.
        contribs: bool
            if ``True`` all the contributions are plotted. Default is ``False``.
        ch_lengend: bool
            if ``True`` add a legend for the channels color. Default is ``True``.

        Returns
        --------
        :class:`matplotlib.axes.Axes`
            axes with noises plotted

        """

        noise_keys = [
            x for x in self.input_table.keys() if "noise" in x or "custom" in x
        ]
        if not contribs:
            noise_keys = [k for k in noise_keys if "photon_noise" not in k]
            noise_keys += ["source_photon_noise", "foreground_photon_noise"]

        self.debug("noise keys : {}".format(noise_keys))
        for k, n in enumerate(noise_keys):
            if n == "total_noise":
                ax.plot(
                    self.input_table["Wavelength"],
                    self.input_table[n],
                    zorder=9,
                    lw=1,
                    c="k",
                    marker=".",
                    markersize=5,
                    label="total_noise",
                    alpha=0.8,
                )  # , c='None')
            else:
                if self.input_table[n].unit == u.hr**0.5:
                    noise = self.input_table[n]
                elif self.input_table[n].unit == u.ct / u.s:
                    self.debug(
                        "{} rescaled by starSignal_inAperture".format(n)
                    )
                    noise = (
                        self.input_table[n]
                        / self.input_table["source_signal_in_aperture"]
                        / (u.hr.to(u.s)) ** 0.5
                    )
                else:
                    self.error(
                        "{} unit not valid: {}".format(
                            n, self.input_table[n].unit
                        )
                    )
                # ax.scatter(self.input_table['Wavelength'], noise, label=n, zorder=10, s=5, color=palette[k])
                ax.plot(
                    self.input_table["Wavelength"],
                    noise,
                    zorder=9,
                    lw=1,
                    alpha=0.5,
                    marker=".",
                    label=n,
                )  # color=palette[k])  # c='None')

        #        ax.grid(zorder=0, which='both')
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
        ax.yaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12
        )
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="y", which="minor", alpha=0.3)
        ax.grid(axis="y", which="major", alpha=0.5)
        #        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_title("Noise Budget")
        ax.set_xlabel(r"Wavelength [$\mu m$]")
        ax.set_ylabel(r"relative noise [$\sqrt{{hr}}$]")
        ax.set_yscale(scale)
        # ax.set_xscale('log')
        ax = self.plot_bands(ax, scale, channel_edges, add_legend=ch_lengend)
        ax.legend(
            prop={"size": 12},
            loc="upper left",
            ncol=3,
            bbox_to_anchor=(0.05, -0.25),
            labelspacing=1.2,
            handlelength=1,
        )
        return ax

    def plot_signal(
        self,
        ax: plt.Axes,
        ylim: Tuple[float, float] = None,
        scale: str = "log",
        channel_edges: bool = True,
        contribs: bool = False,
        ch_lengend: bool = True,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        It plots the signal components found in the input table in the indicated axes.

        Parameters
        -----------
        ylim: float or (float, float)
            ylim for :class:`matplotlib.axes.Axes`.
        ax: :class:`matplotlib.axes.Axes`
            axes where to plot the signals
        scale: str
            x axes scale. Default is `log`.
        channel_edges: bool
            if ``True`` the x axes ticks are placed at the channel edges. Default is ``True``.
        contribs: bool
            if ``True`` all the contributions are plotted. Default is ``False``.
        ch_lengend: bool
            if ``True`` add a legend for the channels color. Default is ``True``.

        Returns
        --------
        :class:`matplotlib.axes.Axes`
            axes with signals plotted
        """

        keys = ["source_signal_in_aperture", "foreground_signal_in_aperture"]

        if contribs:
            keys = [
                x
                for x in self.input_table.keys()
                if "signal_in_aperture" in x and "noise" not in x
            ]

        self.debug("signal keys : {}".format(keys))
        for k, s in enumerate(keys):
            ax.plot(
                self.input_table["Wavelength"],
                self.input_table[s],
                zorder=9,
                lw=1,
                alpha=0.5,
                marker=".",
                label=s,
            )

        if ylim:
            ax.set_ylim(ylim)
        # elif ax.get_ylim()[0]<:
        #     ax.set_ylim(1e-3)
        #        ax.grid(zorder=0, which='both')
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
        ax.yaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12
        )
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="y", which="minor", alpha=0.3)
        ax.grid(axis="y", which="major", alpha=0.5)
        #        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_title("Signals")
        ax.set_xlabel(r"Wavelength [$\mu m$]")
        ax.set_ylabel("$ct/s$")
        ax.set_yscale(scale)

        ax = self.plot_bands(ax, scale, channel_edges, add_legend=ch_lengend)
        ax.legend(
            prop={"size": 12},
            loc="upper left",
            ncol=3,
            bbox_to_anchor=(0.05, -0.25),
            labelspacing=1.2,
            handlelength=1,
        )
        return ax

    def plot_table(
        self,
        scale: str = "log",
        channel_edges: bool = True,
        contribs: bool = False,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        It produces a figure with signal and noise for the input table.

        Parameters
        ----------
        scale: str
            x axes scale. Default is `log`.
        channel_edges: bool
            if ``True`` the x axes ticks are placed at the channel edges. Default is ``True``.
        contribs: bool
            if ``True`` all the contributions are plotted. Default is ``False``.

        Returns
        --------
        :class:`matplotlib.figure.Figure`
            plotted figure
        (:class:`matplotlib.axes.Axes`, :class:`matplotlib.axes.Axes`)
            tuple of axis. First axes is for signal, second is for noise.

        """
        self.info("plotting radiometric table")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # fig.suptitle(self.input_table.meta['name'])
        ax1 = self.plot_signal(
            ax1, scale=scale, channel_edges=channel_edges, contribs=contribs
        )
        ax2 = self.plot_noise(
            ax2,
            scale=scale,
            channel_edges=channel_edges,
            contribs=contribs,
            ch_lengend=False,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.22, hspace=0.7, right=0.8)
        self.fig = fig
        return fig, (ax1, ax2)

    def plot_efficiency(
        self,
        scale: str = "log",
        channel_edges: bool = False,
        ch_lengend: bool = True,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
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
        Returns
        --------
        :class:`matplotlib.figure.Figure`
            plotted figure
        (:class:`matplotlib.axes.Axes`, :class:`matplotlib.axes.Axes`)
            tuple of axis. First axes is for signal, second is for noise.

        """
        self.info("plotting efficiency table")
        # fig.suptitle(self.input_table.meta['name'])
        channels = set(self.input_table["ch_name"])
        channels = list(channels)
        channels.sort()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        with h5py.File(self.input, "r") as f:
            for ch in channels:
                eff_path = "channels/{}/efficiency".format(ch)
                eff = load_signal(f[eff_path])
                ax1.plot(eff.spectral, eff.data[0, 0], label=ch)

                resp_path = "channels/{}/responsivity".format(ch)
                resp = load_signal(f[resp_path])
                ax2.plot(resp.spectral, resp.data[0, 0], label=ch)

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
        # ax.set_ylabel('$ct/s$')
        # ax1.set_yscale(scale)
        ax2.set_title("Responsivity")
        ax2.set_ylabel(r"${}$".format(resp.data_units))
        ax2.set_xlabel(r"Wavelength [$\mu m$]")
        ax2.set_yscale("log")

        ax1 = self.plot_bands(ax1, scale, channel_edges, add_legend=ch_lengend)
        ax2 = self.plot_bands(ax2, scale, channel_edges, add_legend=ch_lengend)

        ax2.legend(
            prop={"size": 12},
            loc="upper left",
            ncol=7,
            bbox_to_anchor=(0.05, -0.25),
            labelspacing=1.2,
            handlelength=1,
        )
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.92, bottom=0.12, hspace=0.7, right=0.8, left=0.1
        )
        self.fig = fig
        return fig, (ax1, ax2)

    def plot_apertures(self) -> plt.Figure:
        """
        It produces a figure with apertures superimposed to the focal plane.

        Returns
        --------
        :class:`matplotlib.figure.Figure`
            plotted figure

        """
        self.info("plotting apertures")
        table = self.load_table(self.input)

        def _prepare_figure():
            with h5py.File(self.input, "r") as f:
                ch_list = list(f["channels"].keys())
                ch_list.sort()
                widths = []
                for ch in ch_list:
                    size_x = f["channels"][ch]["focal_plane"][
                        "spectral"
                    ].shape[0]
                    size_y = f["channels"][ch]["focal_plane"]["spatial"].shape[
                        0
                    ]
                    widths += [int(np.ceil(size_x / size_y)), 0.1]

                heights = [1]
                scale = np.ceil(len(widths) / len(heights))
                size_y_fig = 10
                size_x_fig = size_y_fig * scale
                fig = plt.figure(
                    constrained_layout=True,
                    dpi=150,
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

        def _load_apertures(ch):
            center_spectral = table["spectral_center"][table["ch_name"] == ch]
            spectral_size = table["spectral_size"][table["ch_name"] == ch]
            center_spatial = table["spatial_center"][table["ch_name"] == ch]
            spatial_size = table["spatial_size"][table["ch_name"] == ch]
            shape = table["aperture_shape"][table["ch_name"] == ch]

            aperture_shapes = {
                "rectangular": photutils.aperture.RectangularAperture,
                "elliptical": photutils.aperture.EllipticalAperture,
            }

            aper = []
            for i in range(center_spectral.size):
                aper += [
                    aperture_shapes[shape[i]](
                        (center_spectral[i] - 1, center_spatial[i] - 1),
                        spectral_size[i],
                        spatial_size[i],
                    )
                ]

            return aper

        fig, spec, ch_list = _prepare_figure()
        i = 0
        for ch in ch_list:
            with h5py.File(self.input, "r") as f:
                file_path = os.path.join("channels", ch)
                focal_plane = load_signal(
                    f[os.path.join(file_path, "focal_plane")]
                )
                foreground = load_signal(
                    f[os.path.join(file_path, "frg_focal_plane")]
                )
                osf = focal_plane.metadata["oversampling"]

                final = focal_plane.data[0] + foreground.data[0]
                ax0 = fig.add_subplot(spec[0, i])

                im = ax0.imshow(
                    final[osf // 2 :: osf, osf // 2 :: osf],
                    interpolation="none",
                )
            ax0.set_title(ch)
            apertures = _load_apertures(ch)
            for aperture in apertures:
                aperture.plot(color="r", lw=2)

            i += 1

            plt.colorbar(im, ax=ax0, cax=fig.add_subplot(spec[i]))
            i += 1

        self.fig = fig
        return fig

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
