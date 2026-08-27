import datetime
import os
from collections import OrderedDict

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.table import QTable, hstack, vstack
from scipy.interpolate import interp1d

import exosim.tasks.radiometric as radiometric
from exosim import (
    __author__,
    __branch__,
    __citation__,
    __commit__,
    __copyright__,
    __license__,
    __pkg_name__,
    __title__,
    __url__,
    __version__,
)
from exosim.log import Logger
from exosim.models.channel import Channel
from exosim.output import SetOutput
from exosim.output.hdf5.utils import load_signal
from exosim.tasks.load.load_source_list import LoadSourceList
from exosim.tasks.radiometric import utils
from exosim.utils.ascii_arts import astronomer4
from exosim.utils.klass_factory import find_task
from exosim.utils.output_cleaners import prune_output
from exosim.utils.prepare_recipes import clean_config_files, load_options
from exosim.utils.run_config import RunConfig
from exosim.utils.timed_class import TimedClass
from exosim.utils.types import OutputType

from .create_focal_plane import CreateFocalPlane


class RadiometricModel(CreateFocalPlane):
    """
    Pipeline to create the radiometric model.
    This pipeline has three working modes:

    - it can load an already produced focal plane and use it to estimate a radiometric model;
    - it can produce a single source focal plane and estimate the radiometric model;
    - it can load a target list and produce the radiometric model for each target of the target list.

    Attributes
    ------------
    mainConfig: dict
        This is parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
    output: :class:`~exosim.output.output.Output`
        input/output file
    payloadConfig: dict
        payload configuration dictionary extracted from mainConfig`
    table: :class:`~astropy.table.QTable`
        table for the radiometric estimations

    Examples
    --------

    If the user wants to estimate the radiometric model of an existing focal plane

    >>> import exosim.recipes as recipes
    >>> rm = recipes.RadiometricModel(options_file= 'main _configuration.xml',
    >>>                                output_file = 'focal_plane.h5')

    Otherwise, if a focal plane has not been produced yet, this recipe can produce it,
    if a destination not existing file is provided:

    >>> import exosim.recipes as recipes
    >>> rm = recipes.RadiometricModel(options_file= 'main _configuration.xml',
    >>>                                output_file = 'desired_output.h5')

    In both cases, to store the produced table into the output file,
    the :func:`~exosim.recipes.radiometric_model.RadiometricModel.write`
    is to be used:

    >>> rm.write()

    """

    def __init__(
        self,
        options_file: str | dict,
        output_file: str,
        store_config: bool = False,
        plot: bool = False,
        isolate_every_opt: bool = False,
        slim_output: bool = False,
    ) -> None:
        """
        Initialize the RadiometricModel recipe.

        Parameters
        ----------
        options_file: str or dict
            Path to the main configuration file or a dictionary with the configuration.
        output_file: str
            Path to the output file where the radiometric model will be stored.
        store_config: bool
            If True, the input configuration will be stored in the output file.
        slim_output: bool
            If True, only the necessary data will be stored in the output file to reduce its size.
        """
        Logger.__init__(self)
        TimedClass.__init__(self)
        self.graphics(astronomer4)
        RunConfig.stats()
        self.announce("started")

        clean_config_files()

        # load_options returns a dict for mainConfig
        self.mainConfig, self.payloadConfig = load_options(options_file)
        self.master_table = None
        self.table = None

        # Initialize output file handling
        self.output_file = output_file

        # Initialize grids and pointing from configuration
        self._initialize_grids_and_pointing()

        self.output_file = output_file
        self.out_folder = os.path.dirname(self.output_file)
        self.plot_folder = self.out_folder + "/plots"
        os.makedirs(self.plot_folder, exist_ok=True)

        self.plot = plot

        # isolate every optical element
        if isolate_every_opt:
            self.info("Isolating every optical element")
            self._isolate_every_opt()

        # Decide which of the three flows to run
        # Case 1: existing focal plane
        if (
            os.path.exists(self.output_file)
            and "targetlist_filepath" not in self.mainConfig["sky"]["source"]
        ):
            self.info(f"Input file {self.output_file} exists, using it as focal plane")
            self.output = SetOutput(str(self.output_file), replace=False)
            self.info("Loading existing focal plane for radiometric model")
            self.single_file_pipeline()
            self.common_noise_pipeline()
            self.write()
            self.write_table()
            self.plot_apertures()

        # Case 2: target list
        elif "targetlist_filepath" in self.mainConfig["sky"]["source"]:
            self.info(
                "Target list provided, loading sources and creating focal plane for each source"
            )
            self.target_list_pipeline(store_config, slim_output)
        # Case 3: single source config
        else:
            self.info("Single source config, creating focal plane for the source")
            # define a one-step time grid
            self.mainConfig["time_grid"] = {
                "start_time": 0 * u.hr,
                "end_time": 1 * u.hr,
                "low_frequencies_resolution": 1 * u.hr,
            }
            # Initialize output for standard_run
            self.output = SetOutput(str(self.output_file))
            self.standard_run(store_config, slim_output)
            self.single_file_pipeline()
            self.common_noise_pipeline()
            self.write()
            self.write_table()
            self.plot_apertures()

        self.log_runtime_complete("recipe ended", "info")
        self.announce("ended")

    def plot_apertures(self) -> None:
        if self.plot:
            self.info("Plotting apertures")

            from exosim.plots import RadiometricPlotter

            radiometric_plotter = RadiometricPlotter(
                input=self.output_file,
            )

            radiometric_plotter.plot_apertures()
            fig_name = os.path.join(self.plot_folder, "apertures.png")
            radiometric_plotter.save_fig(fig_name)

    def target_list_pipeline(self, store_config, slim_output: bool = False) -> None:
        """
        Executes the radiometric model pipeline for a list of targets.

        Parameters
        ----------
        store_config : bool
            If True, the configuration for each target will be stored in its respective output group.
        slim_output: bool
            If True, only the necessary data will be stored in the output file to reduce its size
        """

        self.mainConfig["time_grid"] = {
            "start_time": 0 * u.hr,
            "end_time": 1 * u.hr,
            "low_frequencies_resolution": 1 * u.hr,
        }
        self.mainConfig = self.remove_oversampling(self.mainConfig)
        self.PayloadConfig = self.remove_oversampling(self.payloadConfig)

        sky_src = self.mainConfig["sky"]["source"]
        self.info("Target-list config: loading source list")

        self.debug(
            "Loading sources from target list: {}".format(
                sky_src["targetlist_filepath"]
            )
        )

        loader = LoadSourceList()
        sources_list = loader(
            targetlist_filepath=sky_src["targetlist_filepath"],
            source_type=sky_src.get("source_type"),
            column_mapping=sky_src.get("column_mapping"),
        )
        self.info(f"Loaded {len(sources_list)} sources from target list")
        sky_src["sources"] = sources_list

        self.info(f"Sources loaded: {len(sources_list)}")
        self.debug(f"Sources: {sources_list}")

        self.output = SetOutput(self.output_file, replace=True)
        self.log_runtime("preparation ended time", "info")

        with self.output.use(append=True, cache=True) as out:
            self.announce("creating sources focal planes")
            out_targets = out.create_group("targets")
            for i, (source_name, parameters) in enumerate(sources_list.items()):
                self.announce(f"Processing source: {source_name}")

                out_target = out_targets.create_group(source_name)

                self.mainConfig["sky"]["source"] = {
                    **parameters,
                    "value": source_name,
                }

                if store_config:
                    out_target.store_dictionary(self.mainConfig, "configuration")

                sources, common_path = self.prepare_environment(out_target)
                # instrument focal plane production
                self.create_channel_focal_planes(out_target, sources, common_path)

                if i == 0:
                    self.table = utils.create_table(self.payloadConfig)
                    self.rebin_efficiencies()

                    self.info("Computing apertures signals for the first source")
                    self.compute_apertures(target_name=source_name)

                    rad_out = out.create_group("radiometric_apertures")
                    rad_out.write_table("table", self.table, replace=True)

                    self.master_table = self.table.copy()

                    if self.plot:
                        self.info("Plotting efficiencies")

                        from exosim.plots import FocalPlanePlotter

                        focal_plane_plotter = FocalPlanePlotter(
                            input=self.output_file,
                        )
                        focal_plane_plotter.plot_efficiency()
                        focal_plane_plotter.save_fig(
                            f"{self.plot_folder}/efficiency.png"
                        )

                else:
                    self.table = self.master_table.copy()

                self.table.meta.update(sources_list[source_name]["metadata"])

                self.compute_sub_foregrounds_signals(
                    channels_path="targets/" + source_name
                )

                self.compute_foreground_signals(channels_path="targets/" + source_name)
                self.compute_source_signals(channels_path="targets/" + source_name)
                self.table, _ = utils.compute_saturation(
                    self.table,
                    self.payloadConfig,
                    self.output,
                    channels_path="targets/" + source_name,
                    logger=self._logger,
                )
                self.common_noise_pipeline()

                self.write(
                    output_file=self.output_file,
                    target=source_name,
                )

                self.write_table(source_name)

                if self.plot:
                    self.info(f"Plotting radiometric table for {source_name}")

                    from exosim.plots import RadiometricPlotter

                    radiometric_plotter = RadiometricPlotter(
                        input=self.table,
                    )
                    radiometric_plotter.plot_table(contribs=True)
                    radiometric_plotter.save_fig(
                        os.path.join(self.plot_folder, f"radiometric_{source_name}.png")
                    )

                    radiometric_plotter.plot_apertures(out_target["channels"])
                    radiometric_plotter.save_fig(
                        os.path.join(self.plot_folder, f"apertures_{source_name}.png")
                    )
                if slim_output:
                    prune_output(out, logger=self)

            self.info(f"output {self.output_file} size: {out.getsize():.3f}")

    def write_table(self, source_name=None) -> None:
        attrs = {
            "file_time": datetime.datetime.now().isoformat(),
            "creator": self.__class__.__name__,
            "program_name": str(__title__),
            "package name": str(__pkg_name__),
            "program_version": str(__version__),
            "author": str(__author__),
            "copyright": str(__copyright__),
            "license": str(__license__),
            "url": str(__url__),
            "citation": str(__citation__),
            "git commit": str(__commit__),
            "git branch": str(__branch__),
        }
        self.table.meta.update(attrs)
        self.table.write(
            os.path.join(
                self.out_folder,
                (
                    f"{source_name}_radiometric_table.ecsv"
                    if source_name
                    else "radiometric_table.ecsv"
                ),
            ),
            overwrite=True,
        )
        self.info(f"Wrote radiometric table for {source_name}")

    def populate_source_focal_plane(
        self, sources: dict, channel: Channel, out: OutputType | None = None
    ) -> None:
        """
        Populates the focal plane for a given channel with the provided sources.

        Parameters
        ----------
        sources : dict
            A dictionary of sources to propagate.
        channel : :class:`~exosim.models.channel.Channel`
            The channel object to populate the focal plane for.
        out : :class:`~exosim.utils.types.OutputType`, optional
            The output object. Defaults to None.
        """
        channel.propagate_sources(
            sources=sources, Atel=self.payloadConfig["Telescope"]["Atel"]
        )
        channel.rescale_contributions()
        channel.populate_focal_plane()

    def _isolate_every_opt(self) -> None:
        """
        Iterates over the optical elements to isolate them and create sub focal planes.
        """
        from exosim.utils.iterators import iterate_over_opticalElements

        self.mainConfig["sky"] = iterate_over_opticalElements(
            self.mainConfig["sky"], "foregrounds", "isolate", True
        )
        self.payloadConfig = iterate_over_opticalElements(
            self.payloadConfig, "Telescope", "isolate", True
        )
        self.payloadConfig = iterate_over_opticalElements(
            self.payloadConfig, "channel", "isolate", True
        )

    def single_file_pipeline(self) -> None:
        """
        Radiometric pipeline to run for a single target with an already
        produced focal plane. The involved steps are:

        1. creation of the wavelength table with :func:`~exosim.recipes.radiometric_model.RadiometricModel.create_table`;
        2. estimation of the apertures sizes and number of pixels involved with :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_apertures`;
        3. estimation of the signals in the apertures for the sub foregrounds, if any: :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_sub_foregrounds_signals`;
        4. estimation of the total foreground signal in the apertures:  :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_foreground_signals`;
        5. estimation of the source focal plane signal in the aperture: :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_source_signals`;
        6. estimation of the saturation time in the channel: :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_saturation`;

        The pipeline will update the `table` attribute.
        """
        self.info("Computing radiometric signals")

        self.table = utils.create_table(self.payloadConfig)
        self.rebin_efficiencies()
        self.compute_apertures()
        self.compute_sub_foregrounds_signals()
        self.compute_foreground_signals()
        self.compute_source_signals()
        self.table, _ = utils.compute_saturation(
            self.table, self.payloadConfig, self.output
        )

    def common_noise_pipeline(self) -> None:
        """
        Radiometric pipeline to run starting from a radiometric table with already estimated signals.
        It computes the noise.

        1. estimation of the multiaccum factors :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_multiaccum`;
        2. estimation shot noise :func:`~exosim.recipes.radiometric_model.RadiometricModel.compute_photon_noise`;
        3. update total noise :func:`~exosim.recipes.radiometric_model.RadiometricModel.update_total_noise`

        The pipeline will update the `table` attribute.
        """
        self.table, _ = utils.compute_multiaccum(
            self.table, self.payloadConfig, logger=self._logger
        )

        self.table = utils.compute_photon_noise(
            self.table, self.payloadConfig, logger=self._logger
        )

        # dark current noise
        self.table = utils.compute_noise_column(
            self.table,
            self.payloadConfig,
            noise_key="dark_current",
            task_key="dark_current_task",
            default_task=radiometric.ComputeConstantDarkCurrentNoise,
            signal_col="source_signal_in_aperture",
            gain_col="multiaccum_shot_gain",
            output_col="darkcurrent_noise",
            logger=self._logger,
        )

        # read noise
        self.table = utils.compute_noise_column(
            self.table,
            self.payloadConfig,
            noise_key="read_noise",
            task_key="read_noise_task",
            default_task=radiometric.ComputeConstantReadNoise,
            signal_col="source_signal_in_aperture",
            gain_col="multiaccum_read_gain",
            output_col="read_noise",
            logger=self._logger,
        )

        # custom noise
        self.table = utils.compute_noise_column(
            self.table,
            self.payloadConfig,
            noise_key="custom_noise",
            task_key="custom_noise_task",
            default_task=radiometric.ComputeCustomNoise,
            signal_col="source_signal_in_aperture",
            gain_col=None,
            output_col=None,  # we want all the columns
            logger=self._logger,
        )

        self.table, _ = utils.update_total_noise(self.table, logger=self._logger)

    def compute_apertures(self, target_name=None, store: bool = True) -> QTable:
        """
        Estimates the photometric aperture for each spectral bin
        using :class:`~exosim.tasks.radiometric.estimate_apertures.EstimateApertures` by default.

        Parameters
        ----------
        target_name : str, optional
            Name of the target to compute apertures for. Defaults to None.

        Returns
        -------
        :class:`~astropy.table.QTable`
            Table with the apertures for each channel and bin.
        """
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            table_list = []
            for ch in self.payloadConfig["channel"]:
                self.info(f"estimating apertures on {ch}")
                description = self.payloadConfig["channel"][ch]
                estimateApertures_task = (
                    find_task(
                        description["radiometric"]["aperture_photometry"][
                            "apertures_task"
                        ],
                        radiometric.EstimateApertures,
                    )
                    if "apertures_task"
                    in description["radiometric"]["aperture_photometry"]
                    else radiometric.EstimateApertures
                )
                estimateApertures = estimateApertures_task()

                with self.output.open() as f:
                    try:
                        ch_group = f["channels"][ch]
                    except KeyError:
                        try:
                            ch_group = f["targets"][target_name]["channels"][ch]
                        except KeyError:
                            self.warning(f"Channel {ch} not found in the input file")
                            continue

                    self.debug("extracting wavelength grid")
                    osf = ch_group["focal_plane"]["metadata"]["oversampling"][()]
                    focal_plane = ch_group["focal_plane"]["data"][
                        0, osf // 2 :: osf, osf // 2 :: osf
                    ]
                    wl_grid = ch_group["focal_plane"]["spectral"][
                        osf // 2 :: osf
                    ] * u.Unit(ch_group["focal_plane"]["spectral_units"][()])

                    apertures = estimateApertures(
                        table=self.table[self.table["ch_name"] == ch],
                        focal_plane=focal_plane,
                        description=description["radiometric"]["aperture_photometry"],
                        wl_grid=wl_grid,
                    )
                    if store:
                        os.makedirs(f"{self.out_folder}/apertures", exist_ok=True)
                        apertures.write(
                            f"{self.out_folder}/apertures/{ch}_apertures.csv",
                            overwrite=True,
                        )
                    table_list.append(apertures)
        else:
            description = self.payloadConfig["channel"]
            self.info("estimating apertures on {}".format(description["value"]))
            estimateApertures_task = (
                find_task(
                    description["radiometric"]["aperture_photometry"]["apertures_task"],
                    radiometric.EstimateApertures,
                )
                if "apertures_task" in description["radiometric"]["aperture_photometry"]
                else radiometric.EstimateApertures
            )
            estimateApertures = estimateApertures_task()
            with self.output.open() as f:
                f = f["channels"]
                ch = next(iter(f.keys()))
                self.debug("extracting wavelength grid")
                osf = f[ch]["focal_plane"]["metadata"]["oversampling"][()]
                focal_plane = f[ch]["focal_plane"]["data"][
                    0, osf // 2 :: osf, osf // 2 :: osf
                ]
                wl_grid = f[ch]["focal_plane"]["spectral"][osf // 2 :: osf] * u.Unit(
                    f[ch]["focal_plane"]["spectral_units"][()]
                )

                table_list = [
                    estimateApertures(
                        table=self.table,
                        focal_plane=focal_plane,
                        wl_grid=wl_grid,
                        description=description["radiometric"]["aperture_photometry"],
                    )
                ]
                if store:
                    table_list[0].write(
                        os.path.join(self.out_folder, f"{ch}_apertures.csv"),
                        overwrite=True,
                    )

        stack = vstack(table_list)
        self.table = hstack((self.table, stack))

        return hstack((self.table["ch_name", "wavelength"], stack))

    def write(self, output_file: str | None = None, target=None) -> None:
        """
        It adds the radiometric table to the output.
        If the table exists already in the output file, it replaces it.

        Parameters
        ----------
        output_file: str, optional
            Path to the output file. Default is `self.output`.
        target: str, optional
            Target name in the output file where to store the table. Defaults to None.

        """
        output = (
            self.output
            if output_file is None
            else SetOutput(output_file, replace=False)
        )

        with output.use(append=True) as out:
            self.info(f"radiometric table stored in {output.fname}")
            rad_out = out.create_group("radiometric")
            if target is not None:
                rad_out = rad_out.create_group(target)
            rad_out.write_table("table", self.table, replace=True)

    def compute_sub_foregrounds_signals(self, out=None, channels_path=None) -> QTable:
        """
        Estimates the radiometric signals on the foreground sub focal planes for all the
        channels and returns a table with all the contributions.

        It uses :class:`~exosim.tasks.radiometric.computeSubFrgSignalsChannel.ComputeSubFrgSignalsChannel` by default.

        Parameters
        ----------
        out : :class:`~exosim.utils.types.OutputType`, optional
            The output object. Defaults to None.
        channels_path : str, optional
            Path to the channels in the output file. Defaults to None.

        Returns
        -------
        :class:`~astropy.table.QTable`
            Signal table.
        """

        table_list = []
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            for ch in self.payloadConfig["channel"]:
                self.info(f"estimating sub-foreground radiometric signal for {ch}")
                computeFrgSignalsChannel_task = (
                    find_task(
                        self.payloadConfig["channel"][ch]["radiometric"][
                            "sub_frg_signal_task"
                        ],
                        radiometric.ComputeSubFrgSignalsChannel,
                    )
                    if "sub_frg_signal_task"
                    in self.payloadConfig["channel"][ch]["radiometric"]
                    else radiometric.ComputeSubFrgSignalsChannel
                )
                computeFrgSignalsChannel = computeFrgSignalsChannel_task()
                table_list += [
                    computeFrgSignalsChannel(
                        table=self.table[self.table["ch_name"] == ch],
                        ch_name=ch,
                        input_file=self.output,
                        channels_path=channels_path,
                        parameters=self.payloadConfig["channel"],
                    )
                ]
        else:
            self.info(
                "estimating sub-foreground radiometric signal for {}".format(
                    self.payloadConfig["channel"]["value"]
                )
            )
            computeFrgSignalsChannel_task = (
                find_task(
                    self.payloadConfig["channel"]["radiometric"]["sub_frg_signal_task"],
                    radiometric.ComputeSubFrgSignalsChannel,
                )
                if "sub_frg_signal_task" in self.payloadConfig["channel"]["radiometric"]
                else radiometric.ComputeSubFrgSignalsChannel
            )
            computeFrgSignalsChannel = computeFrgSignalsChannel_task()
            table_list = [
                computeFrgSignalsChannel(
                    table=self.table,
                    ch_name=self.payloadConfig["channel"]["value"],
                    input_file=out if out is not None else self.output,
                )
            ]

        stack = vstack(table_list)
        self.table = hstack((self.table, stack))
        for k in self.table.colnames:
            if hasattr(self.table[k], "filled"):
                self.table[k] = self.table[k].filled(0.0)
        ret_k = ["ch_name", "wavelength", *list(stack.keys())]
        return self.table[ret_k], stack

    def rebin_efficiencies(self, channels_path=None) -> None:
        """
        Rebins the efficiencies in the radiometric table to match the wavelength bins.
        It uses :class:`~exosim.tasks.radiometric.rebin_efficiencies.RebinEfficiencies` by default.
        """

        self.info("Rebinning efficiencies to match wavelength bins")
        tr = np.array([])
        qe = np.array([])
        with self.output.open() as f:
            if channels_path is not None:
                f = f[channels_path]
            for ch in self.payloadConfig["channel"]:
                eff_path = f"channels/{ch}/efficiency"
                eff = load_signal(f[eff_path])

                resp_path = f"channels/{ch}/responsivity"
                resp = load_signal(f[resp_path])

                resp.data[0, 0] = (
                    resp.data[0, 0]
                    / (resp.spectral * u.Unit(resp.spectral_units)).to(u.m)
                    * const.c
                    * const.h
                    / u.count
                    * resp.data_units
                )
                tr_func = interp1d(
                    eff.spectral,
                    eff.data[0, 0],
                    assume_sorted=False,
                    fill_value=0.0,
                    bounds_error=False,
                )

                qe_func = interp1d(
                    eff.spectral,
                    resp.data[0, 0],
                    assume_sorted=False,
                    fill_value=0.0,
                    bounds_error=False,
                )

                tab = self.table[self.table["ch_name"] == ch]

                tr_ = self._bin_signal(
                    tab["wavelength"],
                    tr_func(tab["wavelength"]),
                    tab["left_bin_edge"],
                    tab["right_bin_edge"],
                )
                qe_ = self._bin_signal(
                    tab["wavelength"],
                    qe_func(tab["wavelength"]),
                    tab["left_bin_edge"],
                    tab["right_bin_edge"],
                )

                tr = np.concatenate((tr, tr_))
                qe = np.concatenate((qe, qe_))

        self.table["transmission"] = tr
        self.table["qe"] = qe
        return self.table["transmission", "qe"]

    def _bin_signal(self, wl, signal, leftbin, rightbin):
        bsig = [
            np.mean(signal[np.logical_and(wl >= wlow, wl < whigh)])
            for wlow, whigh in zip(leftbin, rightbin, strict=False)
        ]
        return u.Quantity(bsig)

    def compute_foreground_signals(self, channels_path=None) -> QTable:
        """
        Estimates the radiometric signals on the foreground focal plane for all the
        channels and returns a table with all the contributions.

        It uses :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` by default.

        Parameters
        ----------
        channels_path : str, optional
            Path to the channels in the output file. Defaults to None.

        Returns
        -------
        :class:`~astropy.table.QTable`
            Signal table.
        """

        phot = np.array([])
        total_phot = np.array([])
        with self.output.open() as f:
            if channels_path is not None:
                f = f[channels_path]
            if isinstance(self.payloadConfig["channel"], OrderedDict):
                for ch in self.payloadConfig["channel"]:
                    self.info(f"estimating foreground radiometric signal for {ch}")
                    computeSignalsChannel_task = (
                        find_task(
                            self.payloadConfig["channel"][ch]["radiometric"][
                                "signal_task"
                            ],
                            radiometric.ComputeSignalsChannel,
                        )
                        if "signal_task"
                        in self.payloadConfig["channel"][ch]["radiometric"]
                        else radiometric.ComputeSignalsChannel
                    )
                    computeSignalsChannel = computeSignalsChannel_task()

                    focal_plane = f["channels"][ch]["frg_focal_plane"]
                    phot_ = computeSignalsChannel(
                        table=self.table[self.table["ch_name"] == ch],
                        focal_plane=focal_plane,
                    )
                    phot = np.concatenate((phot, phot_))

                    total_phot_ = self.compute_total_signals(
                        table=self.table[self.table["ch_name"] == ch],
                        parameters=self.payloadConfig["channel"][ch],
                        focal_plane=focal_plane,
                        computeSignalsChannel=computeSignalsChannel,
                    )
                    total_phot = np.concatenate((total_phot, total_phot_))

            else:
                self.info(
                    "estimating foreground radiometric signal for {}".format(
                        self.payloadConfig["channel"]["value"]
                    )
                )
                computeSignalsChannel_task = (
                    find_task(
                        self.payloadConfig["channel"]["radiometric"]["signal_task"],
                        radiometric.ComputeSignalsChannel,
                    )
                    if "signal_task" in self.payloadConfig["channel"]["radiometric"]
                    else radiometric.ComputeSignalsChannel
                )
                computeSignalsChannel = computeSignalsChannel_task()

                ch = next(iter(f["channels"].keys()))
                focal_plane = f["channels"][ch]["frg_focal_plane"]
                phot_ = computeSignalsChannel(table=self.table, focal_plane=focal_plane)
                phot = np.concatenate((phot, phot_))

                total_phot_ = self.compute_total_signals(
                    table=self.table[self.table["ch_name"] == ch],
                    parameters=self.payloadConfig["channel"][ch],
                    focal_plane=focal_plane,
                    computeSignalsChannel=computeSignalsChannel,
                )
                total_phot = np.concatenate((total_phot, total_phot_))

        self.table["foreground_signal_in_aperture"] = phot
        self.table["foreground_total_signal"] = total_phot
        return self.table[
            "ch_name",
            "wavelength",
            "foreground_total_signal",
            "foreground_signal_in_aperture",
        ]

    def compute_total_signals(
        self, table, parameters, focal_plane, computeSignalsChannel
    ) -> np.ndarray:
        _table_extended = table.copy()
        if "type" in parameters and parameters["type"].lower() == "photometer":
            # for photometer we also compute the total signal on the focal plane
            _table_extended["spatial_size"] = parameters["detector"]["spatial_pix"]
            _table_extended["spectral_size"] = parameters["detector"]["spectral_pix"]

        if "type" in parameters and parameters["type"].lower() == "spectrometer":
            # for spectrometer we also compute the total signal on bin columns
            _table_extended["spatial_size"] = parameters["detector"]["spatial_pix"]
        return computeSignalsChannel(table=_table_extended, focal_plane=focal_plane)

    def compute_source_signals(self, channels_path=None) -> QTable:
        """
        Estimates the radiometric signals on the source focal plane for all the
        channels and returns a table with all the contributions.

        It uses :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` by default.

        Parameters
        ----------
        channels_path : str, optional
            Path to the channels in the output file. Defaults to None.

        Returns
        -------
        :class:`~astropy.table.QTable`
            Signal table.
        """

        phot = np.array([])
        total_phot = np.array([])
        with self.output.open() as f:
            if channels_path is not None:
                f = f[channels_path]
            if isinstance(self.payloadConfig["channel"], OrderedDict):
                for ch in self.payloadConfig["channel"]:
                    self.info(f"estimating source radiometric signal for {ch}")
                    computeSignalsChannel_task = (
                        find_task(
                            self.payloadConfig["channel"][ch]["radiometric"][
                                "signal_task"
                            ],
                            radiometric.ComputeSignalsChannel,
                        )
                        if "signal_task"
                        in self.payloadConfig["channel"][ch]["radiometric"]
                        else radiometric.ComputeSignalsChannel
                    )
                    computeSignalsChannel = computeSignalsChannel_task()

                    focal_plane = f["channels"][ch]["focal_plane"]
                    phot_ = computeSignalsChannel(
                        table=self.table[self.table["ch_name"] == ch],
                        focal_plane=focal_plane,
                    )
                    phot = np.concatenate((phot, phot_))

                    total_phot_ = self.compute_total_signals(
                        table=self.table[self.table["ch_name"] == ch],
                        parameters=self.payloadConfig["channel"][ch],
                        focal_plane=focal_plane,
                        computeSignalsChannel=computeSignalsChannel,
                    )
                    total_phot = np.concatenate((total_phot, total_phot_))

            else:
                self.info(
                    "estimating source radiometric signal for {}".format(
                        self.payloadConfig["channel"]["value"]
                    )
                )
                computeSignalsChannel_task = (
                    find_task(
                        self.payloadConfig["channel"]["radiometric"]["signal_task"],
                        radiometric.ComputeSignalsChannel,
                    )
                    if "signal_task" in self.payloadConfig["channel"]["radiometric"]
                    else radiometric.ComputeSignalsChannel
                )
                computeSignalsChannel = computeSignalsChannel_task()

                ch = next(iter(f["channels"].keys()))
                focal_plane = f["channels"][ch]["focal_plane"]
                phot_ = computeSignalsChannel(table=self.table, focal_plane=focal_plane)
                phot = np.concatenate((phot, phot_))

                total_phot_ = self.compute_total_signals(
                    table=self.table[self.table["ch_name"] == ch],
                    parameters=self.payloadConfig["channel"][ch],
                    focal_plane=focal_plane,
                    computeSignalsChannel=computeSignalsChannel,
                )
                total_phot = np.concatenate((total_phot, total_phot_))

        self.table["source_signal_in_aperture"] = phot
        self.table["source_total_signal"] = total_phot
        return self.table[
            "ch_name", "wavelength", "source_total_signal", "source_signal_in_aperture"
        ]

    def remove_oversampling(self, configurations: dict) -> dict:
        """
        Remove oversampling from the configurations.
        This is useful to avoid oversampling in the focal plane.

        Parameters
        ----------
        configurations: dict
            configurations dictionary

        Returns
        -------
        dict
            configurations dictionary without oversampling
        """
        self.debug("Removing oversampling from configurations")
        for key, value in configurations.items():
            if isinstance(value, dict):
                configurations[key] = self.remove_oversampling(value)
            elif isinstance(value, list):
                configurations[key] = [self.remove_oversampling(item) for item in value]
            elif key == "oversampling":
                configurations[key] = 1
                self.debug("Removed oversampling")
        return configurations
