import os.path
from collections import OrderedDict

import astropy.units as u

import exosim.log as log
import exosim.tasks.parse as parse
import exosim.utils as utils
from exosim.models.channel import Channel
from exosim.output import Output, SetOutput
from exosim.utils.ascii_arts import astronomer1
from exosim.utils.output_cleaners import prune_output
from exosim.utils.prepare_recipes import (
    clean_config_files,
    copy_input_files,
    load_options,
)
from exosim.utils.run_config import RunConfig
from exosim.utils.timed_class import TimedClass
from exosim.utils.types import OutputType


class CreateFocalPlane(TimedClass, log.Logger):
    """
    Pipeline to create the instrument focal planes.
    This pipeline loads the configuration file and produces an output, if indicated,
    where all the products are stored.
    It loads the source SED and the foregrounds and, after the optical chain production,
    it estimates the focal plane for the source and for the foregrounds.

    Attributes
    ------------
    mainConfig: dict
        This is parsed from :class:`~exosim.tasks.load.load_options.LoadOptions`
    output: :class:`~exosim.output.output.Output` (optional)
        output file
    payloadConfig: dict
        payload configuration dictionary extracted from mainConfig`
    time: :class:`~astropy.units.Quantity`
        time grid.
    wl_grid: :class:`~numpy.ndarray` or :class:`~astropy.units.Quantity`
        wavelength grid.

    Examples
    --------

    >>> import exosim.recipes as recipes
    >>> recipes.CreateFocalPlane(options_file= 'main _configuration.xml',
    >>>                          output_file = 'output_file.h5')

    """

    def __init__(
        self,
        options_file: str | dict,
        output_file: str,
        store_config: bool = False,
        slim_output: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        options_file: str or dict
            input configuration file
        output_file: str
            output file
        store_config: bool
            store the input configuration into the output (default is False)
        slim_output: bool
            if True, only the focal plane data will be stored in the output file to reduce its size
        """
        super().__init__()

        self.graphics(astronomer1)
        RunConfig.stats()
        self.announce("started")
        clean_config_files()
        self.mainConfig, self.payloadConfig = load_options(options_file)

        if output_file is not None:
            copy_input_files(os.path.dirname(os.path.abspath(output_file)))

        self.output = SetOutput(output_file)
        self.output_file = output_file

        # Initialize grids and pointing from configuration
        self._initialize_grids_and_pointing()

        self.standard_run(store_config=store_config, slim_output=slim_output)
        self.announce("ended")

    def _initialize_grids_and_pointing(self) -> None:
        """
        Initialize wavelength grid, time grid, and pointing from main configuration.

        Notes
        -----
        This method sets up:
        - self.wl_grid: Wavelength grid from wl_grid configuration
        - self.time_grid: Time grid from time_grid configuration, or default [0] * u.hr
        - self.pointing: Telescope pointing (RA, DEC) tuple, or None if not specified

        The wavelength grid is always required and must be present in the configuration.
        Time grid and pointing are optional with sensible defaults.
        """
        # Wavelength grid definition (required)
        if "wl_grid" in self.mainConfig:
            wl_min = self.mainConfig["wl_grid"]["wl_min"]
            wl_max = self.mainConfig["wl_grid"]["wl_max"]
            logbin_resolution = self.mainConfig["wl_grid"]["logbin_resolution"]
        else:
            # Fallback: infer from channel config or use sensible defaults
            wl_min, wl_max, logbin_resolution = 1.0 * u.um, 2.0 * u.um, 100
            try:
                channels_cfg = self.mainConfig.get(
                    "channels"
                ) or self.payloadConfig.get("channels")
                if isinstance(channels_cfg, dict) and len(channels_cfg) > 0:
                    first_ch = next(iter(channels_cfg.values()))
                    wl_min = first_ch.get("wl_min", wl_min)
                    wl_max = first_ch.get("wl_max", wl_max)
            except Exception:
                pass
        self.wl_grid = utils.grids.wl_grid(wl_min, wl_max, logbin_resolution)
        self.debug(
            f"Wavelength grid initialized: {len(self.wl_grid)} bins from "
            f"{wl_min} to {wl_max}"
        )

        # Time grid definition (optional)
        if "time_grid" in self.mainConfig:
            self.time_grid = utils.grids.time_grid(
                self.mainConfig["time_grid"]["start_time"],
                self.mainConfig["time_grid"]["end_time"],
                self.mainConfig["time_grid"].get("low_frequencies_resolution", None),
            )
            self.debug(
                f"Time grid initialized with {len(self.time_grid)} points from "
                f"{self.mainConfig['time_grid']['start_time']} to "
                f"{self.mainConfig['time_grid']['end_time']}"
            )
        else:
            self.time_grid = [0] * u.hr
            self.debug("No time grid specified, using default [0] * u.hr")

        # Pointing definition (optional)
        if "pointing" in self.mainConfig:
            self.pointing = (
                self.mainConfig["pointing"]["ra"],
                self.mainConfig["pointing"]["dec"],
            )
            self.debug(
                f"Telescope pointing set to RA={self.pointing[0]}, DEC={self.pointing[1]}"
            )
        else:
            self.pointing = None
            self.debug("No telescope pointing specified")

    def standard_run(
        self, store_config: bool = False, slim_output: bool = False
    ) -> None:
        """
        It runs the focal plane pipeline, producing the output file.
        Parameters
        ----------
        store_config: bool
            if True, the input configuration is stored in the output file.
        slim_output: bool
            if True, only the necessary data will be stored in the output file to reduce its size
        """
        # starting the pipeline
        self.info("Focal plane pipeline started")
        with self.output.use(append=True, cache=True) as out:
            # store configuration
            if store_config:
                out.store_dictionary(self.mainConfig, "configuration")

            # common path and source preparation
            sources, common_path = self.prepare_environment(out)
            self.log_runtime("preparation ended time", "info")

            # instrument focal plane production
            self.create_channel_focal_planes(out, sources, common_path)

            if slim_output:
                prune_output(out, logger=self)

            self.log_runtime_complete("recipe ended", "info")

            self.info(f"output {self.output_file} size: {out.getsize():.3f}")

    def prepare_environment(self, out: Output) -> tuple[OrderedDict, OrderedDict]:
        """
        Il prepores the input data to build the instrument focal planes

        Parameters
        ----------
        out: :class:`~exosim.output.output.OutputGroup`
            output group

        Returns
        -------
        dict
            sources dict
        `~collections.OrderedDict`
            common path dictionary
        """
        out_sky = out.create_group("sky")

        sources, for_contrib = {}, {}
        parsePath = parse.ParsePath()

        if "sky" in self.mainConfig:
            sky = self.mainConfig["sky"]

            # source preparation
            if "source" in sky:
                parseSources = parse.ParseSources()
                sources = parseSources(
                    parameters=sky["source"],
                    wavelength=self.wl_grid,
                    time=self.time_grid,
                    output=out_sky,
                )

            # foreground preparation
            if "foregrounds" in sky:
                for_contrib = parsePath(
                    parameters=sky["foregrounds"],
                    wavelength=self.wl_grid,
                    time=self.time_grid,
                    output=out_sky,
                    group_name="foregrounds",
                )

        # common optics preparation
        telescope_cfg = (
            self.payloadConfig.get("Telescope")
            if isinstance(self.payloadConfig, dict)
            else None
        )
        if telescope_cfg and "optical_path" in telescope_cfg:
            common_path = parsePath(
                parameters=telescope_cfg["optical_path"],
                wavelength=self.wl_grid,
                time=self.time_grid,
                output=out,
                light_path=for_contrib,
                group_name="telescope",
            )
        else:
            common_path = for_contrib
        return sources, common_path

    def run_channel(
        self,
        description: dict,
        common_path: OrderedDict,
        sources: dict,
        pointing: tuple[u.Quantity, u.Quantity] | None = None,
        out: OutputType = None,
    ) -> None:
        """
        It instantiates and runs the :class:`~exosim.models.channel.Channel` for the indicated channel`

        Parameters
        ----------
        description: dict
            channel description
        common_path: `~collections.OrderedDict`
            dictionary of contributes
        sources:  dict
            dictionary containing :class:`~exosim.models.signal.Sed`
        pointing: (:class:`astropy.units.Quantity`, :class:`astropy.units.Quantity`) (optional)
            telescope pointing direction, expressed ad a tuple of RA and DEC in degrees. Default is ``None``
        out: :class:`~exosim.output.output.OutputGroup (optional)`
            output group

        """
        # Ensure channel has a name
        if "value" not in description:
            # Try to infer a sensible name
            description["value"] = description.get("name", "channel")
        # Provide a minimal optical path if missing
        if "optical_path" not in description:
            description["optical_path"] = {
                "opticalElement": {
                    "default": {
                        "type": "lens",
                        "value": "default_lens",
                        "throughput": 1.0,
                    }
                }
            }
        channel = Channel(
            parameters=description,
            wavelength=self.wl_grid,
            time=self.time_grid,
            output=out,
        )
        channel.parse_path(light_path=common_path)
        channel.estimate_responsivity()
        channel.propagate_foreground()
        channel.define_sources(
            sources=sources,
        )
        channel.propagate_sources(Atel=self.payloadConfig["Telescope"]["Atel"])
        channel.create_focal_planes()
        channel.rescale_contributions()
        channel.populate_focal_plane(pointing)
        # TODO test if no other stars are present
        channel.populate_bkg_focal_plane(pointing)
        if (
            "irf_task" in description["detector"]
            and "oversampling" in description["detector"]
            and description["detector"]["oversampling"] > 1
        ):
            channel.apply_irf()

        channel.populate_foreground_focal_plane()

        channel.focal_plane.write()
        channel.frg_focal_plane.write()
        if channel.bkg_focal_plane:
            channel.bkg_focal_plane.write()

        for value in channel.frg_sub_focal_planes.values():
            value.write()

    def create_channel_focal_planes(
        self, out: Output, sources: dict, common_path: OrderedDict
    ) -> None:
        """
        Create focal planes for all channels in the payload configuration.
        """
        ch_out = out.create_group("channels")

        if isinstance(self.payloadConfig["channel"], OrderedDict):
            # Multiple channels configuration
            for ch in self.payloadConfig["channel"]:
                self.announce(f"channel {ch} started")
                self.run_channel(
                    self.payloadConfig["channel"][ch],
                    common_path,
                    sources,
                    self.pointing,
                    ch_out,
                )
                self.log_runtime(f"{ch} ended in", "info")
        else:
            # Single channel configuration
            channel_config = self.payloadConfig["channel"]
            channel_name = channel_config.get("value", "single_channel")

            self.run_channel(
                channel_config,
                common_path,
                sources,
                self.pointing,
                ch_out,
            )

            self.log_runtime(f"channel {channel_name} ended time", "info")
