import os.path
from collections import OrderedDict
from typing import Tuple
from typing import Union

import astropy.units as u

import exosim.log as log
import exosim.tasks.parse as parse
import exosim.utils as utils
from exosim.models.channel import Channel
from exosim.output import Output
from exosim.output import SetOutput
from exosim.utils.ascii_arts import astronomer1
from exosim.utils.prepare_recipes import copy_input_files
from exosim.utils.prepare_recipes import load_options
from exosim.utils.runConfig import RunConfig
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
        This is parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
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
        options_file: Union[str, dict],
        output_file: str,
        store_config: bool = False,
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
        """
        super().__init__()

        self.graphics(astronomer1)
        RunConfig.stats()
        self.announce("started")
        self.mainConfig, self.payloadConfig = load_options(options_file)

        if output_file is not None:
            copy_input_files(os.path.dirname(os.path.abspath(output_file)))

        self.output = SetOutput(output_file)

        # wavelength and time grid definition
        self.wl_grid = utils.grids.wl_grid(
            self.mainConfig["wl_grid"]["wl_min"],
            self.mainConfig["wl_grid"]["wl_max"],
            self.mainConfig["wl_grid"]["logbin_resolution"],
        )

        if "time_grid" in self.mainConfig.keys():
            self.time_grid = utils.grids.time_grid(
                self.mainConfig["time_grid"]["start_time"],
                self.mainConfig["time_grid"]["end_time"],
                self.mainConfig["time_grid"]["low_frequencies_resolution"],
            )
        else:
            self.time_grid = [0] * u.hr

        if "pointing" in self.mainConfig.keys():
            pointing = (
                self.mainConfig["pointing"]["ra"],
                self.mainConfig["pointing"]["dec"],
            )
        else:
            pointing = None

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
            ch_out = out.create_group("channels")
            if isinstance(self.payloadConfig["channel"], OrderedDict):
                for ch in self.payloadConfig["channel"].keys():
                    self.announce("channel {} started".format(ch))
                    self.run_channel(
                        self.payloadConfig["channel"][ch],
                        common_path,
                        sources,
                        pointing,
                        ch_out,
                    )
                    self.log_runtime("{} ended in".format(ch), "info")

            # else, we load the only source available
            else:
                self.run_channel(
                    self.payloadConfig["channel"],
                    common_path,
                    sources,
                    pointing,
                    ch_out,
                )

                self.log_runtime(
                    "channel {} ended time".format(
                        self.payloadConfig["channel"]["value"]
                    ),
                    "info",
                )

            self.log_runtime_complete("recipe ended", "info")

            self.info(
                "output {} size: {:.3f}".format(output_file, out.getsize())
            )

        self.announce("ended")

    def prepare_environment(
        self, out: Output
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        Il prepores the input data to build the instrument focal planes

        Parameters
        ----------
        out: :class:`~exosim.output.output.OutputGroup`
            output group

        Returns
        -------
        dict
            soureces dict
        `~collections.OrderedDict`
            common path dictionary
        """
        out_sky = out.create_group("sky")

        sources, for_contrib = {}, {}
        parsePath = parse.ParsePath()

        if "sky" in self.mainConfig.keys():
            sky = self.mainConfig["sky"]

            # source preparation
            if "source" in sky.keys():
                parseSources = parse.ParseSources()
                sources = parseSources(
                    parameters=sky["source"],
                    wavelength=self.wl_grid,
                    time=self.time_grid,
                    output=out_sky,
                )

            # foreground preparation
            if "foregrounds" in sky.keys():
                for_contrib = parsePath(
                    parameters=sky["foregrounds"],
                    wavelength=self.wl_grid,
                    time=self.time_grid,
                    output=out_sky,
                    group_name="foregrounds",
                )

        # common optics preparation
        if "optical_path" in self.payloadConfig["Telescope"].keys():
            common_path = parsePath(
                parameters=self.payloadConfig["Telescope"]["optical_path"],
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
        pointing: Tuple[u.Quantity, u.Quantity] = None,
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
        channel = Channel(
            parameters=description,
            wavelength=self.wl_grid,
            time=self.time_grid,
            output=out,
        )
        channel.parse_path(light_path=common_path)
        channel.estimate_responsivity()
        channel.propagate_foreground()
        channel.propagate_sources(
            sources=sources, Atel=self.payloadConfig["Telescope"]["Atel"]
        )
        channel.create_focal_planes()
        channel.rescale_contributions()
        channel.populate_focal_plane(pointing)
        # TODO test if no other stars are present
        channel.populate_bkg_focal_plane(pointing)
        if "irf_task" in description["detector"].keys():
            channel.apply_irf()

        channel.populate_foreground_focal_plane()

        channel.focal_plane.write()
        channel.frg_focal_plane.write()
        if channel.bkg_focal_plane:
            channel.bkg_focal_plane.write()

        for key, value in channel.frg_sub_focal_planes.items():
            value.write()
