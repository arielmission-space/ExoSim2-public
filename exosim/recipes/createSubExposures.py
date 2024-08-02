import os
from collections import OrderedDict
from typing import Tuple
from typing import Union

import h5py

import exosim.log as log
from exosim.models.signal import CountsPerSecond
from exosim.output import HDF5OutputGroup
from exosim.output import SetOutput
from exosim.output.hdf5.utils import load_signal
from exosim.tasks.astrosignal import ApplyAstronomicalSignal
from exosim.tasks.astrosignal import FindAstronomicalSignals
from exosim.tasks.subexposures import AddForegrounds
from exosim.tasks.subexposures import ApplyQeMap
from exosim.tasks.subexposures import EstimatePointingJitter
from exosim.tasks.subexposures import InstantaneousReadOut
from exosim.tasks.subexposures import LoadILS
from exosim.tasks.subexposures import LoadQeMap
from exosim.tasks.subexposures import PrepareInstantaneousReadOut
from exosim.utils.ascii_arts import astronomer2
from exosim.utils.iterators import iterate_over_chunks
from exosim.utils.klass_factory import find_and_run_task
from exosim.utils.prepare_recipes import copy_input_files
from exosim.utils.prepare_recipes import load_options
from exosim.utils.runConfig import RunConfig
from exosim.utils.timed_class import TimedClass


class CreateSubExposures(TimedClass, log.Logger):
    """
    Pipeline to create the focal planes sub-exposures.
    This pipeline loads the configuration file and the produced focal planes to produce the sub-exposures.
    It prepares the pointing jitter first, then it scales it to the instrument focal planes pixels.
    It estimates the detector reading scheme, considering instantaneous readout,
    and it produces the sub-exposures by jittering the focal planes.

    Attributes
    ------------
    input: str
        input file
    mainConfig: dict
        This is parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
    payloadConfig: dict
        payload configuration dictionary extracted from mainConfig
    jitter_spa: :class:`~astropy.units.Quantity`
        pointing jitter in the spatial direction expressed in units of :math:`deg`.
    jitter_spe: :class:`~astropy.units.Quantity`
        pointing jitter in the spectral direction expressed in units of :math:`deg`.


    Examples
    --------

    >>> import exosim.recipes as recipes
    >>> recipes.CreateSubExposures(input_file='./input_file.h5',
    >>>                            output_file = 'se_file.h5',
    >>>                            options_file='main _configuration.xml')


    """

    def __init__(
        self, input_file: str, output_file: str, options_file: Union[str, dict]
    ) -> None:
        """
        Parameters
        ----------
        input_file: str
            input file
        output_file: str
            output file
        options_file: str or dict
            input configuration file
        """

        super().__init__()
        self.graphics(astronomer2)
        RunConfig.stats()
        self.announce("started")
        self.input = input_file
        self.mainConfig, self.payloadConfig = load_options(options_file)
        copy_input_files(os.path.dirname(os.path.abspath(output_file)))

        self.jitter_spa, self.jitter_spe, self.jitter_time = None, None, None
        output = SetOutput(output_file)

        # prepare channel list
        with h5py.File(self.input, "r") as f:
            ch_list = list(f["channels"].keys())
            ch_list.sort()

        with output.use(append=True, cache=True) as out:
            # if jitter is requested, run the jitter task
            use_slicing = False
            if "jitter" in self.mainConfig.keys():
                jitter_instance = find_and_run_task(
                    self.mainConfig["jitter"],
                    "jitter_task",
                    EstimatePointingJitter,
                )
                (
                    self.jitter_spa,
                    self.jitter_spe,
                    self.jitter_time,
                ) = jitter_instance(
                    parameters=self.mainConfig, output_file=out
                )

                if "slicing" in self.mainConfig["jitter"].keys():
                    use_slicing = self.mainConfig["jitter"]["slicing"]

            output_grp = out.create_group("channels")
            instantaneousReadOut = InstantaneousReadOut()
            prepareInstantaneousReadOut = PrepareInstantaneousReadOut()
            addForegrounds = AddForegrounds()

            for ch in ch_list:
                self.announce("computing sub-exposure for {}".format(ch))
                # prepare the output group
                ch_grp = output_grp.create_group(ch)

                # copy the simulation data from the input file to the output file
                sim_grp = self.copy_simulation_data(ch, ch_grp)

                # load the focal plane
                fp, frg_fp, bkg_fp = self.load_focal_plane(ch)

                # determine if only a channel is provided or a list of channels
                if isinstance(self.payloadConfig["channel"], OrderedDict):
                    ch_param = self.payloadConfig["channel"][ch]
                else:
                    ch_param = self.payloadConfig["channel"]

                # prepare the readout task
                (
                    readout_parameters,
                    integration_time,
                ) = prepareInstantaneousReadOut(
                    main_parameters=self.mainConfig,
                    parameters=ch_param,
                    focal_plane=fp,
                    pointing_jitter=(
                        self.jitter_spa,
                        self.jitter_spe,
                        self.jitter_time,
                    ),
                    output_file=ch_grp,
                )

                # run the readout task
                out_se = instantaneousReadOut(
                    parameters=ch_param,
                    focal_plane=fp,
                    readout_parameters=readout_parameters,
                    pointing_jitter=(
                        self.jitter_spa,
                        self.jitter_spe,
                        self.jitter_time,
                    ),
                    slicing=use_slicing,
                    output_file=ch_grp,
                )

                # introducing the astronomical signal
                ils, timeline = None, None

                findAstronomicalSignals = FindAstronomicalSignals()
                applyAstronomicalSignal = ApplyAstronomicalSignal()
                astrosignals = findAstronomicalSignals(
                    sky_parameters=self.mainConfig["sky"],
                )
                for source, signals in astrosignals.items():
                    astro_grp = ch_grp.create_group("Astrosignal")
                    source_grp = astro_grp.create_group(source)
                    # TODO what if different signals are found for the same source with the same name?
                    for signal, params in signals.items():
                        signal_grp = source_grp.create_group(signal)
                        self.info(
                            "{}-{} astronomical signal found.".format(
                                source, signal
                            )
                        )

                        # load the instrument line shape
                        ils_instance = find_and_run_task(
                            ch_param["detector"], "ils_task", LoadILS
                        )
                        ils = (
                            ils_instance(
                                input_file=self.input,
                                ch_param=ch_param,
                                wl_grid=fp.spectral * fp.spectral_units,
                            )
                            if ils is None
                            else ils
                        )

                        # prepare the timeline for the astronomical signal as the middle of each sub-exposure
                        timeline = (
                            (
                                out_se.time * out_se.time_units
                                + integration_time / 2
                            )
                            if timeline is None
                            else timeline
                        )

                        # estimating the signal model
                        signal_task = params["task"]()
                        model_timeline, signal_model = signal_task(
                            timeline=timeline,
                            wl_grid=fp.spectral * fp.spectral_units,
                            ch_parameters=ch_param,
                            source_parameters=params["parsed_parameters"],
                            output=signal_grp,
                        )

                        # apply the signal model to the sub-exposures
                        out_se = applyAstronomicalSignal(
                            model=signal_model,
                            subexposures=out_se,
                            focal_plane=fp,
                            ils=ils,
                            timeline=model_timeline,
                            source=params,
                            ch_parameters=ch_param,
                            pointing=(
                                self.mainConfig["pointing"]["ra"],
                                self.mainConfig["pointing"]["dec"],
                            ),
                        )

                # adding background stars if any
                add_background_to_se = False
                if "add_background_to_se" in ch_param["detector"].keys():
                    if ch_param["detector"]["add_background_to_se"]:
                        add_background_to_se = True
                else:
                    if bkg_fp:
                        self.warning(
                            "No indication for adding background found. Adding them by default."
                        )
                        add_background_to_se = True

                if bkg_fp and add_background_to_se:
                    bkg_se = instantaneousReadOut(
                        parameters=ch_param,
                        focal_plane=bkg_fp,
                        readout_parameters=readout_parameters,
                        pointing_jitter=(
                            self.jitter_spa,
                            self.jitter_spe,
                            self.jitter_time,
                        ),
                        dataset_name="sub_exposures_bkg",
                        output_file=ch_grp,
                    )

                    for chunk in iterate_over_chunks(
                        out_se.dataset, desc="adding background stars"
                    ):
                        out_se.dataset[chunk] += bkg_se.dataset[chunk]
                        out_se.output.flush()

                # add the foregrounds to the sub-exposures
                try:
                    if ch_param["detector"]["add_foregrounds_to_se"]:
                        out_se = addForegrounds(
                            subexposures=out_se,
                            frg_focal_plane=frg_fp,
                            integration_time=integration_time,
                        )
                except KeyError:
                    self.warning(
                        "No indication for adding foregrounds found. Adding them by default."
                    )
                    out_se = addForegrounds(
                        subexposures=out_se,
                        frg_focal_plane=frg_fp,
                        integration_time=integration_time,
                    )

                # apply the QE map if present
                try:
                    qe_map_instance = find_and_run_task(
                        ch_param["detector"], "qe_map_task", LoadQeMap
                    )
                    qe_map = qe_map_instance(
                        parameters=ch_param,
                        time=fp.time * fp.time_units,
                        output=sim_grp,
                    )
                    applyQEMap = ApplyQeMap()
                    out_se = applyQEMap(subexposures=out_se, qe_map=qe_map)
                except KeyError:
                    self.warning(
                        "No quantum efficiency variation map detected"
                    )
                # store the results in output
                out_se.write()

                self.log_runtime("{} ended in".format(ch), "info")

            self.info(
                "output {} size: {:.3f}".format(output_file, out.getsize())
            )
        self.log_runtime_complete("recipe ended", "info")
        self.announce("ended")

    def load_focal_plane(
        self, ch: str
    ) -> Tuple[CountsPerSecond, CountsPerSecond]:
        """
        It loads the channel focal plane from the input file:

        Parameters
        ----------
        ch: str
            channel name

        Returns
        -------
        :class:`~exosim.models.signal.CountsPerSecond`
            focal plane
        :class:`~exosim.models.signal.CountsPerSecond`
            foreground focal plane
        :class:`~exosim.models.signal.CountsPerSecond`
            bkg focal plane"""
        with h5py.File(self.input, "r") as f:
            file_path = os.path.join("channels", ch)
            fp = load_signal(f[os.path.join(file_path, "focal_plane")])
            frg_fp = load_signal(f[os.path.join(file_path, "frg_focal_plane")])
            try:
                bkg_fp = load_signal(
                    f[os.path.join(file_path, "bkg_focal_plane")]
                )
            except KeyError:
                bkg_fp = None

        self.debug("focal planes loaded from {}".format(ch))

        return fp, frg_fp, bkg_fp

    def load_source(self, ch: str, source: str) -> Tuple[CountsPerSecond]:
        """
        It loads the channel focal plane from the input file:

        Parameters
        ----------
        ch: str
            channel name
        source: str
            source name

        Returns
        -------
        :class:`~exosim.models.signal.CountsPerSecond`
            source
        """
        with h5py.File(self.input, "r") as f:
            file_path = os.path.join("channels", ch)
            path = os.path.join(file_path, "sources")
            source = load_signal(f[os.path.join(path, source)])

        self.debug("{} source loaded from {}".format(source, ch))

        return source

    def copy_simulation_data(
        self, ch: str, ch_grp: HDF5OutputGroup
    ) -> HDF5OutputGroup:
        """
        It copies relevant data from the input file into the output one.
        The copied data are: `info`, `qe_map`, `efficiency` and `responsivity`.

        Parameters
        ----------
        ch: str
            channel name
        ch_grp: :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`
            output file

        Returns
        --------
        :class:`~exosim.output.hdf5.hdf5.HDF5OutputGroup`
            output group

        """
        sim_grp = ch_grp.create_group("simulation_data")

        with h5py.File(self.input, "r") as f:
            # copy the original data info
            f.copy("info", sim_grp._entry, name="focal_plane_info")

            # copy simulation data
            file_path = os.path.join("channels", ch)
            for info in ["efficiency", "responsivity"]:
                try:
                    f.copy(os.path.join(file_path, info), sim_grp._entry)
                    self.debug("{} data copied to output".format(info))
                except KeyError:
                    continue
        return sim_grp
