import os
from collections import OrderedDict
from typing import Union

import astropy.units as u
import numpy as np
from astropy.table import hstack
from astropy.table import QTable
from astropy.table import vstack

import exosim.log as log
import exosim.recipes as recipes
import exosim.tasks.radiometric as radiometric
from exosim.output import SetOutput
from exosim.utils.ascii_arts import astronomer4
from exosim.utils.klass_factory import find_task
from exosim.utils.prepare_recipes import load_options
from exosim.utils.runConfig import RunConfig
from exosim.utils.timed_class import TimedClass


class RadiometricModel(TimedClass, log.Logger):
    """

    Pipeline to create the radiometric model.
    This pipeline has three working modes:

    - it can load an already produced focal plane and use it to estimate a radiometric model;
    - it can produce a single sourec focal planet and estimate the radiometric model;
    - it can load a target list and produce the radiometric model for each target of the target list.

    Attributes
    ------------
    mainConfig: dict
        This is parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
    input: :class:`~exosim.output.output.Output`
        input/output file
    payloadConfig: dict
        payload configuration dictionary extracted from mainConfig`
    table: :class:`~astropy.table.QTable`
        table for the radiometric estimations

    Examples
    --------

    If the user wants to estimate the radiometric model af an existing focal plane

    >>> import exosim.recipes as recipes
    >>> rm = recipes.RadiometricModel(options_file= 'main _configuration.xml',
    >>>                                input_file = 'focal_plane.h5')

    Otherwhise, if a focal planet has not been produced yet, this recipy can produce it,
    if a destination not existing file is provided:

    >>> import exosim.recipes as recipes
    >>> rm = recipes.RadiometricModel(options_file= 'main _configuration.xml',
    >>>                                input_file = 'desired_output.h5')

    In both cases, to store the produced table into the output file,
    the :func:`~exosim.recipes.radiometricModel.RadiometricModel.write`
    is to be used:

    >>> rm.write()

    """

    def __init__(
        self, options_file: Union[str, dict], input_file: str
    ) -> None:
        """

        Parameters
        ----------
        options_file: str or dict
            input configuration file
        input_file: str
            input file
        """
        super().__init__()

        self.graphics(astronomer4)
        RunConfig.stats()
        self.announce("started")

        self.mainConfig, self.payloadConfig = load_options(options_file)

        self.table = None

        # if focal plane already exists it loads it
        if os.path.isfile(input_file):
            self.input = SetOutput(input_file, replace=False)
            self.info("Running radiometric model on existing focal plane")
            self.single_file_pipeline()
        # if focal plane doesn't exist it create it for a single step
        elif input_file.endswith("h5"):
            self.info(
                "Focal plane not found: running CreateFocalPlane pipeline"
            )
            # single time step
            self.mainConfig["time_grid"] = {
                "start_time": 0 * u.hr,
                "end_time": 1 * u.hr,
                "low_frequencies_resolution": 1 * u.hr,
            }
            self._isolate_every_opt()
            recipes.CreateFocalPlane(
                options_file=self.mainConfig, output_file=input_file
            )
            self.input = SetOutput(input_file, replace=False)
            self.single_file_pipeline()

        # if input is a target list, it creates the focal plane only for
        # instruments and foregrounds
        elif input_file.endswith(".csv"):
            raise NotImplementedError
            # # single time step
            # self.mainConfig['time_grid'] = {'start_time': 0 * u.hr,
            #                                 'end_time': 1 * u.hr,
            #                                 'low_frequencies_resolution': 1 * u.hr}
            # # remove source
            # if 'source' in self.mainConfig['sky'].keys():
            #     self.mainConfig['sky'].pop('source')
            # self._isolate_every_opt()
            # recipes.CreateFocalPlane(options_file=self.mainConfig,
            #                          output_file=input_file)

        self.common_pipeline()
        self.write()
        self.log_runtime_complete("recipe ended", "info")
        self.announce("ended")

    def _isolate_every_opt(self) -> None:
        """
        it iterates over the optical elements to isolate them and craete sub
        focal planes
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

        1. creation of the wavelength table with :func:`~exosim.recipes.radiometricModel.RadiometricModel.create_table`;
        2. estimation of the apertures sizes and number of pixels involved with :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_apertures`;
        3. estimation of the signals in the apertures for the sub foregrounds, if any: :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_sub_foregrounds_signals`;
        4. estimation of the total foreground signal in the apertures:  :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_foreground_signals`;
        5. estimation of the source focal plane signal in the aperture: :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_source_signals`;
        6. estimation of the saturation time in the channel: :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_saturation`;

        The pipeline will update the `table` attribute.
        """
        self.table = self.create_table()
        self.compute_apertures()
        self.compute_sub_foregrounds_signals()
        self.compute_foreground_signals()
        self.compute_source_signals()
        self.compute_saturation()

    def common_pipeline(self) -> None:
        """
        Radiometric pipeline to run starting from a radiometric table with already estimated signals.
        It computes the noise.

        1. estimation of the multiaccum factors :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_multiaccum`;
        2. estimation shot noise :func:`~exosim.recipes.radiometricModel.RadiometricModel.compute_photon_noise`;
        3. update total noise :func:`~exosim.recipes.radiometricModel.RadiometricModel.update_total_noise`

        The pipeline will update the `table` attribute.
        """
        self.compute_multiaccum()
        self.compute_photon_noise()
        self.update_total_noise()

    def create_table(self) -> QTable:
        """
        Produces the starting radiometric table with the spectral bins and their edges.
        It is based on :class:`~exosim.tasks.radiometric.estimateSpectralBinning.EstimateSpectralBinning` by default.

        Returns
        -------
        :class:`~astropy.table.QTable`
            table for the radiometric estimations with wavelength grid.
        """

        self.info("Computing radiometric signals")
        estimateSpectralBinning_task = (
            find_task(
                self.payloadConfig["channel"]["spectral_binning_task"],
                radiometric.EstimateSpectralBinning,
            )
            if "spectral_binning_task" in self.payloadConfig["channel"].keys()
            else radiometric.EstimateSpectralBinning
        )
        estimateSpectralBinning = estimateSpectralBinning_task()
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            table_list = [
                estimateSpectralBinning(
                    parameters=self.payloadConfig["channel"][ch]
                )
                for ch in self.payloadConfig["channel"].keys()
            ]
        else:
            table_list = [
                estimateSpectralBinning(
                    parameters=self.payloadConfig["channel"]
                )
            ]
        return vstack(table_list)

    def compute_apertures(self) -> QTable:
        """
        Estimates the photometric aperture for each spectral bin
        using :class:`~exosim.tasks.radiometric.estimateApertures.EstimateApertures` by default.

        Returns
        -------
        :class:`~astropy.table.QTable`
            table with the apertures for each channel and bin
        """
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            table_list = []
            for ch in self.payloadConfig["channel"].keys():
                self.info("estimating apertures on {}".format(ch))
                description = self.payloadConfig["channel"][ch]
                estimateApertures_task = (
                    find_task(
                        description["radiometric"]["aperture_photometry"][
                            "apertures_task"
                        ],
                        radiometric.EstimateApertures,
                    )
                    if "apertures_task"
                    in description["radiometric"]["aperture_photometry"].keys()
                    else radiometric.EstimateApertures
                )
                estimateApertures = estimateApertures_task()

                with self.input.open() as f:
                    f = f["channels"]
                    self.debug("extracting wavelength grid")
                    osf = f[ch]["focal_plane"]["metadata"]["oversampling"][()]
                    focal_plane = f[ch]["focal_plane"]["data"][
                        0, osf // 2 :: osf, osf // 2 :: osf
                    ]
                    wl_grid = f[ch]["focal_plane"]["spectral"][
                        osf // 2 :: osf
                    ] * u.Unit(f[ch]["focal_plane"]["spectral_units"][()])

                    table_list += [
                        estimateApertures(
                            table=self.table[self.table["ch_name"] == ch],
                            focal_plane=focal_plane,
                            description=description["radiometric"][
                                "aperture_photometry"
                            ],
                            wl_grid=wl_grid,
                        )
                    ]
        else:
            description = self.payloadConfig["channel"]
            self.info(
                "estimating apertures on {}".format(description["value"])
            )
            estimateApertures_task = (
                find_task(
                    description["radiometric"]["aperture_photometry"][
                        "apertures_task"
                    ],
                    radiometric.EstimateApertures,
                )
                if "apertures_task"
                in description["radiometric"]["aperture_photometry"].keys()
                else radiometric.EstimateApertures
            )
            estimateApertures = estimateApertures_task()
            with self.input.open() as f:
                f = f["channels"]
                ch = list(f.keys())[0]
                self.debug("extracting wavelength grid")
                osf = f[ch]["focal_plane"]["metadata"]["oversampling"][()]
                focal_plane = f[ch]["focal_plane"]["data"][
                    0, osf // 2 :: osf, osf // 2 :: osf
                ]
                wl_grid = f[ch]["focal_plane"]["spectral"][
                    osf // 2 :: osf
                ] * u.Unit(f[ch]["focal_plane"]["spectral_units"][()])

                table_list = [
                    estimateApertures(
                        table=self.table,
                        focal_plane=focal_plane,
                        wl_grid=wl_grid,
                        description=description["radiometric"][
                            "aperture_photometry"
                        ],
                    )
                ]

        stack = vstack(table_list)
        self.table = hstack((self.table, stack))
        return hstack((self.table["ch_name", "Wavelength"], stack))

    def write(self, output_file: str = None) -> None:
        """
        It adds the radiometric table to the output.
        If the table exists already in the output file, it replaces it.

        Parameters
        ----------
        output_file: str (optional)
            output file. Default is `input`

        """
        output = (
            self.input
            if output_file is None
            else SetOutput(output_file, replace=False)
        )

        with output.use(append=True) as out:
            self.info("radiometric table stored in {}".format(output.fname))
            rad_out = out.create_group("radiometric")
            rad_out.write_table("table", self.table, replace=True)

    def compute_sub_foregrounds_signals(self) -> QTable:
        """
        It estimates the radiometric signals on the foreground sub focal planes for all the
        channels and returns a table with all the contributions.

        It uses :class:`~exosim.tasks.radiometric.computeSubFrgSignalsChannel.ComputeSubFrgSignalsChannel` by default.

        Returns
        -------
        astropy.table.QTable:
            signal table
        """

        table_list = []
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            for ch in self.payloadConfig["channel"].keys():
                self.info(
                    "estimating sub-foreground radiometric signal for {}".format(
                        ch
                    )
                )
                computeFrgSignalsChannel_task = (
                    find_task(
                        self.payloadConfig["channel"][ch]["radiometric"][
                            "sub_frg_signal_task"
                        ],
                        radiometric.ComputeSubFrgSignalsChannel,
                    )
                    if "sub_frg_signal_task"
                    in self.payloadConfig["channel"][ch]["radiometric"].keys()
                    else radiometric.ComputeSubFrgSignalsChannel
                )
                computeFrgSignalsChannel = computeFrgSignalsChannel_task()
                table_list += [
                    computeFrgSignalsChannel(
                        table=self.table[self.table["ch_name"] == ch],
                        ch_name=ch,
                        input_file=self.input,
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
                    self.payloadConfig["channel"]["radiometric"][
                        "sub_frg_signal_task"
                    ],
                    radiometric.ComputeSubFrgSignalsChannel,
                )
                if "sub_frg_signal_task"
                in self.payloadConfig["channel"]["radiometric"].keys()
                else radiometric.ComputeSubFrgSignalsChannel
            )
            computeFrgSignalsChannel = computeFrgSignalsChannel_task()
            table_list = [
                computeFrgSignalsChannel(
                    table=self.table,
                    ch_name=self.payloadConfig["channel"]["value"],
                    input_file=self.input,
                )
            ]

        stack = vstack(table_list)
        self.table = hstack((self.table, stack))
        for k in self.table.keys():
            if hasattr(self.table[k], "filled"):
                self.table[k] = self.table[k].filled(0.0)
        ret_k = ["ch_name", "Wavelength"] + list(stack.keys())
        return self.table[ret_k], stack

    def compute_foreground_signals(self) -> QTable:
        """
        It estimates the radiometric signals on the foreground focal plane for all the
        channels and returns a table with all the contributions.

        It uses :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` by default.

        Returns
        -------
        astropy.table.QTable:
            signal table
        """

        phot = np.array([])
        with self.input.open() as f:
            if isinstance(self.payloadConfig["channel"], OrderedDict):
                for ch in self.payloadConfig["channel"].keys():
                    self.info(
                        "estimating foreground radiometric signal for {}".format(
                            ch
                        )
                    )
                    computeSignalsChannel_task = (
                        find_task(
                            self.payloadConfig["channel"][ch]["radiometric"][
                                "signal_task"
                            ],
                            radiometric.ComputeSignalsChannel,
                        )
                        if "signal_task"
                        in self.payloadConfig["channel"][ch][
                            "radiometric"
                        ].keys()
                        else radiometric.ComputeSignalsChannel
                    )
                    computeSignalsChannel = computeSignalsChannel_task()

                    focal_plane = f["channels"][ch]["frg_focal_plane"]
                    phot_ = computeSignalsChannel(
                        table=self.table[self.table["ch_name"] == ch],
                        focal_plane=focal_plane,
                    )
                    phot = np.concatenate((phot, phot_))
            else:
                self.info(
                    "estimating foreground radiometric signal for {}".format(
                        self.payloadConfig["channel"]["value"]
                    )
                )
                computeSignalsChannel_task = (
                    find_task(
                        self.payloadConfig["channel"]["radiometric"][
                            "signal_task"
                        ],
                        radiometric.ComputeSignalsChannel,
                    )
                    if "signal_task"
                    in self.payloadConfig["channel"]["radiometric"].keys()
                    else radiometric.ComputeSignalsChannel
                )
                computeSignalsChannel = computeSignalsChannel_task()

                ch = list(f["channels"].keys())[0]
                focal_plane = f["channels"][ch]["frg_focal_plane"]
                phot_ = computeSignalsChannel(
                    table=self.table, focal_plane=focal_plane
                )
                phot = np.concatenate((phot, phot_))

        self.table["foreground_signal_in_aperture"] = phot
        return self.table[
            "ch_name", "Wavelength", "foreground_signal_in_aperture"
        ]

    def compute_source_signals(self) -> QTable:
        """
        It estimates the radiometric signals on the source focal plane for all the
        channels and returns a table with all the contributions.

        It uses :class:`~exosim.tasks.radiometric.computeSignalsChannel.ComputeSignalsChannel` by default.

        Returns
        -------
        astropy.table.QTable:
            signal table
        """

        phot = np.array([])
        with self.input.open() as f:
            if isinstance(self.payloadConfig["channel"], OrderedDict):
                for ch in self.payloadConfig["channel"].keys():
                    self.info(
                        "estimating source radiometric signal for {}".format(
                            ch
                        )
                    )
                    computeSignalsChannel_task = (
                        find_task(
                            self.payloadConfig["channel"][ch]["radiometric"][
                                "signal_task"
                            ],
                            radiometric.ComputeSignalsChannel,
                        )
                        if "signal_task"
                        in self.payloadConfig["channel"][ch][
                            "radiometric"
                        ].keys()
                        else radiometric.ComputeSignalsChannel
                    )
                    computeSignalsChannel = computeSignalsChannel_task()

                    focal_plane = f["channels"][ch]["focal_plane"]
                    phot_ = computeSignalsChannel(
                        table=self.table[self.table["ch_name"] == ch],
                        focal_plane=focal_plane,
                    )
                    phot = np.concatenate((phot, phot_))

            else:
                self.info(
                    "estimating source radiometric signal for {}".format(
                        self.payloadConfig["channel"]["value"]
                    )
                )
                computeSignalsChannel_task = (
                    find_task(
                        self.payloadConfig["channel"]["radiometric"][
                            "signal_task"
                        ],
                        radiometric.ComputeSignalsChannel,
                    )
                    if "signal_task"
                    in self.payloadConfig["channel"]["radiometric"].keys()
                    else radiometric.ComputeSignalsChannel
                )
                computeSignalsChannel = computeSignalsChannel_task()

                ch = list(f["channels"].keys())[0]
                focal_plane = f["channels"][ch]["focal_plane"]
                phot_ = computeSignalsChannel(
                    table=self.table, focal_plane=focal_plane
                )
                phot = np.concatenate((phot, phot_))

        self.table["source_signal_in_aperture"] = phot
        return self.table["ch_name", "Wavelength", "source_signal_in_aperture"]

    def compute_saturation(self) -> QTable:
        """
        It computes and adds the saturation time to the radiometric table

        Returns
        -------
        astropy.table.QTable:
            saturation table
        """
        saturations, integration_time, max_signal_in_bin, min_signal_in_bin = (
            [],
            [],
            [],
            [],
        )

        saturationChannel = radiometric.SaturationChannel()
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            for ch in self.payloadConfig["channel"].keys():
                sat, t_int, max_, min_ = saturationChannel(
                    table=self.table,
                    description=self.payloadConfig["channel"][ch],
                    input_file=self.input,
                )
                saturations += sat
                integration_time += t_int
                max_signal_in_bin += max_
                min_signal_in_bin += min_
        else:
            sat, t_int, max_, min_ = saturationChannel(
                table=self.table,
                description=self.payloadConfig["channel"],
                input_file=self.input,
            )
            saturations += sat
            integration_time += t_int
            max_signal_in_bin += max_
            min_signal_in_bin += min_

        self.table["saturation_time"] = saturations
        self.table["integration_time"] = integration_time
        self.table["max_signal_in_bin"] = max_signal_in_bin
        self.table["min_signal_in_bin"] = min_signal_in_bin

        return self.table[
            "ch_name",
            "Wavelength",
            "saturation_time",
            "integration_time",
            "max_signal_in_bin",
            "min_signal_in_bin",
        ]

    def compute_multiaccum(self) -> QTable:
        """
        It estimates the multiaccum gain factors using :class:`~exosim.tasks.radiometric.multiaccum.Multiaccum`. The

        Returns
        -------
        astropy.table.QTable:
           multiaccum factors
        """

        read_gain, shot_gain = [], []

        multiaccum = radiometric.Multiaccum()
        if isinstance(self.payloadConfig["channel"], OrderedDict):
            for ch in self.payloadConfig["channel"].keys():
                if (
                    "multiaccum"
                    in self.payloadConfig["channel"][ch]["radiometric"].keys()
                ):
                    read, shot = multiaccum(
                        parameters=self.payloadConfig["channel"][ch][
                            "radiometric"
                        ]["multiaccum"]
                    )
                    read_gain += [read] * len(
                        self.table[self.table["ch_name"] == ch]
                    )
                    shot_gain += [shot] * len(
                        self.table[self.table["ch_name"] == ch]
                    )
                else:
                    read_gain += [1] * len(
                        self.table[self.table["ch_name"] == ch]
                    )
                    shot_gain += [1] * len(
                        self.table[self.table["ch_name"] == ch]
                    )

        else:
            if (
                "multiaccum"
                in self.payloadConfig["channel"]["radiometric"].keys()
            ):
                read, shot = multiaccum(
                    parameters=self.payloadConfig["channel"]["radiometric"][
                        "multiaccum"
                    ]
                )
                read_gain += [read] * len(self.table)
                shot_gain += [shot] * len(self.table)
            else:
                read_gain += [1] * len(self.table)
                shot_gain += [1] * len(self.table)

        self.table["multiaccum_read_gain"] = read_gain
        self.table["multiaccum_shot_gain"] = shot_gain
        return self.table[
            "ch_name",
            "Wavelength",
            "multiaccum_read_gain",
            "multiaccum_shot_gain",
        ]

    def compute_photon_noise(self) -> QTable:
        """
        It computes and adds the photon noise to the radiometric table using :class:`~exosim.tasks.radiometric.computePhotonNoise.ComputePhotonNoise`.

        Returns
        -------
        astropy.table.QTable:
           photon noise
        """

        computePhotonNoise = radiometric.ComputePhotonNoise()
        signals = [k for k in self.table.keys() if "_signal_in_aperture" in k]
        for sig in signals:
            phot_noise = []
            for i in range(len(self.table["ch_name"])):
                ch_description = (
                    self.payloadConfig["channel"][self.table["ch_name"][i]]
                    if isinstance(self.payloadConfig["channel"], OrderedDict)
                    else self.payloadConfig["channel"]
                )
                phot_noise += [
                    computePhotonNoise(
                        signal=self.table[sig][i],
                        description=ch_description,
                        multiaccum_gain=self.table["multiaccum_shot_gain"][i],
                    )
                ]
            name = sig.replace("_signal_in_aperture", "")

            phot_noise = (
                np.array([p.value for p in phot_noise]) * phot_noise[0].unit
            )
            self.table["{}_photon_noise".format(name)] = phot_noise

        return self.table[
            "ch_name",
            "Wavelength",
            "source_photon_noise",
            "foreground_photon_noise",
        ]

    def update_total_noise(self) -> QTable:
        """Updates the total noise column in the radiometric table."""
        computeTotalNoise = radiometric.ComputeTotalNoise()
        total_noise = computeTotalNoise(table=self.table)
        self.table["total_noise"] = total_noise
        return self.table["ch_name", "Wavelength", "total_noise"]
